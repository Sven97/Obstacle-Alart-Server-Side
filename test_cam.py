from __future__ import absolute_import, division, print_function

import time
import cv2
import os
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
from multiprocessing import Process, Pipe

import torch
from torchvision import transforms, datasets

import networks
from layers import disp_to_depth
from utils import download_model_if_doesnt_exist


# Runs inference on image (function run in a separate process)
def infer_image(conn, input_encoder, input_depth_decoder):
    while True:
        try:
            # Receive image tensor and its dimensions
            image, original_height_img, original_width_img = conn.recv()
        except EOFError:
            break

        # Run prediction
        features_img = input_encoder(image)
        outputs_img = input_depth_decoder(features_img)
        disp_img = outputs_img[("disp", 0)]

        # Resize disparity map back to original image size
        disp_resized_img = torch.nn.functional.interpolate(disp_img, (original_height_img, original_width_img),
                                                           mode="nearest")
        # Calculate depth and convert to numpy array
        _, pred_depth_img = disp_to_depth(disp_resized_img, 0.1, 100)
        pred_depth_img = pred_depth_img.squeeze().cpu().detach().numpy()

        # Convert pred_depth_img (gives depth per pixel) into 3x4 grid by calculating average value of each 160x160
        # pixel square
        # TODO: Look into dividing into more squares
        depth_map_img = np.zeros([3, 4])
        for i in range(len(depth_map_img)):
            for j in range(len(depth_map_img[0])):
                depth_map_img[i][j] = get_avg_depth(pred_depth_img, 160 * i, 160 * j, 160 * i + 160, 160 * j + 160)

        # Convert to numpy array
        disp_resized_img = disp_resized_img.squeeze().cpu().detach().numpy()

        # Send back the depth map and disparity map
        conn.send((depth_map_img, disp_resized_img))


def test_cam():
    """Function to predict for a camera image stream
    """

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    download_model_if_doesnt_exist("mono+stereo_640x192")
    model_path = os.path.join("models", "mono+stereo_640x192")
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()

    print("-> Prediction is initialized")

    # Create pipe between connection for the process running infer_image and the current process
    # child_conn responsible for receiving images and sending back depth map and disparity map
    # parent_conn responsible for sending images and receiving depth map and disparity map
    parent_conn, child_conn = Pipe()
    p = Process(target=infer_image, args=(child_conn, encoder, depth_decoder))
    p.start()

    # Initialize webcam to capture image stream
    # Change the value to 0 when using default camera
    cap = cv2.VideoCapture(0)

    # PREDICTING ON CAMERA IMAGE STREAM
    depth_map = None
    disp_resized_np = None
    total_frames = 0
    num_frames = 5
    curr_time = np.zeros(num_frames)
    time_start = time.time()
    with torch.no_grad():
        while cap.isOpened():

            # Capture frame-by-frame
            ret, frame = cap.read()

            # Calculate fps
            curr_time[1:] = curr_time[:-1]
            curr_time[0] = time.time()
            fps = num_frames / (curr_time[0] - curr_time[len(curr_time) - 1])

            # Our operations on the frame come here
            input_image = pil.fromarray(frame).convert('RGB')
            original_width, original_height = input_image.size
            input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)
            input_image = input_image.to(device)

            # Send image and its dimensions to child connection
            parent_conn.send((input_image, original_height, original_width))

            # Determine warning by dividing 3x4 grid into columns (1 for LEFT, 2 in center for AHEAD, 1 for RIGHT)
            if disp_resized_np is not None and depth_map is not None:
                if depth_map[0, 1] <= 1 or depth_map[1, 1] <= 1 or depth_map[0, 2] <= 1 or depth_map[1, 2] <= 1:
                    if depth_map[1, 1] <= 1 and depth_map[1, 2] <= 1:
                        print("Dangerous!!! AHEAD")
                    else:
                        if depth_map[0, 1] <= 1 or depth_map[1, 1] <= 1:
                            print("Dangerous!!! LEFT")
                        if depth_map[0, 2] <= 1 or depth_map[1, 2] <= 1:
                            print("Dangerous!!! RIGHT")
                elif np.sum(depth_map[0:2, 2:3]) <= 7 or np.sum(depth_map[0:2, 2:3]) <= 7:  # TODO: Test threshold value
                    if np.sum(depth_map[0:2, 0:1]) <= 7:
                        print("Careful!! LEFT")
                    if np.sum(depth_map[0:2, 2:3]) <= 7:
                        print("Careful!! RIGHT")
                else:
                    print("Clear")

                # Create image for displaying disparity image
                normalizer = mpl.colors.Normalize(vmin=0, vmax=0.5)
                mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
                colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
                im = pil.fromarray(colormapped_im)
                result_img = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)

                # Blend original and result frames
                alpha = 0.2
                beta = 1.0 - alpha
                blended_result = cv2.addWeighted(frame, alpha, result_img, beta, 0.0)

                # Display the resulting frame
                cv2.putText(blended_result, str(fps), (30, 30), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255))
                cv2.imshow('Result', result_img)
                cv2.imshow('Original', frame)

                cv2.imshow('Blended Result', blended_result)

            # Receive depth map and disparity map
            depth_map, disp_resized_np = parent_conn.recv()

            total_frames = total_frames + 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print('-> Done!')
                parent_conn.close()
                child_conn.close()
                break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    time_end = time.time()
    print(f"Average time per frame: {(time_end - time_start)/total_frames} seconds")


# TODO: Trim the box
def get_avg_depth(depth, left, top, right, bottom):
    box = depth[left:(right + 1), top:(bottom + 1)]
    return np.mean(box)


if __name__ == '__main__':
    test_cam()