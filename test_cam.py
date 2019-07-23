from __future__ import absolute_import, division, print_function

import time
import cv2
import os
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

import torch
from torchvision import transforms, datasets

import networks
from layers import disp_to_depth
from utils import download_model_if_doesnt_exist


def test_cam():
    """Function to predict for a camera image stream
    """

    # Can be changed to cpu if no cuda
    device = torch.device("cuda")

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

    # Initialize webcam to capture image stream
    # Change the value to 0 when using default camera
    cap = cv2.VideoCapture(0)

    # PREDICTING ON CAMERA IMAGE STREAM
    with torch.no_grad():
        while cap.isOpened():
            time_begin = time.time()
            # Capture frame-by-frame
            ret, frame = cap.read()
            print("Capture time")
            print(time.time() - time_begin)

            time_begin = time.time()
            # Our operations on the frame come here
            input_image = pil.fromarray(frame).convert('RGB')
            original_width, original_height = input_image.size
            input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)
            print("Preprocessing")
            print(time.time() - time_begin)

            time_begin = time.time()
            # PREDICTION
            input_image = input_image.to(device)
            features = encoder(input_image)
            outputs = depth_decoder(features)

            disp = outputs[("disp", 0)]
            disp_resized = torch.nn.functional.interpolate(disp, (original_height, original_width), mode="nearest") #, align_corners=False)
            print("Prediction")
            print(time.time() - time_begin)

            time_begin = time.time()
            # Saving numpy file
            scaled_disp, _ = disp_to_depth(disp_resized, 0.1, 100)
            # Compute depth
            pred_depth = 5.4 / scaled_disp
            pred_depth_np = pred_depth.squeeze().cpu().detach().numpy()
            print("Compute depth")
            print(time.time() - time_begin)

            time_begin = time.time()
            # Display colormapped depth image
            disp_resized_np = disp_resized.squeeze().cpu().detach().numpy()
            vmax = np.percentile(disp_resized_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
            im = pil.fromarray(colormapped_im)
            result_img = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)

            # Display the resulting frame
            cv2.imshow('Result', result_img)
            cv2.imshow('Original', frame)

            # Blend images
            alpha = 0.2
            beta = 1.0 - alpha
            blended_result = cv2.addWeighted(frame, alpha, result_img, beta, 0.0)
            cv2.imshow('Blended Result', blended_result)
            print("Display")
            print(time.time() - time_begin)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print('-> Done!')
                break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    test_cam()
