from __future__ import absolute_import, division, print_function

import time

import cv2
import os
import argparse
import numpy as np
import PIL.Image as pil
from webcam import WebcamVideoStream
from display import DisplayImage
import statistics

import torch
from torchvision import transforms, datasets

import networks
from layers import disp_to_depth
from utils import download_model_if_doesnt_exist


def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing function for Monodepthv2 models.')

    parser.add_argument('--model_name', type=str,
                        help='name of a pretrained model to use',
                        choices=[
                            "mono_640x192",
                            "stereo_640x192",
                            "mono+stereo_640x192",
                            "mono_no_pt_640x192",
                            "stereo_no_pt_640x192",
                            "mono+stereo_no_pt_640x192",
                            "mono_1024x320",
                            "stereo_1024x320",
                            "mono+stereo_1024x320"])
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')
    parser.add_argument('--webcam',
                        help='integer corresponding to desired webcam, default is 0')

    return parser.parse_args()


def test_cam(args):
    """Function to predict for a camera image stream
    """

    model_name = args.model_name if args.model_name is not None else "mono+stereo_640x192"
    no_cuda = args.no_cuda if args.no_cuda is not None else False
    webcam = args.webcam if int(args.webcam) else 0
    print(webcam)

    if torch.cuda.is_available() and not no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    download_model_if_doesnt_exist(model_name)
    model_path = os.path.join("models", model_name)
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # Extract the height and width of image that this model was trained with
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

    print("-> Loading complete, initializing the camera")

    # Initialize camera to capture image stream
    # Change the value to 0 when using default camera
    video_stream = WebcamVideoStream(src=webcam).start()

    # Object to display images
    image_display = DisplayImage()

    # Number of frames to capture to calculate fps
    num_frames = 5
    curr_time = np.zeros(num_frames)
    with torch.no_grad():
        while True:
            # Capture frame-by-frame
            frame = video_stream.read()

            # Calculate the fps
            curr_time[1:] = curr_time[:-1]
            curr_time[0] = time.time()
            fps = num_frames / (curr_time[0] - curr_time[len(curr_time) - 1])

            # Our operations on the frame come here
            input_image = pil.fromarray(frame).convert('RGB')
            original_width, original_height = input_image.size
            input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            # PREDICTION
            input_image = input_image.to(device)
            features = encoder(input_image)
            outputs = depth_decoder(features)

            disp = outputs[("disp", 0)]
            disp_resized = torch.nn.functional.interpolate(disp, (original_height, original_width), mode="nearest")

            # Get the predict depth
            scaled_disp, pred_depth = disp_to_depth(disp_resized, 0.1, 100)
            pred_depth_np = pred_depth.squeeze().cpu().detach().numpy()

            # Initialize a 3x4 depth map
            depth_map = np.zeros([3, 4])
            for i in range(len(depth_map)):
                for j in range(len(depth_map[0])):
                    # Cut and store the average value of depth information of 640x480 into 3x4 grid
                    depth_map[i][j] = get_avg_depth(pred_depth_np, 160 * i, 160 * j, 160 * i + 160, 160 * j + 160)

            # Giving a simple decision logic
            if depth_map[0, 1] <= 1 or depth_map[1, 1] <= 1 or depth_map[0, 2] <= 1 or depth_map[1, 2] <= 1:
                if depth_map[1, 1] <= 1 and depth_map[1, 2] <= 1:
                    print("Dangerous!!! AHEAD")
                else:
                    if depth_map[0, 1] <= 1 or depth_map[1, 1] <= 1:
                        print("Dangerous!!! LEFT")
                    if depth_map[0, 2] <= 1 or depth_map[1, 2] <= 1:
                        print("Dangerous!!! RIGHT")
            elif np.sum(depth_map[0:2, 2:3]) <= 7 or np.sum(depth_map[0:2, 2:3]) <= 7:
                if np.sum(depth_map[0:2, 0:1]) <= 7:
                    print("Careful!! LEFT")
                if np.sum(depth_map[0:2, 2:3]) <= 7:
                    print("Careful!! RIGHT")
            else:
                print("Clear")

            # DISPLAY
            # Generate color-mapped depth image
            disp_resized_np = disp_resized.squeeze().cpu().detach().numpy()
            image_display.display(frame, disp_resized_np, fps, original_width, original_height, blended=True)
            print(str(fps))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print('-> Done!')
                break

    # When everything is done, stop camera stream and close windows
    video_stream.stop()
    image_display.close()


# TODO: Trim the box
def get_avg_depth(depth, left, top, right, bottom):
    """Function to get average depth of a bounding boxed area from a depth map (2D numpy array)
    """

    box = depth[left:(right + 1), top:(bottom + 1)]
    return np.mean(box)


if __name__ == '__main__':
    args = parse_args()
    test_cam(args)
