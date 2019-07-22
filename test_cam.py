from __future__ import absolute_import, division, print_function

import cv2
import os
import time
import os
import sys
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
    cap = cv2.VideoCapture(1)

    count = 0

    # PREDICTING ON CAMERA IMAGE STREAM
    with torch.no_grad():
        while cap.isOpened():
            # Capture frame-by-frame
            ret, frame = cap.read()

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
            disp_resized = torch.nn.functional.interpolate(disp, (original_height, original_width), mode="bilinear",
                                                           align_corners=False)

            # Saving numpy file
            scaled_disp, _ = disp_to_depth(disp, 0.1, 100)

            # Display colormapped depth image
            disp_resized_np = disp_resized.squeeze().cpu().detach().numpy()
            vmax = np.percentile(disp_resized_np, 100)
            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
            im = pil.fromarray(colormapped_im)

            # Display the resulting frame
            cv2.imshow('Result', cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR))
            cv2.imshow('Original', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print('-> Done!')
                break

    # When everything done, release the capture
    cap.release()


if __name__ == '__main__':
    test_cam()
