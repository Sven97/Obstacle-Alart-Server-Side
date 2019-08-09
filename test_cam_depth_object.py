from __future__ import absolute_import, division, print_function

import time

import cv2
import os
import sys
import argparse
import numpy as np
import PIL.Image as pil
from pynput import keyboard
from threading import Thread
import ctypes
import pycuda.autoinit
import pycuda.driver as cuda
import coco
import uff
import tensorrt as trt
import graphsurgeon as gs
from config import model_ssd_mobilenet_v2_coco_2018_03_29 as model

import torch
from torchvision import transforms, datasets

import networks
from layers import disp_to_depth
from utils import download_model_if_doesnt_exist

from webcam import WebcamVideoStream
from display import DisplayImage


def parse_args():
    parser = argparse.ArgumentParser(
        description='Uses monodepthv2 on webcam')

    parser.add_argument('--model_name', type=str, default="mono+stereo_640x192",
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
    parser.add_argument('--webcam', type=int, default=0,
                        help='integer corresponding to desired webcam, default is 0')
    parser.add_argument('--no_process',
                        help='if set, displays image in current process, might improve performance on machines without a GPU',
                        action='store_true')
    parser.add_argument('--no_blend',
                        help='if set, does not display blended image',
                        action='store_true')
    parser.add_argument('--no_display',
                        help='if set, does not display images, only prints fps',
                        action='store_true')

    return parser.parse_args()


def predict_depth(image, input_width, input_height, device, encoder, decoder):
    # Our operations on the frame come here
    input_image = pil.fromarray(image).convert('RGB')
    original_width, original_height = input_image.size
    input_image = input_image.resize((input_width, input_height), pil.LANCZOS)
    input_image = transforms.ToTensor()(input_image).unsqueeze(0)

    # PREDICTION
    input_image = input_image.to(device)
    features = encoder(input_image)
    outputs = decoder(features)

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
            danger_level = "Dangerous!!! AHEAD"
            danger_code = 3
        else:
            if depth_map[0, 1] <= 1 or depth_map[1, 1] <= 1:
                danger_level = "Dangerous!!! LEFT"
                danger_code = 2
            if depth_map[0, 2] <= 1 or depth_map[1, 2] <= 1:
                danger_level = "Dangerous!!! RIGHT"
                danger_code = 2
    elif np.sum(depth_map[0:2, 2:3]) <= 7 or np.sum(depth_map[0:2, 2:3]) <= 7:
        if np.sum(depth_map[0:2, 0:1]) <= 7:
            danger_level = "Careful!! LEFT"
            danger_code = 1
        if np.sum(depth_map[0:2, 2:3]) <= 7:
            danger_level = "Careful!! RIGHT"
            danger_code = 1
    else:
        danger_level = "Clear"
        danger_code = 0

    return disp_resized, danger_level, danger_code,  original_width, original_height


def detect_objects(frame, host_inputs, host_outputs, cuda_inputs, cuda_outputs, bindings, stream, context, COCO_LABELS):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (model.dims[2],model.dims[1]))
    image = (2.0/255.0) * image - 1.0
    image = image.transpose((2, 0, 1))
    np.copyto(host_inputs[0], image.ravel())

    start_time = time.time()
    cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
    context.execute_async(bindings=bindings, stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(host_outputs[1], cuda_outputs[1], stream)
    cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
    stream.synchronize()
    print("execute times "+str(time.time()-start_time))
    detections = []
    output = host_outputs[0]
    height, width, channels = frame.shape
    for i in range(int(len(output)/model.layout)):
        prefix = i*model.layout
        index = int(output[prefix+0])
        label = int(output[prefix+1])
        conf  = output[prefix+2]
        xmin  = int(output[prefix+3]*width)
        ymin  = int(output[prefix+4]*height)
        xmax  = int(output[prefix+5]*width)
        ymax  = int(output[prefix+6]*height)

        if conf > 0.7:
            print("Detected {} with confidence {}".format(COCO_LABELS[label], "{0:.0%}".format(conf)))
            detections.append([label, conf, (xmin+xmax)/2, (ymin+ymax)/2])
            cv2.rectangle(frame, (xmin,ymin), (xmax, ymax), (0,0,255),3)
            cv2.putText(frame, COCO_LABELS[label],(xmin+10,ymin+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

    return detections


def test_cam(args):
    """Function to predict for a camera image stream
    """

    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    download_model_if_doesnt_exist(args.model_name)
    model_path = os.path.join("models", args.model_name)
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

    ctypes.CDLL("../TRT_object_detection/lib/libflattenconcat.so")
    COCO_LABELS = coco.COCO_CLASSES_LIST


    # initialize
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(TRT_LOGGER, '')
    runtime = trt.Runtime(TRT_LOGGER)


    # compile model into TensorRT
    if not os.path.isfile(model.TRTbin):
        dynamic_graph = model.add_plugin(gs.DynamicGraph(model.path))
        uff_model = uff.from_tensorflow(dynamic_graph.as_graph_def(), model.output_name, output_filename='tmp.uff')

        with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.UffParser() as parser:
            builder.max_workspace_size = 1 << 28
            builder.max_batch_size = 1
            builder.fp16_mode = True

            parser.register_input('Input', model.dims)
            parser.register_output('MarkOutput_0')
            parser.parse('tmp.uff', network)
            engine = builder.build_cuda_engine(network)

            buf = engine.serialize()
            with open(model.TRTbin, 'wb') as f:
                f.write(buf)


    # create engine
    with open(model.TRTbin, 'rb') as f:
        buf = f.read()
        engine = runtime.deserialize_cuda_engine(buf)


    # create buffer
    host_inputs  = []
    cuda_inputs  = []
    host_outputs = []
    cuda_outputs = []
    bindings = []
    stream = cuda.Stream()

    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        host_mem = cuda.pagelocked_empty(size, np.float32)
        cuda_mem = cuda.mem_alloc(host_mem.nbytes)

        bindings.append(int(cuda_mem))
        if engine.binding_is_input(binding):
            host_inputs.append(host_mem)
            cuda_inputs.append(cuda_mem)
        else:
            host_outputs.append(host_mem)
            cuda_outputs.append(cuda_mem)
    context = engine.create_execution_context()

    # Initialize camera to capture image stream
    # Change the value to 0 when using default camera
    video_stream = WebcamVideoStream(src=args.webcam).start()

    if not args.no_display:
        # Object to display images
        image_display = DisplayImage(not args.no_process)

    # Flag that records when 'q' is pressed to break out of inference loop below
    quit_inference = False
    def on_release(key):
        if key == keyboard.KeyCode.from_char('q'):
            nonlocal quit_inference
            quit_inference = True
            return False

    keyboard.Listener(on_release=on_release).start()

    # Number of frames to capture to calculate fps
    num_frames = 5
    curr_time = np.zeros(num_frames)
    with torch.no_grad():
        while True:
            if quit_inference:
                if args.no_display:
                    print('-> Done')
                break

            # Capture frame-by-frame
            frame = video_stream.read()

            # Calculate the fps
            curr_time[1:] = curr_time[:-1]
            curr_time[0] = time.time()
            fps = num_frames / (curr_time[0] - curr_time[len(curr_time) - 1])

            disp_resized, danger_level, danger_code, original_width, original_height = predict_depth(frame, feed_width, feed_height, device, encoder, depth_decoder)
            print(danger_level)
            if danger_code > 0:
                detections = detect_objects(frame, host_inputs, host_outputs, cuda_inputs, cuda_outputs, bindings, stream, context, COCO_LABELS)
                print("Detections")
                print(detections)
            print(f"original_width: {original_width}, original_height: {original_height}")
            if not args.no_display:
                # DISPLAY
                # Generate color-mapped depth image
                disp_resized_np = disp_resized.squeeze().cpu().detach().numpy()
                image_display.display(frame, disp_resized_np, fps, original_width, original_height, blended=not args.no_blend)
                #cv2.imshow("Object detection", object_frame)
                cv2.waitKey(1)
            else:
                print(f"FPS: {fps}")

    # When everything is done, stop camera stream
    video_stream.stop()


# TODO: Trim the box
def get_avg_depth(depth, left, top, right, bottom):
    """Function to get average depth of a bounding boxed area from a depth map (2D numpy array)
    """

    box = depth[left:(right + 1), top:(bottom + 1)]
    return np.mean(box)


if __name__ == '__main__':
    args = parse_args()
    test_cam(args)
