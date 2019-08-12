from __future__ import absolute_import, division, print_function

import time

import cv2
import os
import socket
import argparse
import numpy as np
import PIL.Image as pil
from pynput import keyboard
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

    # Get the predicted depth
    scaled_disp, pred_depth = disp_to_depth(disp_resized, 0.1, 100)
    pred_depth_np = pred_depth.squeeze().cpu().detach().numpy()

    # Initialize a 3x4 depth map
    depth_map = np.zeros([3, 4])
    grid_width = original_width // 4
    grid_height = original_height // 3
    for i in range(len(depth_map)):
        for j in range(len(depth_map[0])):
            # Cut and store the average value of depth information of 640x480 into 3x4 grid
            depth_map[i][j] = get_avg_depth(pred_depth_np,
                                            grid_width * i,
                                            grid_height * j,
                                            grid_width * (i + 1),
                                            grid_height * (j + 1))

    # Giving a simple decision logic
    if depth_map[0, 1] <= 1 or depth_map[1, 1] <= 1 or depth_map[0, 2] <= 1 or depth_map[1, 2] <= 1:
        if depth_map[1, 1] <= 1 and depth_map[1, 2] <= 1:
            print("Dangerous!!! AHEAD")
            danger_level = 2
        else:
            if depth_map[0, 1] <= 1 or depth_map[1, 1] <= 1:
                print("Dangerous!!! LEFT")
                danger_level = 2
            if depth_map[0, 2] <= 1 or depth_map[1, 2] <= 1:
                print("Dangerous!!! RIGHT")
                danger_level = 2
    elif np.sum(depth_map[0:2, 2:3]) <= 7 or np.sum(depth_map[0:2, 2:3]) <= 7:
        if np.sum(depth_map[0:2, 0:1]) <= 7:
            print("Careful!! LEFT")
            danger_level = 1
        if np.sum(depth_map[0:2, 2:3]) <= 7:
            print("Careful!! RIGHT")
            danger_level = 1
    else:
        print("Clear")
        danger_level = 0

    return disp_resized, danger_level, original_width, original_height


def detect_objects(frame, host_inputs, host_outputs, cuda_inputs, cuda_outputs, bindings, stream, context, label_dict):
    # Transform image
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (model.dims[2],model.dims[1]))
    image = (2.0/255.0) * image - 1.0
    image = image.transpose((2, 0, 1))
    np.copyto(host_inputs[0], image.ravel())

    start_time = time.time()

    # Transfer input data to gpu memory
    cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)

    # Execute inference
    context.execute_async(bindings=bindings, stream_handle=stream.handle)

    # Transfer outputs from model back to cpu
    cuda.memcpy_dtoh_async(host_outputs[1], cuda_outputs[1], stream)
    cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
    stream.synchronize()
    print("execute times "+str(time.time()-start_time))

    # Collect detections into list if confidence is at least 0.7 and add bounding boxes to frame
    detections = []
    output = host_outputs[0]
    height, width, channels = frame.shape
    for i in range(int(len(output)/model.layout)):
        prefix = i*model.layout
        index = int(output[prefix+0])
        label = int(output[prefix+1])
        conf = output[prefix+2]
        xmin = int(output[prefix+3]*width)
        ymin = int(output[prefix+4]*height)
        xmax = int(output[prefix+5]*width)
        ymax = int(output[prefix+6]*height)

        if conf > 0.7:
            print("Detected {} with confidence {}".format(label_dict[label], "{0:.0%}".format(conf)))
            detections.append([label_dict[label], conf, (xmin+xmax)/2, (ymin+ymax)/2])
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3)
            cv2.putText(frame, label_dict[label], (xmin+10, ymin+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                        cv2.LINE_AA)

    return detections


def test_cam(args):
    """Function to predict for an image stream
    """

    # Determine where to run inference
    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Download model given in args if it doesn't exist
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

    # Get coco labels
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
    host_inputs = []
    cuda_inputs = []
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

    if not args.no_display:
        # Object to display images
        image_display = DisplayImage(not args.no_process)

    # Flag that records when 'q' is pressed to break out of inference loop below
    quit_inference = False

    # Listener for key board presses and updates quit_inference
    def on_release(key):
        if key == keyboard.KeyCode.from_char('q'):
            nonlocal quit_inference
            quit_inference = True
            return False

    # Initialize listener
    keyboard.Listener(on_release=on_release).start()

    # Number of frames to capture to calculate fps
    num_frames = 5
    curr_time = np.zeros(num_frames)

    # Initialize and bind socket
    with socket.socket() as s:
        print("Socket created")
        s.bind(("", 9002))
        s.listen(5)
        print("Socket binded to port 9002 and listening")

        # Wait for client socket to connect
        conn, addr = s.accept()
        print(f"Connected by: {addr}")

        with torch.no_grad():
            # TODO: Probably need to do some error catching in here
            while True:
                if quit_inference:
                    if args.no_display:
                        print('-> Done')
                    break

                # Receive size from client and convert to integer
                image_size = int.from_bytes(conn.recv(4), byteorder="big")
                print(f"\nReceived size: {image_size}")

                # Send back size that was received to confirm that it is correct
                conn.send(image_size.to_bytes(4, byteorder="big"))

                # Keep reading from connection until enough bytes have been read to reach the image size
                total_data = []
                while len(total_data) < image_size:
                    data = conn.recv(1024)
                    total_data.extend(data)
                print("Received image bytes")

                # Send confirmation that image was received back to client
                ok = "OK\n"
                conn.send(ok.encode('utf-8'))

                # Convert bytes to bytearray, then to numpy array, then to cv2 matrix for image
                total_data = bytearray(total_data)
                total_data = np.asarray(total_data)
                frame = cv2.imdecode(total_data, cv2.IMREAD_COLOR)
                print("Decoded frame from bytes, printed below:")
                print(frame)

                # Calculate the fps
                curr_time[1:] = curr_time[:-1]
                curr_time[0] = time.time()
                fps = num_frames / (curr_time[0] - curr_time[len(curr_time) - 1])

                # Do depth inference
                disp_resized, danger_level, original_width, original_height = predict_depth(frame, feed_width,
                                                                                            feed_height, device,
                                                                                            encoder, depth_decoder)

                # Only do object detection if danger level is above 0 (i.e. Careful or Dangerous)
                print(f"Danger level: {danger_level}")
                detections_str = ""
                if danger_level > 0:
                    detections = detect_objects(frame, host_inputs, host_outputs, cuda_inputs, cuda_outputs, bindings,
                                                stream, context, COCO_LABELS)
                    detections_str = '\n' + '\n'.join('$'.join(map(str, obj)) for obj in detections)
                    print(f"Detections: {detections_str}")

                # Construct string with danger level and END signal
                # Separate each piece (i.e. danger level, each detection, END) with new line so client socket knows
                # where each item ends
                result = str(danger_level) + detections_str + "\nEND\n"

                # Send results to client socket
                print("Sending results")
                conn.send(result.encode())

                # Get confirmation from client socket before moving on to receiving next image
                print("Waiting for confirmation...")
                confirmation = conn.recv(32).decode('utf-8')

                # If confirmation is not "OK", break since connection is corrupted somehow
                if confirmation != "OK":
                    print("Confirmation incorrect, stopping")
                    break  # TODO: Look into potentially making server try to regain connection with client socket

                print(f"Confirmation: {confirmation}")

                if not args.no_display:
                    # Generate color-mapped depth image and display alongside original frame and blended, if chosen
                    disp_resized_np = disp_resized.squeeze().cpu().detach().numpy()
                    image_display.display(frame, disp_resized_np, fps, original_width, original_height,
                                          blended=not args.no_blend)
                    cv2.waitKey(1)
                else:
                    print(f"FPS: {fps}")


# TODO: Trim the box
def get_avg_depth(depth, left, top, right, bottom):
    """Function to get average depth of a bounding boxed area from a depth map (2D numpy array)
    """

    box = depth[left:(right + 1), top:(bottom + 1)]
    return np.mean(box)


if __name__ == '__main__':
    args = parse_args()
    test_cam(args)
