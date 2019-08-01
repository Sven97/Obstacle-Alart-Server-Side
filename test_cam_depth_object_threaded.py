from __future__ import absolute_import, division, print_function

import time

import cv2
import os
import sys
import argparse
import numpy as np
import PIL.Image as pil
from pynput import keyboard
from threading import Thread, Lock
from queue import LifoQueue
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
from threads import ObstacleDetectionThread, DepthInferenceThread


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


def test_cam(args):
    """Function to predict for a camera image stream
    """

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

    image_queue = LifoQueue()
    depth_result_queue = LifoQueue()
    #object_result_queue = LifoQueue()
    cuda_lock = Lock()

    # Initialize and start threads for object detection and depth inference
    #object_detection_thread = ObstacleDetectionThread(image_queue, object_result_queue)
    depth_inference_thread = DepthInferenceThread(image_queue, depth_result_queue, cuda_lock, args)

    # Initialize camera to capture image stream
    # Change the value to 0 when using default camera
    video_stream = WebcamVideoStream(src=args.webcam).start()

    if not args.no_display:
        print("Trying to initinalize DisplayImage()")
        # Object to display images
        image_display = DisplayImage(not args.no_process)
        print("Finished initializing DisplayImage()")
    # Flag that records when 'q' is pressed to break out of inference loop below
    quit_inference = False
    def on_release(key):
        if key == keyboard.KeyCode.from_char('q'):
            nonlocal quit_inference
            quit_inference = True
            return False

    keyboard.Listener(on_release=on_release).start()
    print("Finished starting keyboard listener")
    #object_detection_thread.start()
    depth_inference_thread.start()
    print("Started depth_inference_thread")

    #finished = True
    disp_resized = None
    danger_level = None
    original_width = 640
    original_height = 480

    # Number of frames to capture to calculate fps
    num_frames = 5
    curr_time = np.zeros(num_frames)
    with torch.no_grad():
        print("Starting inference loop")
        while True:
            if quit_inference:
                if args.no_display:
                    print('-> Done')
                break

            # Capture and send frame to obstacle detection and depth inference thread to be process
            frame = video_stream.read()
            copy_frame = frame

            # Capture and send frame to obstacle detection and depth inference thread to be process
            #if finished:
            print("Sent image to depth thread")
            image_queue.put(copy_frame)
            #    finished = False
            #else:
            #    print("Still doing last frame")

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (model.dims[2],model.dims[1]))
            image = (2.0/255.0) * image - 1.0
            image = image.transpose((2, 0, 1))
            np.copyto(host_inputs[0], image.ravel())

            start_time = time.time()
            print("Right before copying inputs, acquiring lock")
            try:
                cuda_lock.acquire()
                print("Object acquired lock")
                cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
                print("Right before execute")
                context.execute_async(bindings=bindings, stream_handle=stream.handle)
                print("Finished execute")
                cuda.memcpy_dtoh_async(host_outputs[1], cuda_outputs[1], stream)
                print("Finished copying outputs")
                cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
                print("Finished copying outputs 2")
                stream.synchronize()
                print("Synchronized stream")
                cuda_lock.release()
                print("Object released lock")
                print("execute times "+str(time.time()-start_time))
            except:
                print("Object couldn't acquire lock, skipping")
                continue

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
                    cv2.rectangle(frame, (xmin,ymin), (xmax, ymax), (0,0,255),3)
                    cv2.putText(frame, COCO_LABELS[label],(xmin+10,ymin+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

            # Calculate the fps
            curr_time[1:] = curr_time[:-1]
            curr_time[0] = time.time()
            fps = num_frames / (curr_time[0] - curr_time[len(curr_time) - 1])
            print("Requesting depth thread to send data back")
            # Receive results from threads
            #frame = None
            print("Requesting obstacle thread to send data back")
            #detections, frame = object_result_queue.get()
            try:
                disp_resized, danger_level = depth_result_queue.get()
                #finished = True
            except:
                print("Didn't get frame from depth thread -- still working")

            #print(f"Detections: {detections}")
            print(danger_level)
            original_width = 640
            original_height = 480
            if not args.no_display and disp_resized is not None:
                print("About to use image_display")
                # DISPLAY
                # Generate color-mapped depth image
                image_display.display(frame, disp_resized, fps, original_width, original_height, blended=not args.no_blend)
            #if frame is not None:
            #cv2.imshow("Object detection", frame)
            #else:
            #    continue
            #cv2.waitKey(1)
            else:
                print(f"FPS: {fps}")

    # When everything is done, stop camera stream
    video_stream.stop()

    depth_inference_thread.join()
    #object_detection_thread.join()

# TODO: Trim the box
def get_avg_depth(depth, left, top, right, bottom):
    """Function to get average depth of a bounding boxed area from a depth map (2D numpy array)
    """

    box = depth[left:(right + 1), top:(bottom + 1)]
    return np.mean(box)


if __name__ == '__main__':
    args = parse_args()
    test_cam(args)
