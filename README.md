# Monodepth2

Based on the original version of [Monodepth2](https://github.com/nianticlabs/monodepth2), We added the ***test_cam.py*** to test Monodepth2 on image stream using a webcam.

<p align="center">
  <img src="assets/test.gif" alt="example input output gif" width="600" />
</p>

## ⚙️ Setup

You can refer to the [README]() from the original project to setup the required environment or just quickly go through the following steps:

Assuming a fresh [Anaconda](https://www.anaconda.com/download/) distribution, you can install the dependencies with:

```shell
conda install pytorch=0.4.1 torchvision=0.2.1 -c pytorch
pip install tensorboardX==1.4
conda install opencv=3.3.1   # just needed for evaluation and display the result
```

## 🔧 Usage

You can predict depth for a single image with:
```shell
python test_simple.py --image_path assets/test_image.jpg --model_name mono+stereo_640x192
```

Or you can predict depth using image stream with a webcam:

```shell
python test_cam.py
```

```shell
# The arguments can be added to run the above command is listed below

--model_name
# default="mono+stereo_640x192"
# choices=["mono_640x192",
# "stereo_640x192",
# "mono+stereo_640x192",
# "mono_no_pt_640x192",
# "stereo_no_pt_640x192",
# "mono+stereo_no_pt_640x192",
# "mono_1024x320",
# "stereo_1024x320",
# "mono+stereo_1024x320"]

--no_cuda
# if set, disables CUDA

--webcam
# default=0
# change to the desired by changing the value

--no_process
# if set, displays image in current process, might improve performance on machines without a GPU

--no_blend
# if set, does not display blended image

--no_display
# if set, does not display images, only prints fps
```

