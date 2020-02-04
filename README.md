DEEP Open Catalogue: Pose Estimation
====================================

[![Build Status](https://jenkins.indigo-datacloud.eu/buildStatus/icon?job=Pipeline-as-code/DEEP-OC-org/posenet-tf/master)](https://jenkins.indigo-datacloud.eu/job/Pipeline-as-code/job/DEEP-OC-org/job/posenet-tf/job/master/)


**Author:** [Lara Lloret Iglesias](https://github.com/laramaktub) (CSIC)

**Project:** This work is part of the [DEEP Hybrid-DataCloud](https://deep-hybrid-datacloud.eu/) project that has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No 777435.

This is a plug-and-play tool for real-time pose estimation using deep neural networks. PoseNet can be used to estimate
either a single pose or multiple poses, meaning there is a version of the algorithm that can detect only one person in
an image/video and one version that can detect multiple persons in an image/video. The module implemented here works on
pictures (either uploaded or using an URL) and gives as output the different body keypoints with the corresponding
coordinates and the associated key score. It also generates an image with the keypoints superimposed.

<p align="center">
<img src="./reports/figures/posenet.png" width="400">
</p>

## Installing this module

### Local installation

> **Requirements**
>
> This project has been tested in Ubuntu 18.04 with Python 3.6.5. Further package requirements are described in the
> `requirements.txt` file.
> - It is a requirement to have [Tensorflow>=1.14.0 installed](https://www.tensorflow.org/install/pip) (either in gpu 
> or cpu mode). This is not listed in the `requirements.txt` as it [breaks GPU support](https://github.com/tensorflow/tensorflow/issues/7166). 
> - Run `python -c 'import cv2'` to check that you installed correctly the `opencv-python` package (sometimes
> [dependencies are missed](https://stackoverflow.com/questions/47113029/importerror-libsm-so-6-cannot-open-shared-object-file-no-such-file-or-directo) in `pip` installations).

To start using this framework clone the repo:

```bash
git clone https://github.com/deephdc/posenet-tf
cd posenet-tf
pip install -e .
```
now run DEEPaaS:
```
deepaas-run --listen-ip 0.0.0.0
```
and open http://0.0.0.0:5000/ui and look for the methods belonging to the `posenetclas` module.

### Docker installation

We have also prepared a ready-to-use [Docker container](https://github.com/deephdc/DEEP-OC-posenet-tf) to
run this module. To run it:

```bash
docker search deephdc
docker run -ti -p 5000:5000 -p 6006:6006 -p 8888:8888 deephdc/deep-oc-posenet-tf
```

Now open http://0.0.0.0:5000/ui and look for the methods belonging to the `posenetclas` module.


## Test

Go to http://0.0.0.0:5000/ui and look for the `PREDICT` POST method. Click on 'Try it out', change whatever test args
you want and click 'Execute'. You can **either** supply a:

* a `data` argument with a path pointing to an image.

OR
* a `url` argument with an URL pointing to an image. 
 Here is an [example](https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQOCB8ImJKc6uD12ZvXhM_2EFkqCi1xcd-izsCMWrDOy-ZMq80X) of such an url
 that you can use for testing purposes.

## Acknowledgements

The original model, weights, code, etc. were created by Google and can be found [here](https://github.com/tensorflow/tfjs-models/tree/master/posenet).
