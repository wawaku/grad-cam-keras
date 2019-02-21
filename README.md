# Grad-CAM implementation for Keras version 2.2.4/ Tensorflow version 1.12.0
## Grad-CAM implementation in Keras ##

Gradient class activation maps are a visualization technique for deep learning networks.

See the paper: https://arxiv.org/pdf/1610.02391v1.pdf

The paper authors torch implementation: https://github.com/ramprs/grad-cam


This code assumes Tensorflow dimension ordering, and uses the VGG16 network in keras.applications by default (the network weights will be downloaded on first use).


Usage: `python grad-cam.py <path_to_image>`


##### Examples

'boxer' 

![](/img_grad_cam/4.png)

'lab coat'

![](/img_grad_cam/2.jpg)



