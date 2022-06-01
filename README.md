# object_detection_in_remote_sensing_images
## Introduction
Object detection in remote sensing images has been widely used in many fields, such as forest and ocean monitoring, military target discrimination and so on. With the development of science and technology, people hope to process massive remote sensing image data and obtain useful information automatically. However, object detection in remote sensing images is always a challenging task due to complex background, variable illumination conditions, large image size, wide range of object scales and angle orientations, dense arrangement and overlapping occlusion between objects. To solve the above problems, we propose an object detection framework based on convolutional neural networks and attention mechanism, with the main work as follows:

Firstly, we analyze the research background and its significance, and then deeply discuss the research status concerning object detection in remote sensing images using deep learning methods. Also, we introduce the basic structure and corresponding mathematical principles of Cascade R-CNN and Swin Transformer, and analyze the advantages of each module.

Secondly, DOTA devkit is used to segment training set and validation set in DOTA-v1.0 and merge detection results, whereas mmdetection is used to build a basic detector framework. In this paper, ResNet-50, ResNet-101, Swin-T, Swin-S and Swin-B backbone networks are utilized to train and test the model in both 1x and 3x training schedules, and the performance of different detectors is compared to verify the advantages of attention mechanism in object detection. The mAP is improved by about 5.1 percentage points from the ResNet-50_1x baseline.

Finally, various data augmentation strategies are used to improve the detection performance of the model. The strategies include image flipping, multi-scale random cropping, shift scale rotation, brightness and contrast transformation, color transformation, multi-scale testing and so on. In this paper, the effectiveness of various data augmentation strategies is illustrated through ablation experiments. Data augmentation strategies have improved mAP by 6.0 percentage points from the Swin-T_3x baseline to 77.7, achieving good detection performance.

## Usage
### Set up Environment
```
wget https://repo.anaconda.com/archive/Anaconda3-2018.12-Linux-x86_64.sh

conda create -n py37 python=3.7

conda activate py37

wget https://bootstrap.pypa.io/get-pip.py

python3 get-pip.py

conda install pytorch=1.7.0 torchvision cudatoolkit=10.2 -c pytorch
```
