# Identify Disease in Tea Leaves

## Description
This repository contains code for a deep learning project that implements various convolutional neural network (CNN) architectures including VGG, ResNet, and EfficientNet. These architectures are widely used for image classification tasks due to their effectiveness in learning complex features from images.


## Models Used
### 1. VGG 16
The VGG-16 model is a convolutional neural network (CNN) architecture that was proposed by the Visual Geometry Group (VGG) at the University of Oxford. It is characterized by its depth, consisting of 16 layers, including 13 convolutional layers and 3 fully connected layers. VGG-16 is renowned for its simplicity and effectiveness, as well as its ability to achieve strong performance on various computer vision tasks, including image classification and object recognition.
![image](https://github.com/Vojaspy/Deep-Learning-Project/assets/116672030/0e7d9374-9821-4e06-a6b9-f54ad61963d8)


### 2. ResNet50
Residual Network: In order to solve the problem of the vanishing/exploding gradient, this architecture introduced the concept called Residual Blocks. In this network, we use a technique called skip connections. The skip connection connects activations of a  layer to further layers by skipping some layers in between. This forms a residual block. Resnets are made by stacking these residual blocks together.
ResNet-50 is a 50-layer convolutional neural network (48 convolutional layers, one MaxPool layer, and one average pool layer). Residual neural networks are a type of artificial neural network (ANN) that forms networks by stacking residual blocks.
![image](https://github.com/Vojaspy/Deep-Learning-Project/assets/116672030/5929507b-856b-41c2-8e2d-a5e56cfcb927)
![image](https://github.com/Vojaspy/Deep-Learning-Project/assets/116672030/6cbcb65b-1597-4415-b10c-ac9c59853f5b)

### 3. EfficientNetB0
EfficientNet uses a technique called compound coefficient to scale up models in a simple but effective manner. Instead of randomly scaling up width, depth or resolution, compound scaling uniformly scales each dimension with a certain fixed set of scaling coefficients.
EfficientNet B0 is the smallest and most efficient model in the EfficientNet family, achieving state-of-the-art results on various image classification benchmarks with just 5.3 million parameters.
EfficientNetB0, the baseline model in the EfficientNet family, typically consists of 7 blocks, each containing multiple layers. The number of layers in each block varies based on the specific implementation and configuration, but the entire model generally has several hundred layers in total.
![image](https://github.com/Vojaspy/Deep-Learning-Project/assets/116672030/d1eb716f-cf69-4baf-985c-a8291ec4805a)


## Installation and Usage
1. git clone this repository https://github.com/Vojaspy/Deep-Learning-Project
2. Download the tea leaves disese dataset from the link https://www.kaggle.com/datasets/shashwatwork/identifying-disease-in-tea-leafs
3. Upload the notebook in Google Colab or Kaggle and run the notebook on GPU




