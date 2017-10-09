# Semantic Segmentation
### Introduction

The goal of this project was to implement a fully covolotional network to segment images from a dashcam into road and non raod pixels. I started by converting the provided pre-trained VGG-16 classifier to a fully convolutional network. At the Udacity Course the [Kitty Dataset](www.cvlibs.net/datasets/kitti/) was used. After that I used the [Tiramisu](https://arxiv.org/abs/1611.09326) Architecture on the [CamVid Dataset](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/)   

### Semantic Segmentation

Semantic Segmentation is the process of assigning each pixel of an image to there corrisponding class. 
There are a couple of different approches to perform semantic segmentation
1. One of the first approces with Deep Learning was patch classification, where each pixel was classfied by using a fixed size patch arround it. This was par example succesfuly used to segment ["Electron Micrsopy Images"](http://people.idsia.ch/~juergen/nips2012.pdf)
2. The logical next step were the [Fully Convolutional Networks(FCN)](https://arxiv.org/abs/1411.4038). For this type of Architekture the Fully Connected Layers where replaced by Convolutional layers. With that adaption in the network architecture, it was possible to handleany input image size.  Derived from the FCN-Architekure two new Architekures are now stat of the Art.

    i. Encoder-Decoder - Architecture
For this Architecture the basic principle is that the encoder gradually reduces the spatial dimensions and aggreagates the context information and the decoder recovers the spatial dimensions and object details. In the context off medical image segmentation the [U-Net](https://arxiv.org/abs/1505.04597) is one of the mosst populat architectures. 

    ii. The second approch uses Dilated convolutions instead of polling layers.
To smooth the final segmentation Conditional Random fields are used and can be used after each Semantic Segmentation Network.  

### Architecture
To solve the problem 
