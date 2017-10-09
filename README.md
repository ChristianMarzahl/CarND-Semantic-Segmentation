# Semantic Segmentation
### Introduction

The goal of this project was to implement a fully covolotional network to segment images from a dashcam into road and non raod pixels. I started by converting the provided pre-trained VGG-16 classifier to a fully convolutional network. At the Udacity Course the [Kitty Dataset](www.cvlibs.net/datasets/kitti/) was used. After that I used the [Tiramisu](https://arxiv.org/abs/1611.09326) Architecture on the [CamVid Dataset](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/) and was able to reproduce the accuray defined in the paper.  

### Semantic Segmentation

Semantic Segmentation is the process of assigning each pixel of an image to there corrisponding class. 
There are a couple of different approches to perform semantic segmentation
1. One of the first approces with Deep Learning was patch classification, where each pixel was classfied by using a fixed size patch arround it. This was par example succesfuly used to segment ["Electron Micrsopy Images"](http://people.idsia.ch/~juergen/nips2012.pdf)
2. The logical next step were the [Fully Convolutional Networks(FCN)](https://arxiv.org/abs/1411.4038). For this type of Architekture the Fully Connected Layers where replaced by Convolutional layers. With that adaption in the network architecture, it was possible to handle any input image size.  Derived from the FCN-Architekure two new Architekures are now stat of the Art.

    i. Encoder-Decoder - Architecture
For this Architecture the basic principle is that the encoder gradually reduces the spatial dimensions and aggreagates the context information and the decoder recovers the spatial dimensions and object details. In the context off medical image segmentation the [U-Net](https://arxiv.org/abs/1505.04597) is one of the mosst populat architectures. 

    ii. The second approch uses Dilated convolutions instead of polling layers.
To smooth the final segmentation Conditional Random fields are used and can be used after each Semantic Segmentation Network.  

### Architecture
To performe the conversion from the provided pre-trained VGG-16 network to a fully convolutional network the final fully connected layer where replaced by 1x1 convolutions with the number of filters set to the number of target classes (Road, No Road). The MaxPolling layer from the pre-trained network decrease the spatial accuracy, to overcome this, two skip connections performing 1x1 convolutions are implemented at layer three and four. This two skip connections are added and than via transposed convolution upsampled, the converted layer seven is first upsampled and than added.     

- kernel_initializer: random_normal_initializer(stddev=0.01)
-kernel_regularizer: l2_regularizer(1e-3)

### Training:
- image_shape: 160, 576 (Even if fully connected networks can handle every input size, the accuracy is not the same)
- epochs: 50
- batch_size: 20
- keep_prob: 0.5 (Dropout)
- Optimizier: Adam
- learning_rate: 0.001

