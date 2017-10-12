# Semantic Segmentation
### Introduction

The goal of this project was to implement a fully covolotional network to segment images from a dashcam into road and non raod pixels. I started by converting the provided pre-trained VGG-16 classifier to a fully convolutional network. At the Udacity Course the [Kitty Dataset](www.cvlibs.net/datasets/kitti/) was used. After that I used the [Tiramisu](https://arxiv.org/abs/1611.09326) Architecture on the [CamVid Dataset](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/) and was able to reproduce the accuray defined in the paper.  

| ![CamVid](./CamVid-Images/0016E5_00390_2017_10_10_09_07_22_drawings.png "CamVid Image") | 
|:--:| 
| *Results from the Tiramisu Network trained on CamVid with 32 different classes.* |

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
- kernel_regularizer: l2_regularizer(1e-3)

| ![VGG-Architecture](./CamVid-Images/Graph.png "VGG16-FCN") | 
|:--:| 
| *The VGG Network converted to VGG16-FCN* |

### Training:
- image_shape: 160, 576 (Even if fully connected networks can handle every input size, the accuracy is not the same)
- epochs: 50
- batch_size: 20
- keep_prob: 0.5 (Dropout)
- Optimizier: Adam
- learning_rate: 0.001
- Augmentation: flip
```python
# performe a vertical flip for every second image 
if flip_lr and np.random.choice(2, 1)[0] == 1:
    image = np.fliplr(image)
    gt_image = np.fliplr(gt_image)
```

| ![Final_Convolution](./CamVid-Images/Histograms.png "Final_Convolution") | 
|:--:| 
| *The histograms for the final conv2d_transpose layer* |


### Results:

| ![VGG-FCN Loss](./CamVid-Images/VGG-Kitty.png "VGG-FCN Loss") | 
|:--:| 
| *The batch loss development, for 50 or 100 epochs with and without data augmentation* |


| ![Kitty_Result](./CamVid-Images/1507737726_full.gif "Kitty_Result") | 
|:--:| 
| *The Kitty Validation images* |

