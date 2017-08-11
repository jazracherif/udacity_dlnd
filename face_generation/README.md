# Face Generation using DCGAN

In this project, I use a DCGAN architecture as described in a 2016 paper by [Radford et. al](https://arxiv.org/abs/1511.06434) to learn both MNIST Digits as well as human faces. 

My implementation slightly differs from the paper in terms of parameters of the Generator and Discriminator modules:

**Generator Architecture**:  
inputz: dim=100 --> 2x2x1024 --> 4x4x512 --> 7x7x256 --> 14x14x128 --> 28x28x3  --> Tanh

Notes:
* Instead of performing the **Transpose Convolution (Deconvolution)**, I use the techniques described in this [distill](https://distill.pub/2016/deconv-checkerboard/) article by Odena et. al to avoid the problem of image checkerboards, which suggests upsampling by first resizing the image using the method of **Nearest Neighbor Interpolation** and then performing the convolution. For example, the layer before last would be resize to a 28x28 images and then performing a convolution with 3 filters:
```
x = tf.image.resize_nearest_neighbor(img, size=(28,28))
x = tf.layers.conv2d(x, 3, (5,5), padding='same', use_bias=False, activation=None)
```
* All middle levels use  **Batch Normalization** with a **Leaky Relu**. The logits are taken through a **tanh** for the purpose of evaluating performance.

**Discriminator Architecture**: input=28x28x28 image --> 14x14x64 --> 7x7x128 --> 4x4x256 --> Sigmoid  
All the middle steps use **Batch Normalization** with a **Leaky Relu**. The logits are taken through a **Sigmoid** for the purpose of evaluating performance.

## Results
Within 100 steps of training, the DCGAN starts generating ghostly pictures of faces:  
![Alt text](results/faces1.jpg?raw=true "Steps=100")

After 3100 steps of training (towards the end of 1 epoch), we are getting realistic images of faces:  
![Alt text](results/faces-last.jpg?raw=true "Steps=100")
