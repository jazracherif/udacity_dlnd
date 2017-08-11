# Face Generation using DCGAN

In this project, I use a DCGAN architecture as describe in a 2016 paper by [Radford et. al](https://arxiv.org/abs/1511.06434) to learn both MNIST Digits as well as human faces. 

My implementation slightly differs in terms of the size of the Generator layer and the expected output:

**Generator Architecture**:  
inputz: dim=100 --> 2x2x1024 --> 4x4x512 --> 7x7x256 --> 14x14x128 --> 28x28x3  

Notes:
* Instead of performing the transpose convolution, I use the techniques described in this [distill](https://distill.pub/2016/deconv-checkerboard/) article by Odena et. al, which suggests upsampling by first resizing the image using the method of **Nearest Neighbor Interpolation** and then performing the convolution. For example, the layer before last would be resize to a 28x28 images and then performing a convolution with 3 filters:
```
x = tf.image.resize_nearest_neighbor(img, size=(28,28))
x = tf.layers.conv2d(x, 3, (5,5), padding='same', use_bias=False, activation=None)
```
* All levels use convolutions the middle steps use  **Batch Normalization** with a **Leaky Relu**. The logits are taken through a tanh for the purpose of evaluating performance.

**Discriminator Architecture**: input=28x28x28 image --> 2x2x1024 --> 4x4x512 --> 7x7x256 --> 14x14x128 --> 28x28x3  
All the middle steps use **Batch Normalization** with a **Leaky Relu**. The logits are taken through a tanh for the purpose of evaluating performance.

## Results
Within 100 step of training, the DCGAN starts generating ghostly pictures:  
![Alt text](results/faces1.jpg?raw=true "Steps=100")

After 3100 steps of training (at the end of the epoch), we get realistic images of faces:  
![Alt text](results/faces-last.jpg?raw=true "Steps=100")
