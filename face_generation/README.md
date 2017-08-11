# Face Generation using DCGAN

In this project, I use a DCGAN architecture as describe in a 2016 paper by [Radford et. al](https://arxiv.org/abs/1511.06434) to learn both MNIST Digits as well as human faces. 

My implementation slightly differs in terms of the size of the Generator layer and the expected output which must be a 28x28 pictures.


## Results
Within 100 step of training, the DCGAN starts generating ghostly pictures:  
![Alt text](results/faces1.jpg?raw=true "Steps=100")

After 3100 steps of training (at the end of the epoch), we get realistic images of faces:  
![Alt text](results/faces-last.jpg?raw=true "Steps=100")
