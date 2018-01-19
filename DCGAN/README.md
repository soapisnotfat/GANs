## Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks
![dcgan](http://www.redhub.io/Tensorflow/DCGAN-tensorflow/raw/master/DCGAN.png)
The model proposed in this paper is known as **Deep Convolutional Generative Adversarial Networks (DCGAN)**. 
In this paper, authors upgrade **Generative Adversarial Networks (GAN)** by changing multi-layer perception to 
convolutional neural network so they stabilize the training process of the model.

Core to their appoach to stabilize the training process of the GAN:
- Replace all pooling layers with strided convolutions (discriminator) and fractional-strided convolutions (generator).
- Remove all fully connected hidden layer. Connect input directly to the convolutional layer (see Figure 1).
- Use batch normalization for every layer except for input layer of discriminator and output layer of generator.
- Use leaky ReLU activation in discriminator for all layers and ReLU activation in generator for all layers except for the output which uses tanh.