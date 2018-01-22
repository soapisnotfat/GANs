# BEGAN: Boundary Equibilibrium Generative Adversarial Networks

This is an implementation of the paper on Boundary Equilibrium Generative Adversarial Networks [(Berthelot, Schumm and Metz, 2017)](#references).

## What are Boundary Equilibrium Generative Adversarial Networks?

Unlike standard generative adversarial networks [(Goodfellow et al. 2014)](#references), boundary equilibrium generative adversarial networks (BEGAN) use an auto-encoder as a disciminator. An auto-encoder loss is defined, and an approximation of the Wasserstein distance is then computed between the pixelwise auto-encoder loss distributions of real and generated samples.

<p align='center'>
<img src='../master/BEGAN/readme/eq_autoencoder_loss.png' width=580>  
</p>

With the auto-encoder loss defined (above), the Wasserstein distance approximation simplifies to a loss function wherein the discriminating auto-encoder aims to perform *well on real samples* and *poorly on generated samples*, while the generator aims to produce adversarial samples which the discriminator can't help but perform well upon.

<p align='center'>
<img src='../master/BEGAN/readme/eq_losses.png' width=380>
</p>

Additionally, a hyper-parameter gamma is introduced which gives the user the power to control sample diversity by balancing the discriminator and generator.

<p align='center'>
<img src='../master/BEGAN/readme/eq_gamma.png' width=170>  
</p>

Gamma is put into effect through the use of a weighting parameter *k* which gets updated while training to adapt the loss function so that our output matches the desired diversity. The overall objective for the network is then:

<p align='center'>
<img src='../master/BEGAN/readme/eq_objective.png' width=510> 
</p>

Unlike most generative adversarial network architectures, where we need to update *G* and *D* independently, the Boundary Equilibrium GAN has the nice property that we can define a global loss and train the network as a whole (though we still have to make sure to update parameters with respect to the relative loss functions)

<p align='center'>
<img src='../master/BEGAN/readme/eq_global.png'>
</p>

The final contribution of the paper is a derived convergence measure M which gives a good indicator as to how the network is doing. We use this parameter to track performance, as well as control learning rate.

<p align='center'>
<img src='../master/BEGAN/readme/eq_conv_measure.png'>
</p>

The overall result is a surprisingly effective model which produces samples well beyond the previous state of the art.

<p align='center'>
<img src='../master/readme/generated_from_Z.png' width=550>
</p>

*128x128 samples generated from random points in Z, from [(Berthelot, Schumm and Metz, 2017)](#references).*


## References

* [Berthelot, Schumm and Metz. BEGAN: Boundary Equilibrium Generative Adversarial Networks. arXiv preprint, 2017](https://arxiv.org/abs/1703.10717)

* [Goodfellow, Ian, et al. "Generative adversarial nets." Advances in neural information processing systems. 2014.](http://papers.nips.cc/paper/5423-generative-adversarial-nets)

* [Liu, Ziwei, et al. "Deep Learning Face Attributes in the Wild." Proceedings of International Conference on Computer Vision. 2015.](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
