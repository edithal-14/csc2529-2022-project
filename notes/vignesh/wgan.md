## Wasserstein GAN tutorial

### References
- https://machinelearningmastery.com/how-to-code-a-wasserstein-generative-adversarial-network-wgan-from-scratch/

- https://jonathan-hui.medium.com/gan-wasserstein-gan-wgan-gp-6a1a2aa1b490

- https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan/wgan.py

- https://github.com/Zeleni9/pytorch-wgan/blob/master/models/wgan_clipping.py

- https://bytepawn.com/training-a-pytorch-wasserstain-mnist-gan-on-google-colab.html

### Extension to GAN
- Improves training stability
- Loss function better correlates with quality of images

### Tutorial Overview
- Theory
- Implementation detail
- How to train

### Author of the WGAN paper
- Martin Arjovsky et al. 2017

### Replace discriminator with a scoring critic
- Stems from theoretical argument
  - Seek minimization of distance between real and generated distributions

### Benefits of WGAN
- Training is more stable, less sensitive to model architecture/hyperparameter changes
  - Convergence is more stable with respect to changes in model architecture/hyperparameters
- Most importantly, loss of discriminator appears to relate to the quality of images generated
  - Loss reflects image quality
- Prevents mode collapse

### Contributions
- Usage of earth mover distance (infimum (greatest lower bound) of transport plan)
  - Instead of KL divergence, Jensen Shannon Divergence
  - Gradient of divergence is small for large divergence, so generator barely learns
  - Large variance of gradients makes model unstable

### Training loop and hyperparameters from the paper
- hyperparameters
- learning rate (alpha)  = 5e-5
- clipping parameter (c) = 1e-2
- batch size (m)         = 64
- number of critic iterations per generator iteration (n_critic) = 5
- training
- For every critic iteration
  - Sample a batch of real images
  - Sample a batch of latent vectors
  - Generate batch of fake images
  - Minimize Wasserstein loss between real and fake images
  - RMSProp to update parameters of critic function
  - Clip critic parameters from -c to c
- For every generator iteration
  - Minimize wasserstein loss of generated fake images
  - RMSProp to update parameters of generator function

### Implementation details
- Use linear activation function in the output layer of discriminator
- Labelling convention
  - Real images: -1
  - Fake images: 1
- Wasserstein loss
  - Maximizing for real images
  - Minimizing for fake images
  - Multiply mean score by class label (-1 for real and 1 for fake)
    - mean(y_true * y_pred)
- WGAN requires gradient clipping for critic model
    - Clip between -0.01 to 0.01
    - Can be implemented using Keras constraint
    - In Pytorch use a custom layer with custom call to Conv2d where weights are clamped
- Add hyperparameter n_critic to update critic more number of times than generator
- Use RMSProp with learning rate of 1e-5

## Example problem
- Generate digit "7" from MNIST dataset
- 28 x 28 pixels image resolution

## Literature survey
- WGAN gradient penalty (GP) is better than weight clipping

## WGAN GP

- Uses gradient penalty instead of weight clipping to enforce the Lipschitz constraint
- A differentiable function _f_ is 1 Lipschitz if and only if it has gradients with norm at most 1 everywhere
- Points interpolated between real and fake distribution should have a gradient norm of 1
- To the loss function apply penalty if gradient moves away from 1
- Batch normalization creates correlations between samples in the same batch, so, it can affect effectiveness of GP term
  - Avoid BN for critic model
- GP increases complexity and affects performance but produces higher quality images  
- Better convergence, quality and training stability