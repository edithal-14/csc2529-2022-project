https://docs.google.com/document/d/1eTyDqG1onTeUFqBHmmsmptLi0dp2C8NSIBOLK4T0jzs/edit

A Deep Cascade of Convolutional Neural Networks for Dynamic MR Image Reconstruction

Work:
The application of CNNs in undersampled MR reconstruction and investigate whether they can exploit data redundancy through learned representations. CNN reconstruction can be seen as solving a de-aliasing problem in the image domain.

Motivation:
MRI is associated with an inherently slow acquisition process. This is because data samples of an MR image are acquired sequentially in k space, and the speed at which k-space can be traversed is limited by physiological and hardware constraints. A long data acquisition procedure imposes significant demands on patients, making this imaging modality expensive and less accessible. To accelerate the acquisition process we can undersample k-space, which can provide an acceleration rate proportional to a reduction factor of a number of k-space traversals required.

Challenge:
Undersampling in k-space violates the Nyquist-Shannon theorem and generates aliasing artefacts when the image is reconstructed. The main challenge in this case is to find an algorithm that can recover an uncorrupted image taking into account the undersampling regime.

Reconstruction of undersampled MR images is challenging because the images typically have low signal-to-noise ratio, yet often high-quality reconstructions are needed for clinical applications.

Methods:
Deep network architecture which forms a cascade of CNNs. The cascade network closely resembles the iterative reconstruction of DL-based methods, and still allows end-to-end optimization of the reconstruction algorithm. For 2D reconstruction, each image can be reconstructed in about 23ms, which is fast enough to enable real-time applications.

For DL-based methods, the optimization problem is solved by alternating between the dealiasing step and the data consistency step until convergence. In contrast, with CNNs, we are performing one step dealiasing and the same network cannot be used to de-alias iteratively. In addition, training such networks may require a long time as well as careful fine-tuning steps. We can train a second CNN which learns to reconstruct from the output of the first CNN. We can use a cascading network, which concatenates a new CNN on the output of the previous CNN to build extremely deep networks which iterate between intermediate de-aliasing and the data consistency reconstruction. If each CNN expresses the dictionary learning reconstruction step, then the cascading CNN can be seen as a direct extension of DLMRI, where the whole reconstruction pipeline can be optimised from training. In particular, owing to the forward and back-backpropagation rules defined for the DC layer, all subnetworks can be trained jointly in an end-toend manner, defining yielding one large network.

The used dataset is relatively small (300 images), but by applying data augmentation including rigid transformation and elastic deformation to counter overfitting. From rigid transformation
alone, we create 0.3 million augmented data per image. Combined with the on-the-fly generation of undersampling masks, we generate a very large dataset.

Network Architecture:
The CNN takes in a two-channeled sequence of images R2NxNyNt, where the channels store real and imaginary parts of the zerofilled reconstruction in the image domain. The network has           nd − 1 3D convolution layers, followed by ReLU as a choice of nonlinearity. The final layer of the CNN module is a convolution layer which projects the extracted representation back to the image domain. The network also has a residual connection, which sums the output of the CNN module with its input. Finally, we form a cascading network by using the DC layers interleaved with the CNN reconstruction modules. The Adam optimiser was used to train all models.

Results:
CNN was able to predict some anatomical details which was not possible by DLMRI. This could be due to the fact that the CNNs has more free parameters to tune with, allowing the network to learn complex but more accurate end-to-end transformations of data.

CNN consistently outperforms stateof-the-art methods for all undersampling factors. For a low
acceleration factor (3x undersampling), all methods performed approximately the same, however, for more aggressive undersampling factors, CNN was able to reduce the error by a
considerable margin.

While training the CNN is time consuming, once it is trained, the inference can be done extremely quickly on a GPU. Reconstructing each slice took 23 ± 0.1 milliseconds, which enables real-time applications. To produce the above results, DLMRI took about 6.1 ± 1.3 hours per subject on CPU because DLMRI requires dozens of iterations of dictionary learning and sparse coding steps. Using a fixed, pre-trained dictionary could remove this bottleneck in computation although this would likely be to the detriment of reconstruction quality.
