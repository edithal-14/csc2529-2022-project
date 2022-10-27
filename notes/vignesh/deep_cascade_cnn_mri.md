### Deep Cascade of CNN for Dynamic MR Image Reconstruction

- Jo Schlemper et. al. (2017)

- Abstract
    - Reconstruct dynamic sequence of 2D cardiac MR images
    - Accelereate data aquisition process
    - Outperforms SOTA 2D compressed sensing approaches
    - Joint reconstruction can learn spatio temporal correlations
      efficiently
    - Reconstruction is very fast (23 ms per frame) enabling real time
      applications

- MRI image aquisition is a slow process
    - Data samples are acquired sequentially in k-space which can be
      traversed at limited speed

- Solution is to undersample k-space which provides acceleration
    - However this violates Nyquist Shannon theorem and leads to aliasing
        artefacts upon image construction
    - Use knowledge of properties of image and undersampling technqiues
        to recover image

- Use compressed sensing
    - Images must be compressible (sparse representation in some transformation)
    - Incoherence between sampling and sparsity domain for unique and attainable solution
    - Can be acheived using random undersampling of k-space which produces correlated noise
    - Image can then be reconstructed using non-linear optimization
    - Learn optimal sparse transform from data directly using dictionary learning.

- For more aggresive undersampling makes use of MR image redundancies
    - Spatio-Temporal redundencies in dynamic imaging
    - Redundancy from adjacent slices in 3D imaging
    - Explicit and Implicit redundancies

- Re-construction problem is de-aliasing problem in image domain

- SOTA
    - 2D reconstuction: Dictionary learning MRI
    - Dynamic reconstruction:
        - Dictionary learning with temporal gradient (DLTG)
        - kt sparse and low rank (kt-SLR)
        - low rank plus sparse matrix decomposition (L + S)

- Code available at: https://github.com/js3611/Deep-MRI-Reconstruction
    - Uses Theano :(

- Reconstruction times
    - 2D:      23ms
    - Dynamic: 10s

- Lots of math which i did not understand