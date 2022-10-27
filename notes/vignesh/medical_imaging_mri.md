### An overview of medical imaging focusing on MRI

- Abstract
    - Brief overview of deep learning with pointers to core resources
    - How deep learning is applied to MRI pipeline
    - Point out interesting data sources and open source codebases for problems related to medical imaging

- Introduction
    - Breakthroughs
        - ImageNet Large Scale Visual Recognition Challenge
            - CNN halved second best error rate
            - This is solved now
            - CNNs have surpasses humans
    - Famous application of re-inforcement learning
        - AlphaGo and AlphaZero, the Go playing machine learning system developed by DeepMind
    
    - Function of artificial neural networks

    - Building blocks of CNN
        - Preserving spatial relationship in the data
        - Convolutional layer
        - Activationd layer
            - Creates feature maps
        - Pooling (Max or mean)
            - Provides translational variance
            - Provides downsampling efffect
                - Could also use filters with more stride length
        - Dropout
            - Averaging method based on stochastic sampling
            - Generates ensembles
        - Batch normalization
            - Generally placed after activation layer
            - Subtract by mean and divide by std. dev.
            - Works as a regularizer
            - Speeds up training, less dependent on parameter initialization
        - Fully connected layer
            - Final layer for regression/classification
    
    - Famous CNN networks: See table

    - Image reconstruction using CNNs
        - Low latency models (30ms)
    
    - MRI pipeline
        - Image acquisition (in complex-valued k-space)
        - Image re-construction
        - Image restoration (de-noising)
        - Image registration
    
    - Quantitative Susceptibility mapping (QSM)
        - QSM is now feasible in standard clinical practice
        - Magnetic resonance fingerprinting
    
    - Magnetic Resonance fingerprinting (MRF)
        - Mapping signals back to tissue parameters is a difficult inverse problem
        - Deep learning
            - Cohen et. al.
            - Map signal magnitude to tissue parameter trained on sparse dictionary
            - MRD Deep RecOnstruction NEtwork (DRONE)
        - Predict quantitative parameters (T1 and T2) from MRF time series
            - Hoppe et. al.
    
    - Denoising in MRI
        - Bermudez et. al.
            - Autoencoder with skip connections for image denoising
                - Outperformed FSL SUSAN denoising software

    - Image super resolution
        - E.g. Deep Resolve
    
    - GANs for biological image synthesis
        - Serve as a form of data augmentation
        - Anonymization tool
        - MR to CT synthesis using unpaired data
    
    - Image registration
        - Mathematically speaking, it is a challenging mix of geometry, analysis
          optimization strategies and numerical schemes
    
    - Medical image segmentation
        - Multi-spectral image classification report by Vannier et. al. (1985)
            - One of the most seminal work in this field
    
    - ImageNet ILSVRC competition is one of the main driver of progress in computer vision since 2012

    - 3D image data can be computationally expensive to deal with

    - Transfer learning alleviates the problem of generalizability