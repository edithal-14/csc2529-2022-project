### Recent advances in ML for medical image analysis

- This is a 3 year old paper written in 2019 so information might not be up to date

- Abstract
    - Complex tasks like classifying, segmenting, localization
        and recognizing objects of interest have become much less
        challenging
    - In reality limmited practical examples being deployed in front-
        line healthcare. Discuss various challenges that are considered
        barriers to acccelerating research
    - Discuss future directions

- Introduction
    - Classify COVID-19 positive based on chest X ray images
    - Scale invariant features
        - Invariant to scaling, rotation, pose and illumination
        - SIFT and SURF
    - Usage of handcrafted features to detect objects from ultrasound
        image
    - Deep networks can automatically learn interesting features
        without much supervision
    - Transfer learning is required sometimes
        - Lack of huge amount of annotated data
        - Using pre-trained models used for other general task
    - Object detection and recognition models
        - R-CNN
        - SSD
        - R-RFCN
        - YOLO
    - Image segmentation
        - Robot assisted surgery education
    - Datasets
        - MURA: 40,561 musculoskeletal radiographs
        - Colon cancer screening dataset
        - Lung CT scan abnormality images
        - CAMUS: 250 images of hearts where only LA and LV were segmented

- Medical images
    - Mass screening for COVID-19 usig thermal camera
    - Thermal imaged for facial paralysis (93% accutacy)
    - Remote health monitoring and treatment

- Object detection
    - Used to detect abnormality
    - Two methods, anchor free and anchor based
        - Anchor based: single stage and multistage
        - multistage is more accurate but computationally expensive
        - single stage
            - YOLO, SSD
        - two stage
            - Region of interest, region proposal network (RPN)
    - YOLO for gallstone detection from CT scans
    - Research on anchor free methods of object detection: CornerNet
        - Single CNN which uses paired key points

- Segmentation
    - Pixel wise classification
    - Generates a pixel mask
    - U-Net is the most famous end-to-end FCN used for segmentation
    - U-Net with GAN used to segment the whole heart from CT image
    - Attention mechanism is used
    - cGAN (conditional generative adversarial network)
    - Segment chambers of the heart from MRI images
        - Weighted ensemble of DL methods using particle swarm optimization

- Challenges
    - Data annotation for creating datasets is challenging
    - Many of the challenges are related to data quality and data availability 
    - More work needs to be done in the area of unsupervised and semi-supervised learning
    - Explanability to all stakeholders

- Image modalities
    - CT
    - MRI
    - Ultrasound
    - X-rays

- Datasets
    - High levels for variation in the data collected. E.g. Thermal imaging
        - This causes the model to not generalize well
    - Data bias
        - High class imbalance
        - Can be solved by adjusting the cost function according to the imbalance ratio
        - GANs play an important role in data augmenting of the minority class
            - Synthetically generated images can improve medical image segementation

- Research Areas
    - Interpretable CV models
    - Data generation using GANs