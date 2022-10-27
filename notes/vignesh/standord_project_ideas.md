### Stanford project ideas

- Phase based motion magnification algortihm
    - By Itamar Terem
    - Motivation
        - What is phase based motion magnification
            - Motion in a specific temporal frequency band is amplified
            - E.g. Cardiac gated MRI, most motion around hart rate freq.
            - Incorporate this prior knowledge into the video inverse
              problem to improve motion magnification performance.

    - Related work
        - Lately people have started using cnn to learn spatial
          decomposition filters

    - Overview
        - Improve from SNR perspective
        - Objective function of OLS kind
        - Phase based motion magnification algo
            - Can be seen as a denoising problem where we are trying
              to magnify the known signal (prior)
        - Pipeline diagram
            - Blur video -> ADMM + DnCNN -> Motion magnification
              -> Qualitave (motion) and Quantitative comparision (SNR)
            - Blur video -> ADMM + Motion magnification -> Comparision
    - Milestones and timeline
        - Week 1: Implementation 
        - Week 2: Getting dataset to test algo
        - Week 3: Writing presentation/report/poster