# David Lindell Office hours

- Project discussion

- 31st October 2022

### GAN synthetic brain tumor dataset generation
    - Motivation
        - Scarcity of original medical data
        - Class imbalance problem
    - Introduction
        - Standalone GAN can only fetch the local features
        - Combining GANs might understand distributed features
    - Aggregation of GANs
        - Two variants of DCGAN
        - WGAN
    - Also applied style transfer technique to increase image resemblance
    - Two publibly available datasets
        - Brain tumor dataset
        - Multimodal brain tumor segmentation challenge (BraTS)
    - Proposed model generate fine-quality images with maximum structural similarity index measure of 0.57 and 0.83 on both the datasets

### Auto Whole Heart Segmentation from CT images using an improved Unet-GAN
    - Challenge
        - High anatomical and signal intensity variations.
        - Lack of high level labeled dataset
    - Proposed model
        - R2Unet - GAN
    - Dice similarity coefficient used for evaluation
    - Personal observation
        - Numerous grammatical mistakes in the paper
    - Future work
        - Focus on slices with long-scale contrast
        - Optimize algorithm with novel pre-processing

### Which project should we choose?
    - Both are reasonable, but choose the one which aligns best with your research

### What novelties can we add on top of the research paper?
    - Model changes
        - Use different activation function
        - Use different loss function
        - Add batch Normalization
    - Add more interpretability
    - Ablation study
        - What happens if you remove a component
    - Dont just change the learning rate and train it    
        - Adjustments to overcome the limitations
        - (u-net or style GAN) can work better and provide more controllability on the image to be generated
    - Where the tumor, type, how big it is

### What are you looking for while grading?
    - Cover all parts of the project report
        - Introduction
            - Explain baseline methods
        - Related works
        - Evaluation
            - How realistic the image looks
            - Compare with baselines
            - Qualitative results
        - Error analysis
            - In case of decreased performance try to explain that
            - Ablation studies are helpful in this case
        - References
    - Poster presentation
        - December 8th
            - Makeup day for thanksgiving
            - 10am to 12pm
        - Instructor and TAs will grade presentation
        - Industry and research community people will also attend


### Can you point us to interesting literature in this domain
    - Style GAN
        - Contralability of what images are generated
    - Style GAN 2
    - Sofgan
        - More interpretability and controlability
    - Hypergan
    - Interfacegan
    - Look at latent space and how to sample points
        - Modify attributes
        - Interpreting the Latent Space of GANs for Semantic Face Editing
            - https://arxiv.org/abs/1907.10786
    - Biomedical image synthesis
    - StyleFlow
        - https://github.com/RameenAbdal/StyleFlow
        - https://rameenabdal.github.io/StyleFlow/ 