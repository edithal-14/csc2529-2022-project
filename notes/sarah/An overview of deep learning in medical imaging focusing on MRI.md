An overview of deep learning in medical imaging focusing on MRI


Overview:
Key concepts of deep learning for clinical radiologists, including radiomics and imaging genomics. Deep learning in neuroimaging and neuroradiology; brain segmentation; stroke imaging; neuropsychiatric disorders; breast cancer; chest imaging; imaging in oncology; medical ultrasound.

Applications in the field of MRI, from acquisition to image retrieval, from segmentation to disease prediction, and in signal processing in MR fingerprinting, denoising and super-resolution, and image synthesis. We divide this into two parts: (i) the signal processing chain close to the physics of MRI, including image restoration and multimodal image registration, and (ii) the use of deep learning in MR image segmentation, disease detection, disease prediction and systems based on images and text data, addressing a few selected organs such as the brain, the kidney, the prostate and the spine.

Applications:
Image Acquisition and Reconstruction:
Using variational networks which enabled real-time image reconstruction. Research explored the potential for transfer learning (pretrained models) and assessed the generalization of learned image reconstruction regarding image contrast, SNR, sampling pattern and image content, using a variational network. Employing generative adversarial networks (GANs) that learns texture details and suppresses high-frequency noise that can produce diagnostic quality reconstructions “on the fly”. Automated transform by manifold approximation (AUTOMAP) consists of a feedforward deep neural network with fully connected layers followed by a sparse convolutional autoencoder, formulating image reconstruction generically as a data-driven supervised learning task.

Quantitative parameters – QSM and MR fingerprinting
MRF uses a pseudo-randomized acquisition that causes the signals from different tissues to have a unique signal evolution (“fingerprint”) that is a function of the multiple material properties being investigated. MRF reconstruction problem is learning an optimal function that maps the recorded signal magnitudes to the corresponding tissue parameter values, trained on a sparse set of dictionary entries.

Image restoration (denoising, artifact detection)
MSE denoising results and run-time comparisons were in favor of sDNN. Deep learning methods has also been applied to MRartifact detection, e.g. poor quality spectra in MRSI; detection and removal of ghosting artifacts in MR spectroscopy; and automated reference-free detection of patient motion artifacts in MRI.



Image super-resolution
Generating super-resolution single and multi-contrast brain MR images using CNNs; and super-resolution musculoskeletal MRI

Image synthesis
GANs can be used for biological image synthesis, text-to-image synthesis, data augmentation, and as an anonymization tool.

Image segmentation
Acute ischemic lesion segmentation in DWI; brain tumor segmentation; segmentation of the striatum; segmentation of organs-at-risks in head and neck CT images; and fully automated segmentation of polycystic kidneys; deformable segmentation of the prostate; and spine segmentation.

Content-based image retrieval
Provides medical cases similar to a given image in order to assist radiologists in the decision-making process.

Resources:
https://github.com/paras42/Hello World Deep Learning, where you’ll be guided through the construction of a system that can differentiate a chest X-ray from an abdominal X-ray using the Keras/TensorFlow framework through a Jupyter Notebook.

Other tutorials are http://bit.ly/adltktutorial, based on the Deep Learning Toolkit (DLTK) [131], and https://github.com/usuyama/ pydata-medical-image, based on the Microsoft Cognitive Toolkit (CNTK).

js3611/Deep-MRI-Reconstruction: Deep Cascade of Convolutional Neural Networks for MR Image Reconstruction: Implementation & Demo (github.com)

image-retrieval · GitHub Topics

Most of the main new ideas and methods: arXiv.org e-Print archive
