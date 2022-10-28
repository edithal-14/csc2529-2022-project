# Possible Applications for MRIs

	- Efficiency improvement in radiology
		- Short-text classification
	- Gadolinium dosage in contrast-enhanced brain MRI
	- CLE diagnostics via deep learning (Brain tumor surgery)
		- Confocal Laser Endomicroscopy (CLE)
		- [Prospects for theranostics in neurosurgical imaging: empowering confocal laser endomicroscopy diagnostics via deep learning](https://www.frontiersin.org/articles/10.3389/fonc.2018.00240/full)
			- Two major approaches reviewed in this paper include the models that can automatically classify CLE images into diagnostic/ND, glioma/nonglioma, tumor/injury/normal categories, and models that can localize histological features on the CLE images using weakly supervised methods
		- [Molecular imaging of the tumor microenvironment for precision medicine and theranostics](https://www.sciencedirect.com/science/article/pii/B9780124116382000070)
	- Dynamic MR Image reconstruction
		- Reconstructing good quality cardiac MR images from highly undersampled complex-valued k-space data by learning spatia-temporal dependencies
		- Convolutional Recurrent Neurel Nets
			- [Convolutional Recurrent Neural Networks for Dynamic MR Image Reconstruction](https://ieeexplore.ieee.org/abstract/document/8425639)
		- Deep Cascade CNN
			- [A Deep Cascade of Convolutional Neural Networks for MR Image Reconstruction](https://link.springer.com/chapter/10.1007/978-3-319-59050-9_51)
			- Potential improvements, model resiliency to radial and spiral undersampling
	- MR Image reconstruction for Fast Compressive Sensing
		- Fast MRI is highly in demand
		- Improves patient experience, reduces motion artefacts and contrast washout
		- Lower bound on sampling frequency (Nyquist-Shannon criteria)
		- conditional GANs for de-aliasing and image reconstruction from undersampled images
		- [Deep De-Aliasing for Fast Compressive Sensing MRI](https://arxiv.org/abs/1705.07137)

# Object Detection in X-Rays and CTs

	- Lung nodules in chest CTs
		- [AWEUNet: an attentionaware weight excitation unet for lung nodule segmentation](https://arxiv.org/abs/2110.05144)
	- Whole heart segmentation from CT images
		- [Auto whole heart segmentation from CT images using an improved UnetGAN](https://www.researchgate.net/publication/349544584_Auto_Whole_Heart_Segmentation_from_CT_images_Using_an_Improved_Unet-GAN)

# Possible open-source dataset

	- https://openneuro.org/search/modality/mri

# Synthetic Data Generation

	- GANs to generate synthetic abnormal MRI images with brain tumors
		- [pix2pix](https://phillipi.github.io/pix2pix)
