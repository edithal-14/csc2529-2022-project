import numpy as np
import skimage.io as io
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import gaussian_filter

# from src.postprocessing.StyleTransfer import perform_style_transfer
import numpy
from scipy.ndimage import sobel


def apply_sobel_filter(image):
    horizontal = sobel(image.astype(np.int32), 0)
    vertical = sobel(image.astype(np.int32), 1)
    sobel_img = numpy.hypot(horizontal, vertical)
    sobel_img = (sobel_img * 255.0 / numpy.max(sobel_img)).astype(np.uint8)
    io.imsave('sobel_image.png', sobel_img)
    return sobel_img


def pixel_average(x, y, channel, images):
    total = 0

    for i in range(len(images)):
        total += images[i][x, y, channel]

    return total / len(images)


def calculate_weights(base_image, images):
    psnrs = []
    ssims = []
    for image in images:
        psnrs.append(psnr(base_image, image))
        ssims.append(ssim(base_image, image))


def aggregate_images(orig_images):  # array of 6 images from each GAN model
    agg_img = np.zeros_like(orig_images[0])

    images = []
    for image in orig_images:
        images.append(gaussian_filter(sobel_image(image), sigma=5))

    for row in range(agg_img.shape[0]):
        for col in range(agg_img.shape[1]):
            for channel in range(3):
                agg_img[row, col, channel] = pixel_average(row, col, channel, images)

    file_name = "aggregated_image.png"
    agg_img = np.clip(agg_img, a_min=0, a_max=255).astype(np.uint8)
    io.imsave(file_name, agg_img)
    return file_name


img1 = cv2.resize(cv2.imread('s1.jpg'), dsize=(512, 512), interpolation=cv2.INTER_CUBIC)
img2 = cv2.resize(cv2.imread('s2.jpg'), dsize=(512, 512), interpolation=cv2.INTER_CUBIC)
img3 = cv2.resize(cv2.imread('s3.jpg'), dsize=(512, 512), interpolation=cv2.INTER_CUBIC)
img4 = cv2.resize(cv2.imread('s4.jpg'), dsize=(512, 512), interpolation=cv2.INTER_CUBIC)
img5 = cv2.resize(cv2.imread('s6.jpg'), dsize=(512, 512), interpolation=cv2.INTER_CUBIC)
img6 = cv2.resize(cv2.imread('s7.jpg'), dsize=(512, 512), interpolation=cv2.INTER_CUBIC)

# images = [img1, img2, img3, img4, img5, img6]
#
# content_image = aggregate_images(images)
style_image = 'style_mri.jpg'
sobel_image = apply_sobel_filter(img1)

agg_img = np.clip(sobel_image, a_min=0, a_max=255).astype(np.uint8)
io.imsave('sobel_img.png', sobel_image)

# apply_sobel_filter(style_image)
# perform_style_transfer(content_image, style_image)
