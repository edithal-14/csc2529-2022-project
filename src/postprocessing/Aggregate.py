import numpy as np
import skimage.io as io
import cv2

from src.postprocessing.StyleTransfer import perform_style_transfer


def aggregate_images(images):  # array of 6 images from each GAN model
    agg_img = np.zeros_like(images[0])

    def pixel_average(x, y, channel, images):
        total = 0

        for i in range(len(images)):
            total += images[i][x, y, channel]

        return total / len(images)

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

images = [img1, img2, img3, img4, img5, img6]

content_image = aggregate_images(images)
style_image = 'style_mri.jpg'
perform_style_transfer(content_image, style_image)
