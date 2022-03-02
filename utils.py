import cv2
from matplotlib import pyplot as plt


def trans_to_visible_images(inp):  # insize: [None, 224, 224, 3]
    return (inp + 1) / 2


def plot_16_image(inp):  # insize: [16, 224, 224, 3]
    plt.clf()
    im = trans_to_visible_images(inp)
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.axis('off')
        plt.imshow(im[:, :, :, [2, 1, 0]][i])
