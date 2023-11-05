# Edge Detector
import os
import numpy as np

from PIL import Image
from scipy.ndimage import convolve
from diffusers.utils import load_image


def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def gaussian_kernel(size, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size : size + 1, -size : size + 1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g = np.exp(-((x**2 + y**2) / (2.0 * sigma**2))) * normal
    return g


def sobel_filters(img):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

    Ix = convolve(img, Kx)
    Iy = convolve(img, Ky)

    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)
    return (G, theta)


def non_max_suppression(img, D):
    M, N = img.shape
    Z = np.zeros((M, N), dtype=np.int32)
    angle = D * 180.0 / np.pi
    angle[angle < 0] += 180

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            try:
                q = 255
                r = 255

                # angle 0
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = img[i, j + 1]
                    r = img[i, j - 1]
                # angle 45
                elif 22.5 <= angle[i, j] < 67.5:
                    q = img[i + 1, j - 1]
                    r = img[i - 1, j + 1]
                # angle 90
                elif 67.5 <= angle[i, j] < 112.5:
                    q = img[i + 1, j]
                    r = img[i - 1, j]
                # angle 135
                elif 112.5 <= angle[i, j] < 157.5:
                    q = img[i - 1, j - 1]
                    r = img[i + 1, j + 1]

                if (img[i, j] >= q) and (img[i, j] >= r):
                    Z[i, j] = img[i, j]
                else:
                    Z[i, j] = 0

            except IndexError as e:
                pass

    return Z


def hysteresis(img, weak_pixel, strong_pixel):
    M, N = img.shape

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            if img[i, j] == weak_pixel:
                try:
                    if (
                        (img[i + 1, j - 1] == strong_pixel)
                        or (img[i + 1, j] == strong_pixel)
                        or (img[i + 1, j + 1] == strong_pixel)
                        or (img[i, j - 1] == strong_pixel)
                        or (img[i, j + 1] == strong_pixel)
                        or (img[i - 1, j - 1] == strong_pixel)
                        or (img[i - 1, j] == strong_pixel)
                        or (img[i - 1, j + 1] == strong_pixel)
                    ):
                        img[i, j] = strong_pixel
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img


def threshold(img, lowThreshold, highThreshold, weak_pixel, strong_pixel):
    highThreshold = img.max() * highThreshold
    lowThreshold = highThreshold * lowThreshold

    M, N = img.shape
    res = np.zeros((M, N), dtype=np.int32)

    weak = np.int32(weak_pixel)
    strong = np.int32(strong_pixel)

    strong_i, strong_j = np.where(img >= highThreshold)
    zeros_i, zeros_j = np.where(img < lowThreshold)

    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    return res


def detect(img, **config):
    sigma = config.get("sigma", 1)
    kernel_size = config.get("kernel_size", 5)
    weak_pixel = config.get("weak_pixel", 75)
    strong_pixel = config.get("strong_pixel", 255)
    low_threshold = config.get("low_threshold", 0.05)
    high_threshold = config.get("high_threshold", 0.15)

    img_smoothed = convolve(img, gaussian_kernel(kernel_size, sigma))
    gradient_mat, theta_mat = sobel_filters(img_smoothed)
    non_max_img = non_max_suppression(gradient_mat, theta_mat)

    threshold_img = threshold(
        non_max_img, low_threshold, high_threshold, weak_pixel, strong_pixel
    )

    return hysteresis(threshold_img, weak_pixel, strong_pixel)


def to_canny_image(url, **config):
    img = load_image(url)
    img = np.array(img)
    img = rgb2gray(img)
    img = detect(img, **config)
    img = Image.fromarray(np.uint8(img))
    return img
