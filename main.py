import cv2 as cv
import numpy as np


def calculate_k_means(clusters_amt, img):
    img_as_rgb_vector = img.reshape((-1,3)) # Actually not a vector but width * height x 3 (RGB) so 2D
    img_as_rgb_vector = np.float32(img_as_rgb_vector)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv.kmeans(img_as_rgb_vector, clusters_amt, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    return res.reshape(img.shape)


if __name__ == '__main__':
    clusters_amt = 32

    img = cv.imread('data/input_images/flowers.jpg')

    # bilateral filtering https://www.geeksforgeeks.org/python-bilateral-filtering/
    bilateral_output = cv.bilateralFilter(img, 15, 75, 75)

    cv.imwrite('data/output_images/bilateral.jpg', bilateral_output)

    # K-means https://docs.opencv.org/3.4/d1/d5c/tutorial_py_kmeans_opencv.html https://scikit-learn.org/stable/auto_examples/cluster/plot_color_quantization.html
    k_means_output = calculate_k_means(clusters_amt, bilateral_output)

    cv.imwrite('data/output_images/k_means.jpg', k_means_output)

    # RAG https://pypi.org/project/img2rag/
