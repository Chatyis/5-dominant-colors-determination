import cv2

if __name__ == '__main__':
    img = cv2.imread('data/input_images/panda.jpg')

    bilateralOutput = cv2.bilateralFilter(img, 15,75,75)

    cv2.imwrite('data/output_images/bilateral.jpg',bilateralOutput)

    # K-means https://scikit-learn.org/stable/auto_examples/cluster/plot_color_quantization.html