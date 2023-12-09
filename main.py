import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from skimage import color
from skimage import graph
from skimage import segmentation


def resize_image(img, max_img_size):
    if(img.shape[0]>img.shape[1]):
        new_shape = (int((img.shape[1]/img.shape[0])*max_img_size), max_img_size)
    else:
        new_shape = (max_img_size,int((img.shape[0]/img.shape[1])*max_img_size))
    return cv.resize(img, new_shape)


def calculate_k_means(clusters_amt, img):
    img_as_rgb_vector = img.reshape((-1,3)) # Actually not a vector but width * height x 3 (RGB) so 2D
    img_as_rgb_vector = np.float32(img_as_rgb_vector)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv.kmeans(img_as_rgb_vector, clusters_amt, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    return res.reshape(img.shape)


def _weight_mean_color(graph, src, dst, n):
    diff = graph.nodes[dst]['mean color'] - graph.nodes[n]['mean color']
    diff = np.linalg.norm(diff)
    return {'weight': diff}


def merge_mean_color(graph, src, dst):
    graph.nodes[dst]['total color'] += graph.nodes[src]['total color']
    graph.nodes[dst]['pixel count'] += graph.nodes[src]['pixel count']
    graph.nodes[dst]['mean color'] = (graph.nodes[dst]['total color'] /
                                      graph.nodes[dst]['pixel count'])

def perform_rag(img, rag_cut_threshold):
    labels = segmentation.slic(img, compactness=10, n_segments=1000, start_label=1)
    g = graph.rag_mean_color(img, labels)
    # labels2 = graph.cut_threshold(labels, g, rag_cut_threshold)
    labels2 = graph.merge_hierarchical(labels, g, thresh=rag_cut_threshold, rag_copy=False,
                                       in_place_merge=True,
                                       merge_func=merge_mean_color,
                                       weight_func=_weight_mean_color)
    return color.label2rgb(labels2, img, kind='avg', bg_label=0)


def contrast_and_saturation_map(image):
    map_of_contrast = {}
    map_of_saturation = {}
    max_contrast_value = 0
    max_saturation_value = 0
    # grab the image dimensions
    height = image.shape[0]
    width = image.shape[1]
    # rgb to lab to get luminance
    image_cielab = color.rgb2lab(image)
    unique_cielab_values = np.empty((0,3), int)
    map_of_colors = {}
    for x in range(0, height):
        for y in range(0, width):
            color_value = str(image[x][y])
            # add empty element if key doesn't exist
            if not map_of_colors.get(color_value):
                map_of_colors[color_value] = (0, 0) #(amt, sum)
                unique_cielab_values = np.vstack((unique_cielab_values,np.array(image_cielab[x][y])))
            # for colors around
            for x_shift in range(-2, 2):
                for y_shift in range(-2, 2):
                    # check for borders
                    if 0 <= x + x_shift < height and 0 <= y + y_shift < width:
                        # when luminance of neighbour pixel is other than selected pixel
                        if image_cielab[x][y][0] != image_cielab[x+x_shift][y+y_shift][0]:
                            # add lum distance between selected and neighbour,
                            # increment amount for further normalisation
                            map_of_colors[color_value] = (map_of_colors[color_value][0]+1, map_of_colors[color_value][1]+abs(image_cielab[x+x_shift][y+y_shift][0] - image_cielab[x][y][0]))

    # setup map_of_contrast with contrast divided by amount, map_of_saturation with saturation calculated from a and b
    for i, key in enumerate(map_of_colors.keys()):
        map_of_contrast[key] = map_of_colors[key][1]/map_of_colors[key][0]
        #sqrt(a^2 + b^2)
        map_of_saturation[key] = np.absolute(unique_cielab_values[i][1] + unique_cielab_values[i][2]*1j)
        if map_of_contrast[key] > max_contrast_value:
            max_contrast_value = map_of_contrast[key]
        if map_of_saturation[key] > max_saturation_value:
            max_saturation_value = map_of_saturation[key]

    # normalise map_of_contrast, map_of_saturation
    for key in map_of_contrast.keys():
        map_of_contrast[key] = map_of_contrast[key] / max_contrast_value
        map_of_saturation[key] = map_of_saturation[key] / max_saturation_value

    return map_of_contrast, map_of_saturation


def occurrence_map(image):
    height = image.shape[0]
    width = image.shape[1]
    map_of_colors = {}

    for x in range(0, height):
        for y in range(0, width):
            color_value = str(image[x][y])
            if not map_of_colors.get(color_value):
                map_of_colors[color_value] = (0) #(amt)
            map_of_colors[color_value] += 1

    max_occurrence_value = 0

    for key in map_of_colors.keys():
        if map_of_colors[key] > max_occurrence_value:
            max_occurrence_value = map_of_colors[key]

    # normalise map_of_occurrence
    for key in map_of_colors.keys():
        map_of_colors[key] = map_of_colors[key] / max_occurrence_value

    return map_of_colors


if __name__ == '__main__':
    clusters_amt = 32
    rag_cut_threshold = 25
    max_img_size = 1000
    img = cv.imread('data/input_images/full_moon.jpg')

    img = resize_image(img, max_img_size)

    # bilateral filtering https://www.geeksforgeeks.org/python-bilateral-filtering/
    bilateral_output = cv.bilateralFilter(img, 15, 75, 75)

    cv.imwrite('data/output_images/bilateral.jpg', bilateral_output)

    # K-means https://docs.opencv.org/3.4/d1/d5c/tutorial_py_kmeans_opencv.html https://scikit-learn.org/stable/auto_examples/cluster/plot_color_quantization.html
    k_means_output = calculate_k_means(clusters_amt, bilateral_output)

    cv.imwrite('data/output_images/k_means.jpg', k_means_output)

    # RAG
    # https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_rag_draw.html
    # https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_rag_mean_color.html#sphx-glr-auto-examples-segmentation-plot-rag-mean-color-py - graph cut
    # https://pypi.org/project/img2rag/

    # TODO Threshold has problems on dark images,
    # TODO also RAG generates black artifact most often in top left corner
    # TODO most propably issues are connected with bad rag parameters or low quality method
    img = perform_rag(k_means_output, rag_cut_threshold)

    cv.imwrite('data/output_images/rag_cut.jpg', img)

    # https://stackoverflow.com/questions/12282232/how-do-i-count-occurrence-of-unique-values-inside-a-list
    # https://docs.python.org/3/library/collections.html#collections.Counter
    # https://stackoverflow.com/questions/28663856/how-do-i-count-the-occurrence-of-a-certain-item-in-an-ndarray
    img_as_vector = img.reshape(-1,3)
    values, counts = np.unique(img_as_vector, return_counts=True, axis=0)
    # print(np.unique(img, return_counts=True, axis=0))

    # calculate contrast values for each color
    occurrence_map = occurrence_map(img)
    map_of_contrast, map_of_saturation = contrast_and_saturation_map(img)
    print(occurrence_map,map_of_contrast,map_of_saturation)
