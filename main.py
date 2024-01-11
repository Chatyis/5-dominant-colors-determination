import math
import re
import sys
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from skimage import color
from skimage import graph
from skimage import segmentation
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
from colormath.color_objects import sRGBColor, LabColor
import time

def patch_asscalar(a):
    return a.item()


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

    if(len(map_of_colors) == 1):
        display_colors([list(map_of_colors.keys())[0]])
        sys.exit()

    return map_of_colors


def coefficient_map(_occurrence_map, _contrast_map, _saturation_map):
    _coefficient_map = {}
    for key in _occurrence_map.keys():
        _coefficient_map[key] = _occurrence_map[key] + _contrast_map[key] + _saturation_map[key]
    return _coefficient_map


def erode_image(image):
    return cv.erode(image,np.ones((2, 2), np.uint8), iterations=1)


def initial_dominant_color_list(occurrence_map, saturation_map):
    most_saturated_color = max(saturation_map, key=saturation_map.get)
    occurrence_map.pop(most_saturated_color)
    return [max(saturation_map, key=saturation_map.get), max(occurrence_map, key=occurrence_map.get)]


def string_rgb_to_srgb(rgb_string):
    _rgb_string = [int(_color) for _color in filter(None, re.split(r',\s*|\s+',rgb_string[1:-1]))]
    return sRGBColor(_rgb_string[2], _rgb_string[1], _rgb_string[0])


def get_hue_angle_value(color_lab):
    a = color_lab.get_value_tuple()[1]
    b = color_lab.get_value_tuple()[2]
    if a == 0:
        if b > 0:
            return 90
        elif b < 0:
            return 270
        else:
            return 0
    else:
        atan_value = math.degrees(math.atan(b/a))

    if a >= 0 and b >= 0:
        return atan_value
    elif a < 0 <= b:
        return atan_value + 180
    elif a < 0 and b < 0:
        return atan_value + 180
    else:
        return atan_value + 360


def weight_map(_coefficient_map, _dominant_colors_list):
    # for each candidate
    # print(_coefficient_map)
    _weight_map = _coefficient_map
    delta_hue_map = {}
    delta_color_map = {}
    for dominant_color in _dominant_colors_list:

        for candidate_key in _coefficient_map.keys():
            candidate_color = string_rgb_to_srgb(candidate_key)
            candidate_color_lab = convert_color(candidate_color, LabColor)

            _dominant_color = string_rgb_to_srgb(dominant_color)
            dominant_color_lab = convert_color(_dominant_color, LabColor)

            delta_hue_map[candidate_key] = get_hue_angle_value(dominant_color_lab) - get_hue_angle_value(candidate_color_lab)
            delta_color_map[candidate_key] = delta_e_cie2000(dominant_color_lab, candidate_color_lab)

        std_hue = np.std(list(delta_hue_map.values()))
        std_color = np.std(list(delta_color_map.values()))

        for candidate_key in delta_hue_map.keys():
            w1 = 1 - math.exp(-(delta_hue_map[candidate_key]**2/std_hue**2))
            w2 = 1 - math.exp(-(delta_color_map[candidate_key]**2/std_color**2))

            _weight_map[candidate_key] = _weight_map[candidate_key] * w1 * w2

    return _weight_map


def display_colors(colors_list):
    fig, ax = plt.subplots()
    for iterator, _color in enumerate(colors_list):
        rgb_string = [int(_color) for _color in filter(None, re.split(r',\s*|\s+', _color[1:-1]))]
        ax.add_patch(plt.Rectangle((1.5*iterator, 0), 1.5, 1.5, color=(rgb_string[2]/255, rgb_string[1]/255, rgb_string[0]/255)))

    ax.set_xlim(0, (len(colors_list))*1.5)
    ax.set_yticks([])
    ax.set_xticks([])

    dominant_colors_list_RGB = list(map(lambda x: ' '.join(list(reversed(re.split(r',\s*|\s+', x[1:-1])))), colors_list))
    figtext_args = (0.5, 0.05,'[' + '] | ['.join(dominant_colors_list_RGB) + ']')
    figtext_kwargs = dict(horizontalalignment="center", fontsize=10, wrap=True)

    # plt.figtext(0.5,0,'['+'],['.join(dominant_colors_list_RGB)+']')
    plt.figtext(*figtext_args, **figtext_kwargs)
    print(time.time() - start_time)
    plt.show()


if __name__ == '__main__':
    start_time = time.time()
    setattr(np, "asscalar", patch_asscalar)

    clusters_amt = 32
    rag_cut_threshold = 25
    max_img_size = 250
    skip_rag = True

    try:
        img = cv.imread('data/input_images/full_moon.jpg')
        if img is None:
            raise FileNotFoundError("Image not found or couldn't be loaded")

        img = resize_image(img, max_img_size)

        # bilateral filtering https://www.geeksforgeeks.org/python-bilateral-filtering/
        img = cv.bilateralFilter(img, 15, 75, 75)

        cv.imwrite('data/output_images/bilateral.jpg', img)

        # K-means https://docs.opencv.org/3.4/d1/d5c/tutorial_py_kmeans_opencv.html https://scikit-learn.org/stable/auto_examples/cluster/plot_color_quantization.html
        img = calculate_k_means(clusters_amt, img)

        cv.imwrite('data/output_images/k_means.jpg', img)

        # RAG
        # https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_rag_draw.html
        # https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_rag_mean_color.html#sphx-glr-auto-examples-segmentation-plot-rag-mean-color-py - graph cut
        # https://pypi.org/project/img2rag/

        # TODO Threshold has problems on dark images,
        # TODO also RAG generates black artifact most often in top left corner
        # TODO most propably issues are connected with bad rag parameters or low quality method
        if not skip_rag:
            img = perform_rag(img, rag_cut_threshold)
            cv.imwrite('data/output_images/rag_cut.jpg', img)

        # https://stackoverflow.com/questions/12282232/how-do-i-count-occurrence-of-unique-values-inside-a-list
        # https://docs.python.org/3/library/collections.html#collections.Counter
        # https://stackoverflow.com/questions/28663856/how-do-i-count-the-occurrence-of-a-certain-item-in-an-ndarray

        # calculate contrast values for each color
        _occurrence_map = occurrence_map(img)
        contrast_map, saturation_map = contrast_and_saturation_map(img)

        # pk = Ck + Ak + Sk
        _coefficient_map = coefficient_map(_occurrence_map,contrast_map,saturation_map)

        # erosion
        img = erode_image(img)

        cv.imwrite('data/output_images/erode.jpg', img)

        # initialising final dominant colors list
        dominant_colors_list = initial_dominant_color_list(_occurrence_map, saturation_map)

        # adding three more values to the final dominant colors list
        for i in range(min(len(_coefficient_map)-2,3)):
            _weight_map = weight_map(_coefficient_map, dominant_colors_list)
            dominant_colors_list.append(max(_weight_map, key=_weight_map.get))
        # print(dominant_colors_list)
        display_colors(dominant_colors_list)
    except FileNotFoundError as e:
        print(e)
        sys.exit()
