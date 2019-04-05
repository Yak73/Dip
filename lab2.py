from skimage import color, io
import os

import lab2_train, lab2_test, lab2_segmentation

NAME_OF_PROJECT = os.getcwd().split("\\")[-1]


if __name__ == '__main__':
    avg_color_set = lab2_train.get_train_avg_color_set(show_bit=1)

    lab2_segmentation.detect_circles('Testing_2/photo7.jpg', show_bit=1)
    lab2_segmentation.segmentation(type_sample="Testing_2")

    print("\n")
    os.chdir("Testing_2/segmented")
    img_list = os.listdir()
    for img in img_list:
        image = io.imread(img)
        img_hsv = color.rgb2hsv(image)
        lab2_test.check_image(img_hsv, avg_color_set)
        print()

    os.chdir(os.getcwd() + "/../../../" + NAME_OF_PROJECT)


