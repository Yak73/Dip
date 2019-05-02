from skimage import color, io
import os

import lab2_test

AMOUNT_OF_CLASS = 4


def create_dataset(sample_number, type_sample="Training/segmented"):
    dataset = []
    if sample_number == -1:
        os.chdir(type_sample)
    else:
        os.chdir(type_sample + "/" + str(sample_number))
    img_list = os.listdir()
    for img in img_list:
        image = io.imread(img)
        img_hsv = color.rgb2hsv(image)
        dataset.append(img_hsv)
    if sample_number == -1:
        os.chdir(os.getcwd() + "/../../")
    else:
        os.chdir(os.getcwd() + "/../../../")
    return dataset


def calc_avg_color(dataset):
    avg_color = 0
    cntr_white_pixels = 0
    for n, img in enumerate(dataset):
        for i in range(dataset[0].shape[0]):
            for j in range(dataset[0].shape[1]):
                try:
                    # print("img[i][j][0] ", img[i][j][0])
                    # print("img[i][j][1] ", img[i][j][1])
                    d = img[i][j][0] + img[i][j][1]  # white pixel on not
                except:
                    d = 0
                if d < 0.05:
                    cntr_white_pixels += 1
                else:
                    avg_color += img[i][j]
    denominator = dataset[0].shape[0] * dataset[0].shape[1] * len(dataset) - cntr_white_pixels
    # print(denominator, avg_color)
    avg_color /= denominator
    return avg_color


def get_train_avg_color_set(show_bit=0):
    train_dataset = []
    avg_color_set = []
    for i in range(AMOUNT_OF_CLASS):
        train_dataset.append(create_dataset(i))
        avg_color_set.append(calc_avg_color(train_dataset[i]))

    if not show_bit:
        return avg_color_set

    for i in range(AMOUNT_OF_CLASS):
        print(lab2_test.get_label_class(i))
        print(avg_color_set[i])
    return avg_color_set


if __name__ == '__main__':
    pass
