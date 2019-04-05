from skimage import color, io
import matplotlib.pyplot as plt

import lab2_train


def check_image(img, local_avg_color_set, show_bit=1):

    decision_class, decision_error = decision(img, local_avg_color_set)
    label = get_label_class(decision_class)

    print("Результат определения:")
    print(label)
    print("Ошибка определения: {:.3}".format(decision_error))

    if not show_bit:
        return

    fig, ax = plt.subplots(1, 2, squeeze=False)
    ax[0][0].set(title='HSV', xlabel="Ошибка: {:.3}".format(decision_error))
    ax[0][0].imshow(img)
    ax[0][1].set(title='RGB', xlabel=label)
    ax[0][1].imshow(color.hsv2rgb(img))
    io.show()


def decision(img, color_set):
    avg_img_color = lab2_train.calc_avg_color([img])
    min_sum = 3*100
    decision_class = -1
    # print(avg_img_color)
    for n, sample in enumerate(color_set):
        d_h = abs(avg_img_color[0] - sample[0])
        d_s = abs(avg_img_color[1] - sample[1])
        d_v = abs(avg_img_color[2] - sample[2])
        d_sum = (d_h**2 + 0.5*d_s**2 + 0.1*d_v**2)**0.5
        if d_sum < min_sum:
            min_sum = d_sum
            decision_class = n
    return decision_class, min_sum


def get_label_class(class_number):
    dict_class_label = {
        0: '0ой класс - Салат с капустой и огурцом',
        1: '1ый класс - Компот',
        2: '2ой класс - Гороховый суп',
        3: '3ий класс - Макароны с котлетами'
    }
    return dict_class_label[class_number]