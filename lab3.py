from skimage import draw, transform, io, color, exposure
import numpy as np
import matplotlib.pyplot as plt
import os

LIGHT_COLOR = 255
DARK_COLOR = 50
NAME_OF_PROJECT = os.getcwd().split("\\")[-1]


def create_haar_sign(w1, w2, w3, h, alpha, show_bit=0):
    img = np.zeros((h, w1+w2+w3), dtype=np.uint8)
    rr, cc = draw.rectangle((0, 0), extent=(h, w1), shape=img.shape)
    img[rr, cc] = LIGHT_COLOR
    rr, cc = draw.rectangle((0, w1), extent=(h, w2), shape=img.shape)
    img[rr, cc] = DARK_COLOR
    if w3:
        rr, cc = draw.rectangle((0, w1+w2), extent=(h, w3), shape=img.shape)
        img[rr, cc] = LIGHT_COLOR

    img = transform.rotate(img, 360-alpha, resize=True, preserve_range=False)
    print(img.shape)
    if not show_bit:
        return img

    io.imshow(img)
    io.show()
    return img


def create_dataset():
    dataset = []
    os.chdir("Car")
    img_list = os.listdir()
    for img in img_list:
        image = io.imread(img)
        img_gray = color.rgb2gray(image)
        img_contrast = exposure.adjust_sigmoid(img_gray, cutoff=0.5, gain=100, inv=False)
        dataset.append((image, img_contrast))
    os.chdir(os.getcwd() + "/../../" + NAME_OF_PROJECT)
    return dataset


def pruning(img, haar_sign_img):
    coord = (-1, -1)
    max_value = -1
    size = haar_sign_img.shape
    for x in range(img.shape[0] - size[0]):
        for y in range(img.shape[1] - size[1]):
            cur_value = detection(img[x:x+size[0], y:y+size[1]], haar_sign_img)
            if cur_value > max_value:
                max_value = cur_value
                coord = x, y
    return coord, max_value


def detection(img, haar_sign_img):
    if img.shape != haar_sign_img.shape:
        print("ERROR: SIZES NOT EQUAL")
        raise IndexError

    threshold = 42 * haar_sign_img.shape[0] * haar_sign_img.shape[1] / 255
    light, dark = 0, 0
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if haar_sign_img[x][y] == LIGHT_COLOR/255:
                light += img[x][y]
            elif haar_sign_img[x][y] == DARK_COLOR/255:
                dark += img[x][y]
    if light - dark > threshold:
        return light - dark
    return -1


def show_zone(img, c_img, coord, size):
    c_img = color.gray2rgb(c_img)
    if coord != (-1, -1):
        rr, cc = draw.rectangle_perimeter(coord, extent=size, shape=img.shape)
        c_img[rr, cc] = (1, 0, 0)
        img[rr, cc] = (255, 0, 0)

    label = "Lorry found :)" if coord[0] != -1 else "Lorry not found :("
    fig, ax = plt.subplots(1, 2, squeeze=False)
    ax[0][0].set(title='Original')
    ax[0][0].imshow(img)
    ax[0][1].set(title='Grayscale (contrast)', xlabel=label)
    ax[0][1].imshow(c_img)
    io.show()


if __name__ == '__main__':

    car_dataset = create_dataset()
    haar_sign = create_haar_sign(w1=4, w2=2, w3=24, h=4, alpha=19, show_bit=0)

    for n, image in enumerate(car_dataset):
        coordinates, value = pruning(image[1], haar_sign)
        print(coordinates, value)
        show_zone(image[0], image[1], coordinates, haar_sign.shape)