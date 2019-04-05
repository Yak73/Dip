# -*- coding: utf-8 -*-
"""
@author: Yar
34 Granadilla Strawberry Собель
"""

from skimage import color, io, filters
from skimage.measure import moments
import os
import matplotlib.pyplot as plt

FRUIT_1 = "Granadilla"
FRUIT_2 = "Strawberry"
NAME_OF_PROJECT = os.getcwd().split("\\")[-1]


def create_dataset_with_filter(fruit, type_sample="Training"):
    """
    Create dataset of image in folder "/type_sample/fruit"
    and apply filter to each

    ARGS:
    fruit - name of folder with fruits
    type_sample - name of root folder (Training or Testing)

    RETURN VALUE:
    list: dataset
    """
    dataset = []
    os.chdir(type_sample + "/" + fruit)
    img_list = os.listdir()
    for img in img_list:
        image = io.imread(img)
        img_grayscale = color.rgb2gray(image)
        edges = filters.sobel(img_grayscale)  # FILTER
        dataset.append(edges)
    os.chdir(os.getcwd() + "/../../../" + NAME_OF_PROJECT)
    return dataset


def calculate_avg_moment(dataset):
    """
    Calculate average moment for dataset

    ARGS:
    dataset - list of images

    RETURN VALUE:
    average value of moment
    """
    summ = 0
    for img in dataset:
        summ += float(moments(img, order=0))
    return summ/len(dataset)
    
    
def decision(fruit1, avg_momnt1, avg_momnt2, type_sample="Testing"):
    """
    Decision which fruit fits each image and calculate error

    ARGS:
    fruit1 - name of 1st fruit
    avg_moment1 - average moment value of dataset[fruit1]
    avg_moment2 - average moment value of dataset[fruit2]
    type_sample - name of root folder (Training or Testing)
    RETURN VALUE:
    list with error for each image
    (If wrong decision: value error have minus)
    """
    os.chdir(type_sample + "/" + fruit1)
    img_list = os.listdir(path=os.getcwd())
    decision_list = []
    for img in img_list:
        image = io.imread(img)
        img_grayscale = color.rgb2gray(image)
        edges = filters.sobel(img_grayscale)  # FILTER
        cur_moment = float(moments(edges, order=0))
        err1 = abs(cur_moment - avg_momnt1)
        err2 = abs(cur_moment - avg_momnt2)
        if err1 < err2:
            decision_list.append(err1)
        else:
            decision_list.append(-1*err2)
    os.chdir(os.getcwd() + "/../../../" + NAME_OF_PROJECT)
    return decision_list
    

def plot(y1, y2):
    """
    Build and show error graph for each image

    ARGS:
    y1 - 1st function decision errors
    y2 - 2nd function decision errors

    """
    max_len = max(len(y1), len(y2))
    x1 = range(len(y1))
    x2 = range(len(y2))
    fig, ax = plt.subplots()
    ax.plot(x1, y1, label=FRUIT_1)
    ax.plot(x2, y2, label=FRUIT_2)
    ax.set_title('Decision errors: testing_sample')
    ax.legend(loc='lower right')
    ax.set_ylabel('Error')
    ax.set_xlabel('Number image in sample')
    ax.set_xlim(xmin=0, xmax=max_len)
    plt.show()
    

if __name__ == '__main__': 
    dataset_train_1 = create_dataset_with_filter(FRUIT_1)
    dataset_train_2 = create_dataset_with_filter(FRUIT_2)
    
    avg_moment1 = calculate_avg_moment(dataset_train_1)
    avg_moment2 = calculate_avg_moment(dataset_train_2)
    
    res_list_1 = decision(FRUIT_1, avg_moment1, avg_moment2)
    res_list_2 = decision(FRUIT_2, avg_moment2, avg_moment1)
    
    plot(res_list_1, res_list_2)
    
    #print(len([x for x in res_list_1 if x<0]))
