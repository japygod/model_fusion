import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt

#import theano
import cv2
import numpy as np
import scipy as sp
import sys
import csv
import torch
from torchvision import models
from torchvision import transforms

import torch.nn as nn
import h5py
import time
import os
import codecs
from PIL import Image
from PIL import ImageFile
import numpy
# K.set_image_data_format('channels_first')
ImageFile.LOAD_TRUNCATED_IMAGES = True
trans = transforms.Compose(
    [
        transforms.ToTensor(),

    ])


# def get_features(model, layer, X_batch):
#     get_feature = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer].output])
#     img_features = get_feature([X_batch, 0])
#     return img_features


# def VGG_16(weights_path=None):
#     model = Sequential()
#     model.add(ZeroPadding2D((1, 1), input_shape=(3, 224, 224)))
#     model.add(Convolution2D(64, 3, 3, activation='relu'))
#     model.add(ZeroPadding2D((1, 1)))
#     model.add(Convolution2D(64, 3, 3, activation='relu'))
#     model.add(MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_first'))
#
#     model.add(ZeroPadding2D((1, 1)))
#     model.add(Convolution2D(128, 3, 3, activation='relu'))
#     model.add(ZeroPadding2D((1, 1)))
#     model.add(Convolution2D(128, 3, 3, activation='relu'))
#     model.add(MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_first'))
#
#     model.add(ZeroPadding2D((1, 1)))
#     model.add(Convolution2D(256, 3, 3, activation='relu'))
#     model.add(ZeroPadding2D((1, 1)))
#     model.add(Convolution2D(256, 3, 3, activation='relu'))
#     model.add(ZeroPadding2D((1, 1)))
#     model.add(Convolution2D(256, 3, 3, activation='relu'))
#     model.add(MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_first'))
#
#     model.add(ZeroPadding2D((1, 1)))
#     model.add(Convolution2D(512, 3, 3, activation='relu'))
#     model.add(ZeroPadding2D((1, 1)))
#     model.add(Convolution2D(512, 3, 3, activation='relu'))
#     model.add(ZeroPadding2D((1, 1)))
#     model.add(Convolution2D(512, 3, 3, activation='relu'))
#     model.add(MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_first'))
#
#     model.add(ZeroPadding2D((1, 1)))
#     model.add(Convolution2D(512, 3, 3, activation='relu'))
#     model.add(ZeroPadding2D((1, 1)))
#     model.add(Convolution2D(512, 3, 3, activation='relu'))
#     model.add(ZeroPadding2D((1, 1)))
#     model.add(Convolution2D(512, 3, 3, activation='relu'))
#     model.add(MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_first'))
#
#     if weights_path:
#         model.load_weights(weights_path)
#
#     return model


def compute_mean_pixel():
    img_dir = r'F:/experiment/data/img'
    img_list = os.listdir(img_dir)
    img_size = 224
    sum_r = 0
    sum_g = 0
    sum_b = 0
    count = 0
    # image = Image.open(r'F:\experiment\data\img\71448fae0e082172abbc9d618533c4ad.jpg')
    # image = image.resize((img_size, img_size))
    # image = numpy.asarray(image)
    # if image.mode == 'P':
    #     image = image.convert('RGB')
    # if image.mode == 'RGBA':
    #     image = image.convert('RGB')
    # image = image.resize((img_size, img_size))
    #
    # print(numpy.asarray(image))
    # img_pixel = cv2.imread(r'F:\experiment\data\img\0e1f75ec8c326029d715bd39282ad522.png')
    # print(image.mode)
    for img_name in img_list:
        img_path = os.path.join(img_dir, img_name)
        print(img_path)
        image = Image.open(img_path)
        if image.mode == 'P':
            image = image.convert('RGB')
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        if image.mode == 'L':
            image = image.convert('RGB')
        image = image.resize((img_size, img_size))
        image = numpy.array(image)
        sum_r = sum_r + image[:, :, 0].mean()
        sum_g = sum_g + image[:, :, 1].mean()
        sum_b = sum_b + image[:, :, 2].mean()
        count = count + 1
    sum_r = sum_r / count
    sum_g = sum_g / count
    sum_b = sum_b / count
    img_mean = [sum_r, sum_g, sum_b]
    print(img_mean)


class Model():
    def __init__(self, model):
        self.model = model
    def show(self, image):
        x = image
        for index, layer in enumerate(self.model):
            # print(index,layer)
            x = layer(x)
        return x

def extract_feature():
    tweet_data_path = r'..\data\train.csv'  # all.txt store
    # store_img_feature = r'..\data\img_vgg_feature_224.h5'  # image feature stored file
    # vgg_img_feature = h5py.File(store_img_feature, "w")  # store the output -- img feature vector
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    mean_pixel = [148.385, 142.428, 137.559]
    pretrained_model = models.vgg16(pretrained=True).features
    pretrained_model.to(device)

    model = Model(pretrained_model)

    img_dir = r'..\data\img'
    img_list = os.listdir(img_dir)
    img_feature_path = r'..\data\img_feature'


    for item in img_list:
        print("process " + item)
        img_path = os.path.join(img_dir, item)
        image = Image.open(img_path)
        if image.mode == 'P':
            image = image.convert('RGB')
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        if image.mode == 'L':
            image = image.convert('RGB')
        image = image.resize((224, 224))
        image = numpy.array(image)
        for c in range(3):
            image[:, :, c] = image[:, :, c] - mean_pixel[c]
        image = trans(image).unsqueeze(0)
        start = time.time()
        image = image.to(device)

        features = model.show(image)

        torch.save(features, img_feature_path + item.replace('.', '_') + '.pt')
        print('%s feature extracted in %f  seconds.' % (img_path, time.time() - start))



if __name__ == '__main__':
    extract_feature()