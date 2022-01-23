import random
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
import matplotlib.image as mpimg
from imgaug import augmenters as iaa
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam


def center_left_right(path):
    return path.split('\\')[-1]


def histograma(coluna):
    plt.figure(figsize=(15, 5))
    sns.histplot(coluna, kde=True, bins=50)
    plt.show()


def import_data(path):
    columns = ['Center', 'Left', 'Right', 'Steering', 'Throttle', 'Brake', 'Speed']
    df = pd.read_csv(os.path.join(path, 'driving_log.csv'), names=columns)
    df['Center'] = df['Center'].apply(center_left_right)
    df['Left'] = df['Left'].apply(center_left_right)
    df['Right'] = df['Right'].apply(center_left_right)
    return df


def balance_data(df, display):
    bins_num = 50
    samples_bin = 2000
    hist, bins = np.histogram(df['Steering'], bins_num)
    remove_index_list = []
    if display:
        histograma(df['Steering'])

    for j in range(bins_num):
        bins_data_index = []
        for i in range(len(df['Steering'])):
            if bins[j] <= df['Steering'][i] <= bins[j + 1]:
                bins_data_index.append(i)
        bins_data_index = shuffle(bins_data_index)
        bins_data_index = bins_data_index[samples_bin:]
        remove_index_list.extend(bins_data_index)
    print('Images Removed: ', len(remove_index_list))
    df.drop(df.index[remove_index_list], inplace=True)
    print('Images Remaining', len(df))

    if display:
        histograma(df['Steering'])

    return df


def load_data(path, df):
    img_path = []
    steering_list = []

    for i in range(df.shape[0]):
        data_indexed = df.iloc[i]
        img_path.append(os.path.join(path, 'IMG', data_indexed['Center']))
        steering_list.append(float(data_indexed['Steering']))
    img_path = np.array(img_path)
    steering_list = np.array(steering_list)
    return img_path, steering_list


def augment_images(img_path, steering):
    img = mpimg.imread(img_path)

    # PAN IMG
    if np.random.rand() < 0.5:
        pan = iaa.Affine(translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)})
        img = pan.augment_image(img)

    # ZOOM IMG
    if np.random.rand() < 0.5:
        zoom = iaa.Affine(scale=(1, 1.2))
        img = zoom.augment_image(img)

    # CHANGE BRIGHTNESS IMG
    if np.random.rand() < 0.5:
        brightness = iaa.Multiply((0.4, 1.2))
        img = brightness.augment_image(img)

    # FLIP IMG
    if np.random.rand() < 0.5:
        img = cv2.flip(img, 1)
        steering = -steering

    return img, steering


def preprocess_img(img):
    img = img[60: 135, :]

    # Mudando o color space para YUV (recomendado pela nvidia)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255

    return img


"""
                            Generators:
Generators são iterables os quais só podem ser percorridos UMA UÚNICA VEZ
Eles não armazenam valores, apenas calculam eles e depois esquecem:
Diferente dos iterables usam () ao invés de []

---Exemplo---
    >>> mygenerator = (x*x for x in range(3))
    >>> for i in mygenerator:
    ...    print(i)
    0
    1
    4
    
                                YEILD:
Yeild é como se fosse um return para Generators, ela não retorna valores, mas sim o próprio Generator
Para entender o Yeild é necessário saber quando ao chamar a função esta não roda o que foi escrito nela
É possível percorrer o Generator e obter seus valores usando for
Ao fazer o loop, a primeira iteração vai rodar o que tiver escrito no corpo da função até atingir o yeild,
retornando assim o valor da iteração, as iterações subsequentes vão retornar seus respectivos valores.
Isso irá continuar até que o código da função seja percorrido e não atingir o yeild, ou seja, ele roda até
o Generator ser considerado vazio
---Exemplo---
    >>> def create_generator():
    ...    mylist = range(3)
    ...    for i in mylist:
    ...        yield i*i
    ...
    >>> mygenerator = create_generator() # create a generator
    >>> print(mygenerator) # mygenerator is an object!
    <generator object create_generator at 0xb7555c34>
    >>> for i in mygenerator:
    ...     print(i)
    0
    1
    4
"""


def batch_generator(img_path, steering_list, batch_size, train_flag):
    while True:
        img_batch_list = []
        steering_batch_list = []

        for i in range(batch_size):
            index = random.randint(0, len(img_path) - 1)
            if train_flag:
                img, steering = augment_images(img_path[index], steering_list[index])
            else:
                img = mpimg.imread(img_path[index])
                steering = steering_list[index]
            img = preprocess_img(img)
            img_batch_list.append(img)
            steering_batch_list.append(steering)
        yield np.asarray(img_batch_list), np.asarray(steering_batch_list)


def criar_modelo():
    model = Sequential()

    model.add(Convolution2D(24, (5, 5), (2, 2), input_shape=(66, 200, 3), activation='elu'))
    model.add(Convolution2D(36, (5, 5), (2, 2), activation='elu'))
    model.add(Convolution2D(48, (5, 5), (2, 2), activation='elu'))
    model.add(Convolution2D(64, (3, 3), activation='elu'))
    model.add(Convolution2D(64, (3, 3), activation='elu'))

    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1, activation='elu'))

    model.compile(Adam(learning_rate=0.0001), loss='mse')
    return model


if __name__ == '__main__':
    path = 'myData'
    data = import_data(path)
    print(data.head())
    # data = balance_data(data, display=True)
    # img_load, steering_load = load_data(path, data)
    # print('IMG: ', img_load)
    # print('STEERING: ', steering_load)

    img_teste = cv2.imread('img_teste.jpg')
    img_teste = preprocess_img(img_teste)
    # # img_teste, steering_teste = augment_images('img_teste.jpg', 0.0)
    plt.imshow(img_teste)
    plt.show()
