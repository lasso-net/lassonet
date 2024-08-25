import pickle
from collections import defaultdict
from os.path import join
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# The code to load some of these datasets is reproduced from
# https://github.com/mfbalin/Concrete-Autoencoders/blob/master/experiments/generate_comparison_figures.py


def load_mice(one_hot=False):
    filling_value = -100000
    X = np.genfromtxt(
        "/home/lemisma/datasets/MICE/Data_Cortex_Nuclear.csv",
        delimiter=",",
        skip_header=1,
        usecols=range(1, 78),
        filling_values=filling_value,
        encoding="UTF-8",
    )
    classes = np.genfromtxt(
        "/home/lemisma/datasets/MICE/Data_Cortex_Nuclear.csv",
        delimiter=",",
        skip_header=1,
        usecols=range(78, 81),
        dtype=None,
        encoding="UTF-8",
    )

    for i, row in enumerate(X):
        for j, val in enumerate(row):
            if val == filling_value:
                X[i, j] = np.mean(
                    [
                        X[k, j]
                        for k in range(classes.shape[0])
                        if np.all(classes[i] == classes[k])
                    ]
                )

    DY = np.zeros((classes.shape[0]), dtype=np.uint8)
    for i, row in enumerate(classes):
        for j, (val, label) in enumerate(zip(row, ["Control", "Memantine", "C/S"])):
            DY[i] += (2**j) * (val == label)

    Y = np.zeros((DY.shape[0], np.unique(DY).shape[0]))
    for idx, val in enumerate(DY):
        Y[idx, val] = 1

    X = MinMaxScaler(feature_range=(0, 1)).fit_transform(X)

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    DY = DY[indices]
    classes = classes[indices]

    if not one_hot:
        Y = DY

    X = X.astype(np.float32)
    Y = Y.astype(np.float32)

    print("X shape: {}, Y shape: {}".format(X.shape, Y.shape))

    return (X[: X.shape[0] * 4 // 5], Y[: X.shape[0] * 4 // 5]), (
        X[X.shape[0] * 4 // 5 :],
        Y[X.shape[0] * 4 // 5 :],
    )


def load_isolet():
    x_train = np.genfromtxt(
        "/home/lemisma/datasets/isolet/isolet1+2+3+4.data",
        delimiter=",",
        usecols=range(0, 617),
        encoding="UTF-8",
    )
    y_train = np.genfromtxt(
        "/home/lemisma/datasets/isolet/isolet1+2+3+4.data",
        delimiter=",",
        usecols=[617],
        encoding="UTF-8",
    )
    x_test = np.genfromtxt(
        "/home/lemisma/datasets/isolet/isolet5.data",
        delimiter=",",
        usecols=range(0, 617),
        encoding="UTF-8",
    )
    y_test = np.genfromtxt(
        "/home/lemisma/datasets/isolet/isolet5.data",
        delimiter=",",
        usecols=[617],
        encoding="UTF-8",
    )

    X = MinMaxScaler(feature_range=(0, 1)).fit_transform(
        np.concatenate((x_train, x_test))
    )
    x_train = X[: len(y_train)]
    x_test = X[len(y_train) :]

    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)

    return (x_train, y_train - 1), (x_test, y_test - 1)


def load_activity():
    x_train = np.loadtxt(
        os.path.join("/home/lemisma/datasets/dataset_uci", "final_X_train.txt"),
        delimiter=",",
        encoding="UTF-8",
    )
    x_test = np.loadtxt(
        os.path.join("/home/lemisma/datasets/dataset_uci", "final_X_test.txt"),
        delimiter=",",
        encoding="UTF-8",
    )
    y_train = np.loadtxt(
        os.path.join("/home/lemisma/datasets/dataset_uci", "final_y_train.txt"),
        delimiter=",",
        encoding="UTF-8",
    )
    y_test = np.loadtxt(
        os.path.join("/home/lemisma/datasets/dataset_uci", "final_y_test.txt"),
        delimiter=",",
        encoding="UTF-8",
    )

    X = MinMaxScaler(feature_range=(0, 1)).fit_transform(
        np.concatenate((x_train, x_test))
    )
    x_train = X[: len(y_train)]
    x_test = X[len(y_train) :]

    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)
    return (x_train, y_train), (x_test, y_test)


import numpy as np


def load_epileptic():
    filling_value = -100000

    X = np.genfromtxt(
        "/home/lemisma/datasets/data.csv",
        delimiter=",",
        skip_header=1,
        usecols=range(1, 179),
        filling_values=filling_value,
        encoding="UTF-8",
    )
    Y = np.genfromtxt(
        "/homelemisma/datasets/data.csv",
        delimiter=",",
        skip_header=1,
        usecols=range(179, 180),
        encoding="UTF-8",
    )

    X = MinMaxScaler(feature_range=(0, 1)).fit_transform(X)

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]

    print(X.shape, Y.shape)

    return (X[:8000], Y[:8000]), (X[8000:], Y[8000:])


import os

from PIL import Image


def load_coil():
    samples = []
    for i in range(1, 21):
        for image_index in range(72):
            obj_img = Image.open(
                os.path.join(
                    "/home/lemisma/datasets/coil-20-proc",
                    "obj%d__%d.png" % (i, image_index),
                )
            )
            rescaled = obj_img.resize((20, 20))
            pixels_values = [float(x) for x in list(rescaled.getdata())]
            sample = np.array(pixels_values + [i])
            samples.append(sample)
    samples = np.array(samples)
    np.random.shuffle(samples)
    data = samples[:, :-1]
    targets = (samples[:, -1] + 0.5).astype(np.int64)
    data = (data - data.min()) / (data.max() - data.min())

    l = data.shape[0] * 4 // 5
    train = (data[:l], targets[:l] - 1)
    test = (data[l:], targets[l:] - 1)
    print(train[0].shape, train[1].shape)
    print(test[0].shape, test[1].shape)
    return train, test


import tensorflow as tf
from sklearn.model_selection import train_test_split


def load_data(fashion=False, digit=None, normalize=False):
    if fashion:
        (x_train, y_train), (x_test, y_test) = (
            tf.keras.datasets.fashion_mnist.load_data()
        )
    else:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    if digit is not None and 0 <= digit and digit <= 9:
        train = test = {y: [] for y in range(10)}
        for x, y in zip(x_train, y_train):
            train[y].append(x)
        for x, y in zip(x_test, y_test):
            test[y].append(x)

        for y in range(10):

            train[y] = np.asarray(train[y])
            test[y] = np.asarray(test[y])

        x_train = train[digit]
        x_test = test[digit]

    x_train = x_train.reshape((-1, x_train.shape[1] * x_train.shape[2])).astype(
        np.float32
    )
    x_test = x_test.reshape((-1, x_test.shape[1] * x_test.shape[2])).astype(np.float32)

    if normalize:
        X = np.concatenate((x_train, x_test))
        X = (X - X.min()) / (X.max() - X.min())
        x_train = X[: len(y_train)]
        x_test = X[len(y_train) :]

    #     print(x_train.shape, y_train.shape)
    #     print(x_test.shape, y_test.shape)
    return (x_train, y_train), (x_test, y_test)


def load_mnist():
    train, test = load_data(fashion=False, normalize=True)
    x_train, x_test, y_train, y_test = train_test_split(test[0], test[1], test_size=0.6)
    return (x_train, y_train), (x_test, y_test)


def load_fashion():
    train, test = load_data(fashion=True, normalize=True)
    x_train, x_test, y_train, y_test = train_test_split(test[0], test[1], test_size=0.6)
    return (x_train, y_train), (x_test, y_test)


def load_mnist_two_digits(digit1, digit2):
    train_digit_1, _ = load_data(digit=digit1)
    train_digit_2, _ = load_data(digit=digit2)

    X_train_1, X_test_1 = train_test_split(train_digit_1[0], test_size=0.6)
    X_train_2, X_test_2 = train_test_split(train_digit_2[0], test_size=0.6)

    X_train = np.concatenate((X_train_1, X_train_2))
    y_train = np.array([0] * X_train_1.shape[0] + [1] * X_train_2.shape[0])
    shuffled_idx = np.random.permutation(X_train.shape[0])
    np.take(X_train, shuffled_idx, axis=0, out=X_train)
    np.take(y_train, shuffled_idx, axis=0, out=y_train)

    X_test = np.concatenate((X_test_1, X_test_2))
    y_test = np.array([0] * X_test_1.shape[0] + [1] * X_test_2.shape[0])
    shuffled_idx = np.random.permutation(X_test.shape[0])
    np.take(X_test, shuffled_idx, axis=0, out=X_test)
    np.take(y_test, shuffled_idx, axis=0, out=y_test)

    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    return (X_train, y_train), (X_test, y_test)


import os

from sklearn.preprocessing import MinMaxScaler


def load_activity():
    x_train = np.loadtxt(
        os.path.join("/home/lemisma/datasets/dataset_uci", "final_X_train.txt"),
        delimiter=",",
        encoding="UTF-8",
    )
    x_test = np.loadtxt(
        os.path.join("/home/lemisma/datasets/dataset_uci", "final_X_test.txt"),
        delimiter=",",
        encoding="UTF-8",
    )
    y_train = (
        np.loadtxt(
            os.path.join("/home/lemisma/datasets/dataset_uci", "final_y_train.txt"),
            delimiter=",",
            encoding="UTF-8",
        )
        - 1
    )
    y_test = (
        np.loadtxt(
            os.path.join("/home/lemisma/datasets/dataset_uci", "final_y_test.txt"),
            delimiter=",",
            encoding="UTF-8",
        )
        - 1
    )

    X = MinMaxScaler(feature_range=(0, 1)).fit_transform(
        np.concatenate((x_train, x_test))
    )
    x_train = X[: len(y_train)]
    x_test = X[len(y_train) :]

    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)
    return (x_train, y_train), (x_test, y_test)
