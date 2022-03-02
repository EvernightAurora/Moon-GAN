from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import cv2
import utils
from matplotlib import pyplot as plt


def reuse_block(input, in_layers):
    input = Conv2D(in_layers, (5, 5), strides=(1, 1), padding='same', use_bias=False)(input)
    input = BatchNormalization()(input)
    input = LeakyReLU()(input)
    input = Conv2DTranspose(in_layers // 2, (5, 5), strides=(2, 2), padding='same', use_bias=False)(input)
    input = BatchNormalization()(input)
    input = LeakyReLU()(input)
    input = Conv2D(in_layers // 2, (5, 5), strides=(1, 1), padding='same', use_bias=False)(input)
    input = BatchNormalization()(input)
    input = LeakyReLU()(input)
    return input


def make_body(input):   # input: 100
    first = Dense(7 * 7 * 256)(input)
    first = BatchNormalization()(first)
    first = LeakyReLU()(first)
    first = Reshape((7, 7, 256))(first)
    assert tuple(first.shape) == (None, 7, 7, 256)

    second = reuse_block(first, 256)
    assert tuple(second.shape) == (None, 14, 14, 128)

    third = reuse_block(second, 128)
    assert tuple(third.shape) == (None, 28, 28, 64)

    fourth = reuse_block(third, 64)
    assert tuple(fourth.shape) == (None, 56, 56, 32)

    fifth = reuse_block(fourth, 32)
    assert tuple(fifth.shape) == (None, 112, 112, 16)

    pre_sixth = Conv2D(8, (5, 5), strides=(1, 1), padding='same', use_bias=False)(fifth)
    pre_sixth = BatchNormalization()(pre_sixth)
    pre_sixth = LeakyReLU()(pre_sixth)
    assert tuple(pre_sixth.shape) == (None, 112, 112, 8)

    sixth = reuse_block(pre_sixth, 8)
    assert tuple(sixth.shape) == (None, 224, 224, 4)

    final = Conv2D(3, (5, 5), strides=(1, 1), padding='same', activation='tanh')(sixth)
    return final


def generater_jnet():
    x = Input((100, ))
    y = make_body(x)
    return Model(inputs=x, outputs=y, name='JNet')


if __name__ == '__main__':
    mod = generater_jnet()
    mod.summary()
    rd = tf.random.normal([16, 100])
    gen = mod(rd, training=False)
    gen = gen.numpy()
    utils.plot_16_image(gen)
    plt.show()
    print('hi')


