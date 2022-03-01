from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import cv2


def pre_layers(X):                  # 224*224*3
    X = ZeroPadding2D((3, 3))(X)    # 230*230*3
    X = Conv2D(64, (7, 7), strides=(2, 2), name='pre_conv2', use_bias=False)(X)  # 112*112*64
    X = BatchNormalization(axis=3, name='pre-bn')(X)
    X = Activation('relu')(X)
    # X = MaxPool2D((3, 3), strides=(2, 2), padding='same')(X)                # 56*56*64
    X = Conv2D(64, (3, 3), strides=(2, 2), padding='same', name='replaced')(X)
    assert tuple(X.shape) == (None, 56, 56, 64)
    return X


def reduce_block(inp, out_n, name):
    X = Conv2D(out_n//4, (1, 1), strides=(2, 2), name=name + 'reduce_c2', use_bias=False)(inp)
    X = BatchNormalization(axis=3, name=name + 'reduce_bn')(X)
    X = LeakyReLU()(X)

    X = Conv2D(out_n//4, (3, 3), strides=(1, 1), padding='same', name=name + 'C2_1', use_bias=False)(X)
    X = BatchNormalization(axis=3, name=name+'bn_1')(X)
    X = LeakyReLU()(X)

    X = Conv2D(out_n, (1, 1), strides=(1, 1), name=name + 'C2_2', use_bias=False)(X)
    X = BatchNormalization(axis=3, name=name+'bn_2')(X)
    # X = Activation('relu')(X)

    p_inp = Conv2D(out_n, (1, 1), strides=(2, 2), name=name + 'res_conv', use_bias=False)(inp)
    p_inp = BatchNormalization(axis=3, name=name + 'res_bn')(p_inp)
    ret = Add()([X, p_inp])
    ret = LeakyReLU()(ret)
    return ret


def block(inp, out_n, name):
    X = Conv2D(out_n//4, (1, 1), strides=(1, 1), name=name + 'C2_1', use_bias=False)(inp)
    X = BatchNormalization(name=name + 'bn_1')(X)
    X = LeakyReLU()(X)

    X = Conv2D(out_n//4, (3, 3), strides=(1, 1), padding='same', name=name + 'C2_2', use_bias=False)(X)
    X = BatchNormalization(axis=3, name=name+'bn_2')(X)
    X = LeakyReLU()(X)

    X = Conv2D(out_n, (1, 1), strides=(1, 1), name=name + 'C2_3', use_bias=False)(X)
    X = BatchNormalization(axis=3, name=name+'bn_3')(X)
    # X = Activation('relu')(X)

    if inp.shape[-1] != out_n:
        inp = Conv2D(out_n, (1, 1), strides=(1, 1), name=name + 'ExC2', use_bias=False)(inp)
        inp = BatchNormalization(axis=3, name=name + 'bn_side')(inp)

    ret = Add()([X, inp])
    ret = LeakyReLU()(ret)
    return ret


def final_layers(X, num_classes):
    # 56*56 -> 28*28 1 -> 14*14 2 -> 7*7->3 avgpool(7*7) -> 1*1
    avg = AvgPool2D((7, 7), strides=(1, 1), name='final_avgpool')(X)
    ffc = Flatten()(avg)
    if num_classes != 1:
        result = Dense(num_classes, activation='softmax', name='final_fc')(ffc)
    else:
        result = Dense(num_classes, activation='sigmoid', name='final_fc')(ffc)
    return result


def resnet_50_poolless():
    X0 = Input((224, 224, 3))
    X = pre_layers(X0)       # 56*56*64

    X = block(X, 256, 'b1_1')
    X = block(X, 256, 'b1_2')
    X = block(X, 256, 'b1_3')   # 28*28*256

    X = reduce_block(X, 512, 'r2_1')
    for i in range(3):
        X = block(X, 512, 'b2_' + str(i+2))

    X = reduce_block(X, 1024, 'r3_1')
    for i in range(5):
        X = block(X, 1024, 'b3_' + str(i+2))

    X = reduce_block(X, 2048, 'r4_1')
    for i in range(2):
        X = block(X, 2048, 'b4_' + str(i+2))
    last = final_layers(X, 1)
    return Model(inputs=X0, outputs=last, name='ResNet50 Poolless')


if __name__ == '__main__':
    md = resnet_50_poolless()
    md.summary()

