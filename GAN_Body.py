import Generater_NN
import Poolless_Resnet50
import DataLoader
import utils
import tensorflow as tf
import os
from matplotlib import pyplot as plt
from tensorflow.keras.layers import *
from tensorflow.keras.losses import binary_crossentropy
import random
from tqdm import tqdm
import numpy as np


IMAGE_DIR = 'generate/'
CHECKPOINT_DIR = 'checkpoint/MGAN-1'


def generate_tick_images(Gan, epochs):
    randoms = tf.random.normal([16, 100])
    result = Gan(randoms, training=False)
    utils.plot_16_image(result)
    os.makedirs(IMAGE_DIR) if not os.path.exists(IMAGE_DIR) else None
    plt.savefig(os.path.join(IMAGE_DIR, '%04d.png' % epochs))


def generate_loss(fake):
    return binary_crossentropy(tf.ones_like(fake), fake)


def discriminate_loss(real, fake):
    return binary_crossentropy(tf.ones_like(real), real) + binary_crossentropy(tf.zeros_like(fake), fake)


def train_step(gen, gen_opt, disc, disc_opt, real_imgs, batch_size=16):
    noise = tf.random.normal((batch_size, 100))
    with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
        fake_imgs = gen(noise)
        fake = disc(fake_imgs)
        real = disc(real_imgs)
        g_loss = generate_loss(fake)
        d_loss = discriminate_loss(real, fake)
    g_gradient = g_tape.gradient(g_loss, gen.trainable_variables)
    d_gradient = d_tape.gradient(d_loss, disc.trainable_variables)
    gen_opt.apply_gradients(zip(g_gradient, gen.trainable_variables))
    disc_opt.apply_gradients(zip(d_gradient, disc.trainable_variables))


def moon_gan(gen, disc, epochs=50, batch_size=16, use_checkpoint=False):
    g_optimizer = tf.keras.optimizers.Adam(1e-4)
    d_optimizer = tf.keras.optimizers.Adam(1e-4)
    prefix = CHECKPOINT_DIR + 'mg'
    checkpoint = tf.train.Checkpoint(g_optimizer=g_optimizer, d_optimizer=d_optimizer, gen=gen, disc=disc)
    img_raw = DataLoader.load_datasets()
    selects = list(range(img_raw.shape[0]))
    if use_checkpoint:
        checkpoint.restore(tf.train.latest_checkpoint(CHECKPOINT_DIR))
    for epoch in range(epochs):
        random.shuffle(selects)
        cnt = len(selects) // batch_size
        for i in tqdm(range(cnt)):
            sels = selects[i: i+batch_size]
            used = []
            for i in sels:
                used.append(img_raw[i])
            used = np.array(used)
            train_step(gen, g_optimizer, disc, d_optimizer, used, batch_size=batch_size)
        print('epoch %d ended', epoch)
        checkpoint.save(file_prefix=prefix)
        generate_tick_images(gen, epochs)


if __name__ == '__main__':
    disc = Poolless_Resnet50.resnet_50_poolless()
    gen = Generater_NN.generater_jnet()
    moon_gan(gen, disc)
