import matplotlib.pyplot as plt
import tensorflow as tf
from scipy import misc
import os
import numpy as np
from PIL import Image
import os

# img=Image.open('1.jpg')
# array=np.array(img)  #uint8
# print(array.dtype,array.shape)
def mkdir(file):
    if not os.path.exists(file):
        os.makedirs(file)

def imwrite(image, path):
    """ save an [-1.0, 1.0] image """

    if image.ndim == 3 and image.shape[2] == 1:  # for gray image
        image = np.array(image, copy=True)
        image.shape = image.shape[0:2]

    imgarray=((image+1.0)*127.5).astype(np.uint8)
    img=Image.fromarray(imgarray)
    img.save(path)



def immerge(images, row, col):
    """
    merge images into an image with (row * h) * (col * w)

    `images` is in shape of N * H * W(* C=1 or 3)
    """

    h, w = images.shape[1], images.shape[2]
    if images.ndim == 4:
        img = np.zeros((h * row, w * col, images.shape[3]))
    elif images.ndim == 3:
        img = np.zeros((h * row, w * col))
    for idx, image in enumerate(images):
        i = idx % col
        j = idx // col
        img[j * h:j * h + h, i * w:i * w + w, ...] = image

    return img

# %matplotlib inline

def vis_img(batch_size, samples):
    fig, axes = plt.subplots(figsize=(7, 7), nrows=8, ncols=8, sharey=True, sharex=True)

    for ax, img in zip(axes.flatten(), samples[batch_size]):
        # print(img.shape)

        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        im = ax.imshow(img.reshape((32, 32, 3)), cmap='Greys_r')
    plt.show()
    return fig, axes


def read_img(path):
    #img = misc.imresize(misc.imread(path), size=[32, 32])
    img = misc.imresize(misc.imread(path), size=[96, 96])##########yang#########
    return img


def get_batch(path, batch_size):
    img_list = [os.path.join(path, i) for i in os.listdir(path)]

    n_batchs = len(img_list) // batch_size
    img_list = img_list[:n_batchs * batch_size]

    for ii in range(n_batchs):
        tmp_img_list = img_list[ii * batch_size:(ii + 1) * batch_size]
        #img_batch = np.zeros(shape=[batch_size, 32, 32, 3])
        img_batch = np.zeros(shape=[batch_size, 96, 96, 3])       ##########yang#########
        for jj, img in enumerate(tmp_img_list):
            img_batch[jj] = read_img(img)
        yield img_batch


def generator(inputs, stddev=0.02, alpha=0.2, name='generator', reuse=False):
    with tf.variable_scope(name, reuse=reuse) as scope:
        fc1 = tf.layers.dense(gen_input, 64 * 8 * 6 * 6, name='fc1')
        re1 = tf.reshape(fc1, (-1, 6, 6, 512), name='reshape')
        bn1 = tf.layers.batch_normalization(re1, name='bn1')
        # ac1 = tf.maximum(alpha * bn1, bn1,name='ac1')
        ac1 = tf.nn.relu(bn1, name='ac1')


        de_conv1 = tf.layers.conv2d_transpose(ac1, 256, kernel_size=[5, 5], padding='same', strides=2,
                                              kernel_initializer=tf.random_normal_initializer(stddev=stddev),
                                              name='decov1')
        bn2 = tf.layers.batch_normalization(de_conv1, name='bn2')
         # ac2 = tf.maximum(alpha * bn2, bn2,name='ac2')
        ac2 = tf.nn.relu(bn2, name='ac2')


        de_conv2 = tf.layers.conv2d_transpose(ac2, 128, kernel_size=[5, 5], padding='same',
                                             kernel_initializer=tf.random_normal_initializer(stddev=stddev), strides=2,
                                             name='decov2')
        bn3 = tf.layers.batch_normalization(de_conv2, name='bn3')
         # ac3 = tf.maximum(alpha * bn3, bn3,name='ac3')
        ac3 = tf.nn.relu(bn3, name='ac3')


        de_conv3 = tf.layers.conv2d_transpose(ac3, 64, kernel_size=[5, 5], padding='same',
                                             kernel_initializer=tf.random_normal_initializer(stddev=stddev), strides=2,
                                             name='decov3')
        bn4 = tf.layers.batch_normalization(de_conv3, name='bn4')
         # ac4 = tf.maximum(alpha * bn4, bn4,name='ac4')
        ac4 = tf.nn.relu(bn4, name='ac4')


        logits = tf.layers.conv2d_transpose(ac4, 3, kernel_size=[5, 5], padding='same',
                                            kernel_initializer=tf.random_normal_initializer(stddev=stddev), strides=2,
                                            name='logits')


        output = tf.tanh(logits)


        return output


def discriminator(inputs, stddev=0.02, alpha=0.2, batch_size=64, name='discriminator', reuse=False):
    with tf.variable_scope(name, reuse=reuse) as scope:
        conv1 = tf.layers.conv2d(inputs, 64, (5, 5), (2, 2), padding='same',
                                 kernel_initializer=tf.random_normal_initializer(stddev=stddev), name='conv1')

        ac1 = tf.maximum(alpha * conv1, conv1, name='ac1')

        conv2 = tf.layers.conv2d(ac1, 128, (5, 5), (2, 2), padding='same',
                                 kernel_initializer=tf.random_normal_initializer(stddev=stddev), name='conv2')
        bn2 = tf.layers.batch_normalization(conv2, name='bn2')
        ac2 = tf.maximum(alpha * bn2, bn2, name='ac2')

        conv3 = tf.layers.conv2d(ac2, 256, (5, 5), (2, 2), padding='same',
                                 kernel_initializer=tf.random_normal_initializer(stddev=stddev), name='conv3')
        bn3 = tf.layers.batch_normalization(conv3, name='bn3')
        ac3 = tf.maximum(alpha * bn3, bn3, name='ac3')

        conv4 = tf.layers.conv2d(ac3, 512, (5, 5), (2, 2), padding='same',
                                 kernel_initializer=tf.random_normal_initializer(stddev=stddev), name='conv4')
        bn4 = tf.layers.batch_normalization(conv4, name='bn4')
        ac4 = tf.maximum(alpha * bn4, bn4, name='ac4')

        flat = tf.reshape(ac4, shape=[batch_size, 6 * 6 * 512], name='reshape')

        fc2 = tf.layers.dense(flat, 1, kernel_initializer=tf.random_normal_initializer(stddev=stddev), name='fc2')
        return fc2




lr = 0.0002
epochs = 100
batch_size = 64

alpha = 0.2
with tf.name_scope('gen_input') as scope:
    gen_input = tf.placeholder(dtype=tf.float32, shape=[None, 100], name='gen_input')

with tf.name_scope('dis_input') as scope:
    real_input = tf.placeholder(dtype=tf.float32, shape=[None, 96, 96, 3], name='dis_input')

gen_out = generator(gen_input, stddev=0.02, alpha=alpha, name='generator', reuse=False)

real_logits = discriminator(real_input, alpha=alpha, batch_size=batch_size)
fake_logits = discriminator(gen_out, alpha=alpha, reuse=True)

train_var = tf.trainable_variables()
var_list_gen = [var for var in train_var if var.name.startswith('generator')]
var_list_dis = [var for var in train_var if var.name.startswith('discriminator')]

with tf.name_scope('metrics') as scope:
    loss_g = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake_logits) * 0.9, logits=fake_logits))
    loss_d_f = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake_logits), logits=fake_logits))
    loss_d_r = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real_logits) * 0.9, logits=real_logits))
    loss_d = loss_d_f + loss_d_r
    gen_optimizer = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(loss_g, var_list=var_list_gen)
    dis_optimizer = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(loss_d, var_list=var_list_dis)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    writer = tf.summary.FileWriter('./graph/DCGAN', sess.graph)
    saver = tf.train.Saver()

    for epoch in range(epochs):
        total_g_loss = 0
        total_d_loss = 0
        KK = 0
        for batch in get_batch('./faces/', batch_size):

            x_real = batch
            x_real = x_real / 127.5 - 1
            x_fake = np.random.uniform(-1, 1, size=[batch_size, 100])

            KK += 1
            print(KK)
            _, tmp_loss_d = sess.run([dis_optimizer, loss_d], feed_dict={gen_input: x_fake, real_input: x_real})

            total_d_loss += tmp_loss_d

            _, tmp_loss_g = sess.run([gen_optimizer, loss_g], feed_dict={gen_input: x_fake})
            _, tmp_loss_g = sess.run([gen_optimizer, loss_g], feed_dict={gen_input: x_fake})
            total_g_loss += tmp_loss_g

        #if epoch % 10 == 0:
        x_fake = np.random.uniform(-1, 1, [64, 100])

        samples = sess.run(gen_out, feed_dict={gen_input: x_fake})
        #samples = (((samples - samples.min()) * 255) / (samples.max() - samples.min())).astype(np.uint8)

        mkdir('out_cartoon')
        imwrite(immerge(samples, 10, 10), 'out_cartoon/' + str(epoch) + '.jpg')

        print('epoch {},loss_g={}'.format(epoch, total_g_loss / 2 / KK))
        print('epoch {},loss_d={}'.format(epoch, total_d_loss / KK))

    writer.close()
    saver.save(sess, "./checkpoints/DCGAN")

