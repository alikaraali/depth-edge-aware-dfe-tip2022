import tensorflow as tf
import tensorflow.keras.layers as L
import numpy as np

from domain_transform import RF, RFedge


def propagate_domaintransform(img, mask, mask_depth, edge_penalty, bmap):
    '''
    Python implementation of defocus blur map propagation from
    A. Karaali, CR. Jung, "Edge-Based Defocus Blur Estimation with
    Adaptive Scale Selection",
    IEEE Transactions on Image Processing (TIP 2018), 2018
    Cite accordingly.

    Written by Ali Karaali
    alixkaraali(classic_at_sign)gmail.com

    :param img: RGB blurry image
    :param mask: Pattern edge map
    :param mask_depth: Depth edge map
    :param edge_penalty: Edge penalty (See Eq(2))
    :param bmap: Estimated sparse blur map
    :return:
    '''

    h, w = mask.shape
    sigma_s = min(h, w) / 8.0

    sigma_r = 3.75
    niter = 5

    Iref = RF(img / 255.0, 7, 0.5, niter)

    F_ic = RFedge(bmap,  mask_depth, edge_penalty,
                  sigma_s, sigma_r, niter, Iref)
    mask_ic = RFedge(mask, mask_depth, edge_penalty,
                     sigma_s, sigma_r, niter, Iref)
    bmapDomainTr = np.divide(F_ic, mask_ic).reshape(h, w)

    return bmapDomainTr


def layer_type1(x_inp, filters, name, kernel_size=(3, 3)):
    x = L.Conv2D(filters, kernel_size, use_bias=False,
                 padding = 'valid', name=name)(x_inp)
    x = L.ReLU()(x)

    return x


def get_weight(values, name):
    return tf.Variable(initial_value=values,
                       name=name , trainable=False)


def layer_type2(x_inp, weights, name):
    filters = get_weight(weights, name)
    x_out = tf.nn.conv2d(x_inp, filters, strides=[1, 1, 1, 1],
                         padding='VALID', name=name)

    return tf.nn.relu(x_out)


def make_BNet(xinp1, xinp2, xinp3, all_weights):
    '''
    The proposed Blur Estimation Network (BNet)
    See Chapter III-A for details

    Written by Ali Karaali
    alixkaraali(classic_at_sign)gmail.com

    :param xinp1: Patch size 41x41
    :param xinp2: Patch size 27x27
    :param xinp3: Patch size 15x15
    :param all_weights: Trained weights for BNet
    :return:
    '''

    x01 = layer_type2(xinp1, all_weights[0], 'W11')
    xp1 = tf.nn.max_pool2d(x01 , ksize=[1,3,3,1],
                           strides = [1,3,3,1], padding='SAME')
    x02 = layer_type2(xp1, all_weights[1], 'W12')
    x03 = layer_type2(x02, all_weights[2], 'W12')
    x04 = layer_type2(x03, all_weights[3], 'W12')

    x11 = layer_type2(xinp2, all_weights[4], 'W21')
    x12 = layer_type2(x11, all_weights[5], 'W22')
    x13 = layer_type2(x12, all_weights[6], 'W23')
    xp2 = tf.nn.max_pool2d(x13 , ksize=[1,3,3,1],
                           strides = [1,3,3,1], padding='SAME')
    x14 = layer_type2(xp2, all_weights[7], 'W24')

    x21 = layer_type2(xinp3, all_weights[8], 'W31')
    x22 = layer_type2(x21, all_weights[9], 'W32')
    x23 = layer_type2(x22, all_weights[10], 'W33')
    x24 = layer_type2(x23, all_weights[11], 'W34')

    C1 = tf.concat([x04, x14, x24], 3)

    xA = layer_type2(C1, all_weights[12], 'WA')
    xB = layer_type2(xA, all_weights[13], 'WB')
    xC = layer_type2(xB, all_weights[14], 'WC')
    xD = layer_type2(xC, all_weights[15], 'WD')

    flatten = tf.reshape(xD, shape=(tf.shape(xD)[0], -1))

    d1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(flatten, all_weights[16]),
                                   all_weights[17].ravel()))
    d2 = tf.nn.relu(tf.nn.bias_add(tf.matmul(d1, all_weights[18]),
                                   all_weights[19].ravel()))
    out = tf.nn.softmax(tf.nn.bias_add(tf.matmul(d2, all_weights[20]),
                                       all_weights[21].ravel()))

    return out


def make_ENet(xinp1, xinp2, xinp3, all_weights):
    '''
    The proposed Edge Classification Network (ENet)
    See Chapter III-B for details

    Written by Ali Karaali
    alixkaraali(classic_at_sign)gmail.com

    :param xinp1: Patch size 41x41
    :param xinp2: Patch size 27x27
    :param xinp3: Patch size 15x15
    :param all_weights: Trained weights for ENet
    :return:
    '''

    x01 = layer_type2(xinp1, all_weights[0], 'W11')
    xp1 = tf.nn.max_pool2d(x01 , ksize=[1,3,3,1],
                           strides = [1,3,3,1], padding='SAME')
    x02 = layer_type2(xp1, all_weights[1], 'W12')
    x03 = layer_type2(x02, all_weights[2], 'W12')
    x04 = layer_type2(x03, all_weights[3], 'W12')

    x01B = layer_type2(xinp1, all_weights[4], 'W11B')
    xp1B = tf.nn.max_pool2d(x01B , ksize=[1,3,3,1],
                            strides = [1,3,3,1], padding='SAME')
    x02B = layer_type2(xp1B, all_weights[5], 'W12B')
    x03B = layer_type2(x02B, all_weights[6], 'W12B')
    x04B = layer_type2(x03B, all_weights[7], 'W12B')

    x11 = layer_type2(xinp2, all_weights[8], 'W21')
    x12 = layer_type2(x11, all_weights[9], 'W22')
    x13 = layer_type2(x12, all_weights[10], 'W23')
    xp2 = tf.nn.max_pool2d(x13 , ksize=[1,3,3,1],
                           strides = [1,3,3,1], padding='SAME')
    x14 = layer_type2(xp2, all_weights[11], 'W24')

    x21 = layer_type2(xinp3, all_weights[12], 'W31')
    x22 = layer_type2(x21, all_weights[13], 'W32')
    x23 = layer_type2(x22, all_weights[14], 'W33')
    x24 = layer_type2(x23, all_weights[15], 'W34')

    C1 = tf.concat([x04B, x04, x14, x24], 3)

    xA = layer_type2(C1, all_weights[16], 'WA')
    xB = layer_type2(xA, all_weights[17], 'WB')
    xC = layer_type2(xB, all_weights[18], 'WC')
    xD = layer_type2(xC, all_weights[19], 'WD')

    flatten = tf.reshape(xD, shape=(tf.shape(xD)[0], -1))

    d1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(flatten, all_weights[20]),
                                   all_weights[21].ravel()) )
    d2 = tf.nn.relu(tf.nn.bias_add(tf.matmul(d1, all_weights[22]),
                                   all_weights[23].ravel()))
    out = tf.nn.softmax(tf.nn.bias_add(tf.matmul(d2, all_weights[24]),
                                       all_weights[25].ravel()))

    return out

