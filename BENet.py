import tensorflow as tf
import argparse
import numpy as np
import cv2
import scipy.io as sio

from skimage import feature, color
from model import make_ENet, make_BNet, propagate_domaintransform


def get_args():
    parser = argparse.ArgumentParser(description='Depth Edge Aware Defocus Blur Estimation \n')

    parser.add_argument('-i', metavar='--image', required=True,
                        type=str, help='Defocused image \n')

    parser.add_argument('-e', metavar='--edge_map',
                        type=str, help='Edge map of the defocus image (optional) \n')

    args = parser.parse_args()
    image = args.i
    edgemap = args.e

    return {'image': image, 'edge_map' : edgemap}


if __name__ == '__main__':

    args = get_args()

    edge_penalty = 20
    img = cv2.imread(args['image'])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if args['edge_map']:
        mask = cv2.imread(args['edge_map'], cv2.IMREAD_GRAYSCALE) / 255.0
    else:
        img_hsv = color.rgb2hsv(img)
        mask = feature.canny(img_hsv[:, :, 0], np.sqrt(2), 0.05, 0.06) * 1.0

    #
    weights_blur = []
    for i in range(22):
        weights_blur.append(sio.loadmat('models_tf1x/blur/{:d}.mat'.format(i))['weights'])

    weights_edge = []
    for i in range(26):
        weights_edge.append(sio.loadmat('models_tf1x/edge/{:d}.mat'.format(i))['weights'])

    psize = 20
    wsize = 41
    img_padded = cv2.copyMakeBorder(img, psize, psize, psize, psize, cv2.BORDER_REFLECT)

    H, W = mask.shape

    y, x = np.where(mask == 1)

    edge_pix_count = len(y)
    patches = np.zeros((edge_pix_count, 41, 41, 3), dtype=np.uint8)

    for pix in range(edge_pix_count):
        patches[pix, :, :, :] = img_padded[y[pix]:y[pix] + wsize, x[pix]:x[pix] + wsize, :]

    predsblur = make_BNet(patches / 255.0,
                          patches[:, 8:35, 8:35, :] / 255.0,
                          patches[:, 14:29, 14:29, :] / 255.0, weights_blur)

    predsedge = make_ENet(patches / 255.0,
                          patches[:, 8:35, 8:35, :] / 255.0,
                          patches[:, 14:29, 14:29, :] / 255.0, weights_edge)

    pb = tf.argmax(predsblur, axis=1)
    pe = tf.argmax(predsedge, axis=1)

    sblur = np.arange(0.5, 6.25, 0.25)
    bmap = np.zeros(shape=(H, W), dtype=np.float32)
    emap = np.zeros(shape=(H, W), dtype=np.float32)
    bmap[y, x] = sblur[pb.numpy()]
    emap[y, x] = np.abs(pe.numpy() / pe.numpy().max() - 1)

    bmap = np.multiply(bmap, emap)
    emapdepth = np.zeros(shape=(H, W), dtype=np.float32)
    emapdepth[y, x] = pe.numpy() / pe.numpy().max()

    fblurmap = propagate_domaintransform(img, emap, emapdepth, edge_penalty, bmap)

    names1 = args['image'].split('.')

    cv2.imwrite(names[0] + '_bmap.png', np.uint8((fblurmap / 6.0)*255))

    edge_map = np.zeros((H,W,3), dtype=np.uint8)
    edge_map[:,:,0] = mask
    edge_map[:,:,2] = emapdepth
    cv2.imwrite(names[0] + '_edge.png', edge_map*255)

