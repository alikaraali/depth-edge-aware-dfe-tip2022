import numpy as np


def image_transpose(img):
    '''

    :param img: The image to be transposed
    :return: The transposed image
    '''
    
    dim_count = img.ndim
    
    if dim_count == 3:
    	h, w, num_channels = img.shape

    	T = np.zeros((w, h, num_channels), dtype=img.dtype)
    	for c in range(num_channels):
        	T[:, :, c] = img[:, :, c].T
        	
    else:
    	
    	T = img.T

    return T


def RFedge(img, bw, edgew, sigma_s, sigma_r, num_iteration=3,
           joint_image=None):
    '''
    Modified Python implementation of Domain Transform for Edge-Aware
    Image and Video Processing. Cite accordingly.
    (http://inf.ufrgs.br/~eslgastal/DomainTransform/)

    Written by Ali Karaali
    alixkaraali(classic_at_sign)gmail.com

    :param img: Input image to be filtered
    :param bw: Depth edges (from the proposed network)
    :param edgew: Edge weight see Eq(2) in the paper
    :param sigma_s: Spatial standard deviation
    :param sigma_r: Range standard deviation
    :param num_iteration: Number of iteration
    :param joint_image: Optional imge for joint filtering
    :return: Filtered image
    '''
    I = img

    if joint_image is None:
        J = I.copy()
    else:
        J = joint_image

    number_of_dim = J.ndim

    if number_of_dim == 3:
        h, w, num_joint_channels = J.shape
    else:
        h, w = J.shape
        num_joint_channels = 1
        J = J[:, :, np.newaxis]

    dIcdy = J[1:, :, :] - J[:h - 1, :, :]
    dIcdx = J[:, 1:, :] - J[:, :w - 1, :]

    dIdx = np.zeros((h, w))
    dIdy = np.zeros((h, w))

    for c in range(num_joint_channels):
        dIdx[:, 1:] += np.abs(dIcdx[:, :, c].reshape((h, w - 1)))
        dIdy[1:, :] += np.abs(dIcdy[:, :, c].reshape((h - 1, w)))

    dHdx = (1 + (sigma_s / sigma_r) * dIdx + bw * edgew)
    dVdy = (1 + (sigma_s / sigma_r) * dIdy + bw * edgew)

    dVdy = dVdy.T

    N = num_iteration

    F = I.copy()
    sigma_H = sigma_s

    for i in range(num_iteration):
        sigma_H_i = sigma_H * np.sqrt(3) * \
                    (2 ** (N - (i + 1))) / np.sqrt(4 ** N - 1)

        F = transform_filter_horizontal_edge(F, dHdx, sigma_H_i)
        F = F.T

        F = transform_filter_horizontal_edge(F, dVdy, sigma_H_i)
        F = F.T

    return F


def RF(img, sigma_s, sigma_r, num_iteration=1, joint_image=None):
    '''
    Python implementation of Domain Transform for Edge-Aware
    Image and Video Processing. Cite accordingly.
    (http://inf.ufrgs.br/~eslgastal/DomainTransform/)

    Written by Ali Karaali
    alixkaraali(classic_at_sign)gmail.com

    :param img: Input image to be filtered
    :param sigma_s: Spatial standard deviation
    :param sigma_r: Range standard deviation
    :param num_iteration: Number of iteration
    :param joint_image: Optional imge for joint filtering
    :return: Filtered image
    '''
    I = img

    if joint_image is None:
        J = I.copy()
    else:
        J = joint_image

    number_of_dim = J.ndim

    if number_of_dim == 3:
        h, w, num_joint_channels = J.shape
    else:
        h, w = J.shape
        num_joint_channels = 1
        J = J[:, :, np.newaxis]

    dIcdy = J[1:, :, :] - J[:h - 1, :, :]
    dIcdx = J[:, 1:, :] - J[:, :w - 1, :]

    dIdx = np.zeros((h, w))
    dIdy = np.zeros((h, w))

    for c in range(num_joint_channels):
        dIdx[:, 1:] += np.abs(dIcdx[:, :, c].reshape((h, w - 1)))
        dIdy[1:, :] += np.abs(dIcdy[:, :, c].reshape((h - 1, w)))

    dHdx = (1 + (sigma_s / sigma_r) * dIdx)
    dVdy = (1 + (sigma_s / sigma_r) * dIdy)

    dVdy = dVdy.T

    N = num_iteration

    F = I.copy()
    sigma_H = sigma_s

    for i in range(num_iteration):
        sigma_H_i = sigma_H * np.sqrt(3) * (2 ** (N - (i + 1))) / np.sqrt(4 ** N - 1)

        F = transform_filter_horizontal_edge(F, dHdx, sigma_H_i)
        F = image_transpose(F)

        F = transform_filter_horizontal_edge(F, dVdy, sigma_H_i)
        F = image_transpose(F)

    return F


def transform_filter_horizontal_edge(img, D, sigma):
    '''

    See paper for details
    Domain Transform for Edge-Aware
    Image and Video Processing. Cite accordingly.

    Written by Ali Karaali
    alixkaraali(classic_at_sign)gmail.com

    :param img:
    :param D:
    :param sigma:
    :return:
    '''
    a = np.exp(-1 * np.sqrt(2) / sigma)
    F = img.copy()
    V = a ** D

    ndim_c = img.ndim

    if ndim_c == 3:
        num_channels = 3
        h, w, c = img.shape

        for i in range(1, w):
            for c in range(num_channels):
                F[:, i, c] += np.multiply(V[:, i], F[:, i - 1, c] - F[:, i, c])

        for i in range(w - 2, -1, -1):
            for c in range(num_channels):
                F[:, i, c] += np.multiply(V[:, i + 1], F[:, i + 1, c] - F[:, i, c])
    else:
        h, w = img.shape

        for i in range(1, w):
            F[:, i] = F[:, i] + np.multiply(V[:, i], F[:, i - 1] - F[:, i])

        for i in range(w - 2, -1, -1):
            F[:, i] = F[:, i] + np.multiply(V[:, i + 1], F[:, i + 1] - F[:, i])

    return F
