import cv2
import numpy as np
import GuidedFilter
import time


def patch_max(img, w):
    '''
    :param img: input grayscale image(red channel or Maximum value of blue and green channels)
    :param w: window size
    :return: the maximum value of the input image
    '''
    return cv2.dilate(img, np.ones((w, w)))


def diff(img):
    """
    :param img: input original image
    :return: the largest differences among three different color channels
    """
    B, G, R = cv2.split(img)
    img_copy = img.copy()
    img_copy[:, :, 2] = 0
    R = patch_max(R, 5) * 1.0
    BG = patch_max(np.max(img_copy, axis=2), 5) * 1.0
    D = R - BG
    return D


def getTransmission(img, diff):
    '''
    :param img: input original image
    :param diff: the largest differences among three different color channels
    :return: transmission map
    '''
    t = diff + (1 - np.max(diff))
    guided_filter = GuidedFilter.GuidedFilter(img, 50, 0.001)
    t = guided_filter.filter(t)
    Max_t = np.max(t)
    Min_t = np.min(t)
    if Min_t >= 0.1 and Max_t >= 0.95:
        tao = min(Max_t - 0.95, Min_t - 0.1)
    elif Min_t <= 0.1 and Max_t <= 0.95:
        tao = min(Max_t - 0.95, Min_t - 0.1)
    else:
        tao = 0
    t = t - tao
    return t


def recover(I, B, tr, tg, tb):
    '''
    :param I: input original image
    :param B: background light
    :param tr tg tb: transmission map
    :return: recovery image
    '''
    rec = np.zeros(I.shape)
    tr = np.clip(tr, 0.15, 0.95)
    tg = np.clip(tg, 0.15, 0.95)
    tb = np.clip(tb, 0.15, 0.95)
    rec[:, :, 0] = ((I[:, :, 0] / 255 - B[0][0] / 255) / tb + B[0][0] / 255)
    rec[:, :, 1] = ((I[:, :, 1] / 255 - B[0][1] / 255) / tg + B[0][1] / 255)
    rec[:, :, 2] = ((I[:, :, 2] / 255 - B[0][2] / 255) / tr + B[0][2] / 255)
    cv2.imshow('r', rec * 255)
    cv2.waitKey(0)


def getBackgroundlightBG_scale(channel, scale, t):
    '''
    :param channel: input channel(blue or green)
    :param scale: Scale
    :param t: transmission map
    :return: background light of input channel
    '''
    channel = channel.copy()
    channel = cv2.resize(channel, (0, 0), fx=1 / scale, fy=1 / scale)
    t = t.copy()
    t = cv2.resize(t, (0, 0), fx=1 / scale, fy=1 / scale)
    M, N = channel.shape
    B_b = 0
    distance = 1000000000
    lst = []
    for i in range(1, 1000):
        J_B = (channel - (i / 1000) * (1 - t)) / t
        J_copy_1 = J_B.copy()
        J_copy = J_B.copy()
        J1 = np.clip(J_copy_1, -100, 0.95)
        J2 = np.clip(J_copy_1, 0, 100)
        J_removal = J_B - J1 + J_B - J2
        J_hist = np.clip(J_copy * 255, 0, 255).astype(np.uint8)
        hist = cv2.calcHist([J_hist], [0], None, [256], [0, 256]).reshape(1, 256)
        hist_standard = np.ones(hist.shape) * M * N / 256
        hist_removal = hist - hist_standard
        hist_over = np.clip(hist_removal, 0, M * N)
        loss = len(np.nonzero(J_removal)[0]) + np.sum(hist_over)
        lst.append(loss)
        if loss < distance:
            B_b = i / 1000
            distance = loss
    return B_b


def loss(i, K, B, t):
    B = B.copy()
    t = t.copy()
    M, N = B.shape
    J_B = (B - (i / K) * (1 - t)) / t
    J_copy_1 = J_B.copy()
    J_copy = J_B.copy()
    J1 = np.clip(J_copy_1, -100, 0.95)
    J2 = np.clip(J_copy_1, 0, K)
    J_removal = J_B - J1 + J_B - J2
    J_hist = np.clip(J_copy * 255, 0, 255).astype(np.uint8)
    hist = cv2.calcHist([J_hist], [0], None, [256], [0, 256]).reshape(1, 256)
    hist_standard = np.ones(hist.shape) * M * N / 256
    hist_removal = hist - hist_standard
    hist_over = np.clip(hist_removal, 0, M * N)
    loss = len(np.nonzero(J_removal)[0]) + np.sum(hist_over)
    return loss


def get_step(x, alpha, K, B, v, t):
    loss1 = loss(x - alpha, K, B, t)
    loss2 = loss(x, K, B, t)
    derivative = (loss2 - loss1) / alpha
    step = derivative * v
    return step, derivative


def getBackgroundlightBG_gradientDescent(channel, t):
    '''
    :param B: input channel(blue or green)
    :param t: transmission map
    :return: background light of input channel
    '''
    x = 20
    i = 0
    derivative = 1000
    while abs(derivative) > 10:
        step, derivative = get_step(x, 10, 1000, channel, 1 - 0.02 * i, t)
        x -= step
        i += 1
    B_b = x / 1000
    return B_b


def getBackgroundlight(img, t, scale, type):
    '''
    :param img: input original image
    :param t: transmission map
    :param scale: scale
    :param type: speed up type
    :return: background light of the input image
    '''
    B, G, R = cv2.split(img)
    if type == 'scale':
        bb = getBackgroundlightBG_scale(B, scale, t)
        bg = getBackgroundlightBG_scale(G, scale, t)
    elif type == 'gradientDescent':
        bb = getBackgroundlightBG_gradientDescent(B, t)
        bg = getBackgroundlightBG_gradientDescent(G, t)
    else:
        print('choose type "scale" or "gradientDescent"')
    I = img
    Jg = (I[:, :, 1] / 255 - bg / 255) / t + bg / 255
    Jb = (I[:, :, 0] / 255 - bb / 255) / t + bb / 255
    if bb >= 0 and abs(np.mean(Jb * 255) - np.mean(Jg * 255)) < 0.04:
        br = (np.mean(R) - (np.mean(Jb * 255) + np.mean(Jg * 255)) * 0.5 * np.mean(t)) / (1 - np.mean(t))
        return [[bb, bg, br]]
    elif bb < bg:
        Jg = (I[:, :, 1] / 255 - bg / 255) / t + bg / 255
        br = (np.mean(R) - (np.mean(Jg * 255)) * np.mean(t)) / (1 - np.mean(t))
        bb = (np.mean(B) - (np.mean(Jg * 255)) * np.mean(t)) / (1 - np.mean(t))
        return [[bb, bg, br]]
    elif bb > bg:
        Jb = (I[:, :, 1] / 255 - bg / 255) / t + bg / 255
        br = (np.mean(R) - (np.mean(Jb * 255)) * np.mean(t)) / (1 - np.mean(t))
        bg = (np.mean(G) - (np.mean(Jb * 255)) * np.mean(t)) / (1 - np.mean(t))
        return [[bb, bg, br]]


path = 'img/'  # image path
file = '3376'  # file name
scale = 16  # scaling size
type = 'gradientDescent'  # Strategy to Speed up: gradient descent
type = 'scale'  # Strategy to Speed up: Scaling down pictures to ask for background light
img1 = cv2.imread(path + file + '.jpg')
img = img1 / 255
# step1: Transmission Map Estimation
D = diff(img)
t = getTransmission(img1, D)
# step2: Background Light of Blue and Green Channels Estimation
# step3: Background Light of Red Channel Estimation
B = getBackgroundlight(img, t, scale, type='scale')
# step4: image restoration
recover(img, B, t, t, t)

# B = getBackgroundlight(img, t, scale, type='gradientDescent')
# recover(img, B, t, t, t)

