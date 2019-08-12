import math
import numpy as np


# Heaviside function
def heaviside(x):
    return x if x >= 0 else 0


def laplace(x):
    '''
    Laplace distribution模拟bright layer, thetaB = 9
    Args
        x
    Returns:
        Laplace分布
    '''
    return float(1) / 9 * math.exp(-(256 - x) / float(9)) * heaviside(256 - x)


def uniform(x):
    '''
    Uniform distribution模拟mild layer,a = 105, b = 225
    Args:
        x
    Returns:
        均匀分布
    '''
    return float(1) / (225 - 105) * (heaviside(x - 105) - heaviside(x - 225))


def gaussian(x):
    '''
    Gaussian distribution模拟dark layer,ud = 11, thetaD = 11
    Args:
        x
    Returns:
        高斯分布
    '''
    return (float(1) / math.sqrt(2 * math.pi * 11) *
            math.exp(-((x - 90) ** 2) / float(2 * (11 ** 2))))


def re(x, type="black"):
    '''
    调节3个函数不同的权重， 用最大似然估计权重的值
    Args:
        x
        type: 图片类型
    Returns:
        权重值
    '''
    if type == "color":
        return 62 * laplace(x) + 30 * uniform(x) + 5 * gaussian(x)
    else:
        return 76 * laplace(x) + 22 * uniform(x) + 2 * gaussian(x)


def match(I, type="black"):
    '''
    输入图片直方图与理论铅笔画直方图进行匹配
    Args：
        I:灰度图像，0~255
        type: 图片类型
    Returns:
        匹配后的图片
    '''
    ho = np.zeros((1, 256))
    po = np.zeros((1, 256))
    for i in range(256):
        po[0, i] = sum(sum(1 * (I == i)))
    po /= float(sum(sum(po)))
    ho[0, 0] = po[0, 0]
    for i in range(1, 256):
        ho[0, i] = ho[0, i - 1] + po[0, i]

    histo = np.zeros((1, 256))
    prob = np.zeros((1, 256))
    for i in range(256):
        # prob[0, i] = p(i+1) # eq.4
        prob[0, i] = re(i, type)
    prob /= float(sum(sum(prob)))
    histo[0] = prob[0]
    for i in range(1, 256):
        histo[0, i] = histo[0, i - 1] + prob[0, i]

    Iadjusted = np.zeros((I.shape[0], I.shape[1]))
    for x in range(I.shape[0]):
        for y in range((I.shape[1])):
            histogram_value = ho[0, I[x, y]]
            index = (abs(histo - histogram_value)).argmin()
            Iadjusted[x, y] = index
    Iadjusted /= float(255)
    return Iadjusted
