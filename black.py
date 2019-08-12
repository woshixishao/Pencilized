import math
import numpy as np
from PIL import Image
from scipy import signal
from scipy.ndimage import interpolation
from scipy.sparse import csr_matrix, spdiags
from scipy.sparse.linalg import spsolve
from match import match
from util import im2double, rot90, rot90c
from stitch import horizontal_stitch, vertical_stitch

line_len_divisor = 40  # 卷积核与图片倍数关系

Lambda = 0.2
texture_resize_ratio = 0.2
texture_file_name = 'texture.jpg'


def get_stroke(J, gammaS=1):
    '''
    得到stroke structure
    Args:
        J:  灰度图片（矩阵）
        gammaS:   线条粗细控制参数
    Returns
        S: 输入图片的笔画结构，类型为矩阵
    '''
    h, w = J.shape
    line_len_double = float(min(h, w)) / line_len_divisor

    line_len = round(line_len_double)
    line_len += line_len % 2

    half_line_len = line_len / 2

    # 计算梯度

    dJ = im2double(J)
    Ix = np.column_stack((abs(dJ[:, 0:-1] - dJ[:, 1:]), np.zeros((h, 1))))
    Iy = np.row_stack((abs(dJ[0:-1, :] - dJ[1:, :]), np.zeros((1, w))))
    Imag = np.sqrt(Ix * Ix + Iy * Iy)

    # 分8个方向，L[:, :, index]是用来表示第index+1个方向的线段，作为卷积核
    L = np.zeros((line_len, line_len, 8))
    for n in range(8):
        if n == 0 or n == 1 or n == 2 or n == 7:
            for x in range(0, line_len):
                y = round(
                    ((x + 1) - half_line_len) * math.tan(math.pi / 8 * n))
                y = half_line_len - y
                if 0 < y <= line_len:
                    L[round(y - 1), x, n] = 1
                if n < 7:
                    L[:, :, n + 4] = rot90c(L[:, :, n])
    L[:, :, 3] = rot90(L[:, :, 7])

    G = np.zeros((J.shape[0], J.shape[1], 8))
    for n in range(8):
        # 选取最大值所在的方向
        G[:, :, n] = signal.convolve2d(Imag, L[:, :, n], "same")

    Gindex = G.argmax(axis=2)

    C = np.zeros((J.shape[0], J.shape[1], 8))
    for n in range(8):
        C[:, :, n] = Imag * (1 * (Gindex == n))

    # 将map set C 与方向向量L卷积，在每一个像素上生成线条
    Spn = np.zeros((J.shape[0], J.shape[1], 8))
    for n in range(8):
        Spn[:, :, n] = signal.convolve2d(C[:, :, n], L[:, :, n], "same")
    # 八个方向求和，归一化
    Sp = Spn.sum(axis=2)
    Sp = (Sp - Sp[:].min()) / (Sp[:].max() - Sp[:].min())
    S = (1 - Sp)**gammaS

    img = Image.fromarray(S * 255)

    return S


def get_tone(J, type, gammaI=1):
    '''
    tone rendering
    Args:
        J:   灰度图片（矩阵）
        type: 图片类型
        gammaI: 深浅控制参数
    Returns:
        T: 输入图片的色调，类型为矩阵
    '''
    # 直方图匹配
    Jadjusted = match(J, type=type)**gammaI
    # texture，铅笔画匹配图
    texture = Image.open(texture_file_name)
    texture = np.array(texture.convert("L"))
    # texture = np.array(texture)
    texture = texture[99:texture.shape[0] - 100, 99:texture.shape[1] - 100]

    ratio = texture_resize_ratio * min(J.shape[0], J.shape[1]) / float(1024)
    texture_resize = interpolation.zoom(texture, (ratio, ratio))
    texture = im2double(texture_resize)
    htexture = horizontal_stitch(texture, J.shape[1])
    Jtexture = vertical_stitch(htexture, J.shape[0])

    size = J.shape[0] * J.shape[1]

    nzmax = 2 * (size - 1)
    i = np.zeros((nzmax, 1))
    j = np.zeros((nzmax, 1))
    s = np.zeros((nzmax, 1))
    for m in range(1, nzmax + 1):
        i[m - 1] = round(math.ceil((m + 0.1) / 2)) - 1
        j[m - 1] = round(math.ceil((m - 0.1) / 2)) - 1
        s[m - 1] = -2 * (m % 2) + 1
    dx = csr_matrix((s.T[0], (i.T[0], j.T[0])), shape=(size, size))

    nzmax = 2 * (size - J.shape[1])
    i = np.zeros((nzmax, 1))
    j = np.zeros((nzmax, 1))
    s = np.zeros((nzmax, 1))
    for m in range(1, nzmax + 1):
        i[m - 1, :] = round(math.ceil((m - 1 + 0.1) / 2) +
                            J.shape[1] * (m % 2)) - 1
        j[m - 1, :] = math.ceil((m - 0.1) / 2) - 1
        s[m - 1, :] = -2 * (m % 2) + 1
    dy = csr_matrix((s.T[0], (i.T[0], j.T[0])), shape=(size, size))

    Jtexture1d = np.log(np.reshape(Jtexture.T,
                        (1, Jtexture.size), order="f") + 0.01)
    Jtsparse = spdiags(Jtexture1d, 0, size, size)
    Jadjusted1d = np.log(np.reshape(Jadjusted.T,
                         (1, Jadjusted.size), order="f").T + 0.01)

    nat = Jtsparse.T.dot(Jadjusted1d)  # lnJ(x)
    a = np.dot(Jtsparse.T, Jtsparse)
    b = dx.T.dot(dx)
    c = dy.T.dot(dy)
    mat = a + Lambda * (b + c)  # lnH(x)

    beta1d = spsolve(mat, nat)
    beta = np.reshape(beta1d, (J.shape[0], J.shape[1]), order="c")
    # 模拟铅笔来回“刷刷刷”，重复画beta次
    T = Jtexture**beta
    T = (T - T.min()) / (T.max() - T.min())

    img = Image.fromarray(T * 255)

    return T


def draw_black(imr, gammaS=1, gammaI=1):  # 画黑白铅笔画
    type = "color" if imr.mode == "RGB" else "black"
    im = imr.convert("L")
    J = np.array(im)
    S = get_stroke(J, gammaS=1)
    T = get_tone(J, type, gammaI=1)
    pencil = S * T
    img = Image.fromarray(pencil * 255)
    return img
