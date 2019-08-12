import cv2
import numpy as np
from PIL import Image
from black import get_stroke, get_tone


def draw_color(imr, gammaS=1, gammaI=1):
    '''
    draw_color
    Args:
        imr: picture
        gammaS=1
        gammaI=1
    Returns:
        img
    如果图片是RGB的，转换到YCbCr进行操作
    '''
    if imr.mode == 'RGB':
        ycbcr = imr.convert('YCbCr')
        Iruv = np.ndarray((imr.size[1], imr.size[0], 3), 'u1', ycbcr.tobytes())
        type = "colour"
    else:
        Iruv = np.array(imr)
        type = "black"

    S = get_stroke(Iruv[:, :, 0], gammaS=gammaS)
    T = get_tone(Iruv[:, :, 0], type, gammaI=gammaI)
    Ypencil = S * T

    new_Iruv = Iruv.copy()
    new_Iruv.flags.writeable = True
    new_Iruv[:, :, 0] = Ypencil * 255

    # 重新转回BGR
    R = cv2.cvtColor(new_Iruv, cv2.COLOR_YCR_CB2BGR)
    img = Image.fromarray(R)
    return img
