import numpy as np


def RemoveBlackFilling(source_img):
    # Remove the unnecessary black filling
    contour = np.sum(source_img, axis=-1)
    bottom = np.max(np.argmin(contour, axis=0))
    right = np.max(np.argmin(contour, axis=1))

    if bottom == 0:
        x1 = np.min(np.argmax(contour, axis=0))
        x2 = source_img.shape[0]
    else:
        x1 = 0
        x2 = bottom

    if right == 0:
        y1 = np.min(np.argmax(contour, axis=1))
        y2 = source_img.shape[1]
    else:
        y1 = 0
        y2 = right

    img_result = source_img[x1:x2, y1:y2, :]

    return img_result
