import numpy as np
import cv2


def load_video(file_path, dtype='float', bgr=False, height=224, width=224):
    """
    ret_type select return as float32 float54, or uint8
    ret_type allows ['float', 'float64']
    """
    cap = cv2.VideoCapture(file_path)
    vid = []
    while True:
        ret, img = cap.read()
        if not ret:
            break
        # torchvision model allow RGB image , range of [0,1] and normalised
        if bgr:
            img = cv2.resize(img, (height, width))
        else:
            img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                             (height, width))
        if dtype == 'float32':
            img = img.astype(np.float32)
        elif dtype == 'float':
            img = img.astype(np.float)
        vid.append(img)
    return np.array(vid)
