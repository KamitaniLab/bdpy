import numpy as np
from torchvision import transforms

def image_preprocess(img, image_mean=np.float32([104, 117, 123]), image_std=np.float32([1, 1, 1])):
    '''convert to Caffe's input image layout'''
    return (np.float32(np.transpose(img, (2, 0, 1))[::-1]) - np.reshape(image_mean, (3, 1, 1))) / np.reshape(image_std, (3, 1, 1))

def image_deprocess(img, image_mean=np.float32([104, 117, 123]), image_std=np.float32([1, 1, 1]), BGR2RGB=False):
    '''convert from Caffe's input image layout'''
    if BGR2RGB:
        return np.dstack((img * np.reshape(image_std, (3, 1, 1)) + np.reshape(image_mean, (3, 1, 1)))[::-1])
    else:
        return np.dstack((img * np.reshape(image_std, (3, 1, 1)) + np.reshape(image_mean, (3, 1, 1))))

def image_deprocess_in_tensor(img, image_mean=np.float32([104, 117, 123]), image_std=np.float32([1, 1, 1])):
    '''convert from Caffe's input image layout'''
    ### in this function, the order of color channels will not be changed ###
    deproc =  transforms.Compose([transforms.Normalize(mean=[0., 0., 0.], std=1/image_std),
                                  transforms.Normalize(mean=-image_mean, std=[1., 1., 1.])])
    return deproc(img)