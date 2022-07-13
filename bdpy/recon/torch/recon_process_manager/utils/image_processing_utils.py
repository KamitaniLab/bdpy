import numpy as np
from torchvision import transforms

__all__ = ['image_preprocess', 'image_deprocess', 'image_deprocess_in_tensor', 'get_image_deprocess_fucntion', 'get_image_preprocess_in_tensor_function']

def image_preprocess(img, image_mean=np.float32([104, 117, 123]), image_std=np.float32([1, 1, 1])):
    '''
    deal with images represented as np.array in the shape of HxWxC
    '''
    return (img - np.reshape(image_mean, (1, 1, 3))) / np.reshape(image_std, (1, 1, 3))

def image_deprocess(img, image_mean=np.float32([104, 117, 123]), image_std=np.float32([1, 1, 1]), BGR2RGB=False):
    '''
    deal with images represented as np.array in the shape of HxWxC
    '''
    if BGR2RGB:
        return (img * np.reshape(image_std, (1, 1, 3)) + np.reshape(image_mean, (1, 1, 3)))[:,:,::-1]
    else:
        return img * np.reshape(image_std, (1, 1, 3)) + np.reshape(image_mean, (1, 1, 3))

def get_image_deprocess_fucntion(image_mean=np.float32([104, 117, 123]), image_std=np.float32([1, 1, 1]), BGR2RGB=False):
    return lambda img_array: image_deprocess(img_array, image_mean=image_mean, image_std=image_std, BGR2RGB=BGR2RGB)

def image_preprocess_in_tensor(img, image_mean=np.float32([104, 117, 123]), image_std=np.float32([1, 1, 1])):
    ### in this function, the order of color channels will not be changed ###
    preproc = transforms.Compose([transforms.Normalize(mean=image_mean, std=[1.,1.,1.]),
                                  transforms.Normalize(mean=[0.,0.,0.], std=image_std)])
    return preproc(img)

def image_deprocess_in_tensor(img, image_mean=np.float32([104, 117, 123]), image_std=np.float32([1, 1, 1])):
    ### in this function, the order of color channels will not be changed ###
    deproc = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.], std=1/image_std),
                                 transforms.Normalize(mean=-image_mean, std=[1., 1., 1.])])
    return deproc(img)

def get_image_preprocess_in_tensor_function(image_mean=np.float32([104, 117, 123]), image_std=np.float32([1, 1, 1])):
    return lambda img_tsr: image_preprocess_in_tensor(img_tsr, image_mean=image_mean, image_std=image_std)