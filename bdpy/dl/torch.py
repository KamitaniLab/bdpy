'''PyTorch module.'''


import os

import numpy as np
from PIL import Image
import torch


class ImageDataset(torch.utils.data.Dataset):
    '''Pytoch dataset for images.'''

    def __init__(self, images, labels=None, label_dirname=False, resize=None, shape='chw', transform=None):
        '''
        Parameters
        ----------
        images : list
            List of image file paths.
        labels : list, optional
            List of image labels (default: image file names).
        label_dirname : bool, optional
            Use directory names as labels if True (default: False).
        resize : None or tuple, optional
            If not None, images will be resized by the specified size.
        shape : str ({'chw', 'hwc', ...}), optional
            Specify array shape (channel, hieght, and width).
        transform
            Transformers (applied after resizing, reshaping, ans scaling to [0, 1])

        Note
        ----
        - Images are converted to RGB. Alpha channels in RGBA images are ignored.
        '''

        image_labels = []
        for imf in images:
            # TODO: validate the image file
            if label_dirname:
                image_labels.append(os.path.basename(os.path.dirname(imf)))
            else:
                image_labels.append(os.path.basename(imf))

        self.data_path = images
        if not labels is None:
            self.labels = labels
        else:
            self.labels = image_labels
        self.n_sample = len(images)
        self.transform = transform
        # Custom transforms
        self.__shape = shape
        self.__resize = resize

    def __len__(self):
        return self.n_sample

    def __getitem__(self, idx):
        img = Image.open(self.data_path[idx])

        # CMYK, RGBA --> RGB
        if img.mode == 'CMYK':
            img = img.convert('RGB')
        if img.mode == 'RGBA':
            bg = PIL.Image.new('RGB', img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            img = bg

        data = np.asarray(img)

        # Resize the image
        if not self.__resize is None:
            data = np.array(Image.fromarray(data).resize(self.__resize, resample=2))  # bicubic

        # Reshape
        s2d = {'h': 0, 'w': 1, 'c': 2}
        data = data.transpose((s2d[self.__shape[0]],
                               s2d[self.__shape[1]],
                               s2d[self.__shape[2]]))

        # Scaling to [0, 1]
        data = data / 255.

        if not self.transform is None:
            date = self.transform(data)
        else:
            data = torch.Tensor(data)
        label = self.labels[idx]
        return data, label
