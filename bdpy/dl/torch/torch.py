'''PyTorch module.'''

__all__ = ['FeatureExtractor', 'ImageDataset']


import os

import numpy as np
from PIL import Image
import torch


class FeatureExtractor(object):
    def __init__(self, encoder, layers=None, layer_mapping=None,
                 device='cpu', detach=True, targets=None,
                 return_final_output=False, final_output_saving_name='model_output',
                 sample_axis_info=None):
        self._encoder = encoder
        self.__layers = layers
        self.__layer_map = layer_mapping
        self.__detach = detach
        self.__device = device

        self._extractor = FeatureExtractorHandle()

        self._encoder.to(self.__device)

        targets = self.get_target_dict(targets)

        self.sample_axis_dict = self.get_sample_axis_dict(sample_axis_info)
        self.return_final_output = return_final_output
        self.final_output_saving_name = final_output_saving_name
        if return_final_output and final_output_saving_name in self.__layers:
            self.__layers.remove(final_output_saving_name)

        self.hook_handles = []
        for layer in self.__layers:
            target = targets[layer]
            if self.__layer_map is not None:
                layer = self.__layer_map[layer]
            exec('handle = self._encoder.{}.register_forward_hook(self._extractor.get_extraction_function("{}"))'.format(layer, target))
            exec('self.hook_handles.append(handle)')

    def __del__(self):
        for handle in self.hook_handles:
            handle.remove()
        self._extractor.clear()
        del self._extractor

    def get_target_dict(self, targets):
        default_target = 'module_out'
        if targets is None:
            targets = {}
        elif isinstance(targets, list):
            target_dict = {}
            for i, target in enumerate(targets):
                target_dict[self.__layers[i]] = target
            targets = target_dict
        elif isinstance(targets, str):
            assert targets in ['module_in', 'module_out']
            default_target = targets
            targets = {}
        assert isinstance(targets, dict)
        for layer in self.__layers:
            if layer not in targets:
                targets[layer] = default_target
        return targets

    def get_sample_axis_dict(self, sample_axis_info):
        if sample_axis_info is None:
            return None
        default_sample_axis = 0
        if isinstance(sample_axis_info, list):
            sample_axis_dict = {}
            for i, sample_axis in enumerate(sample_axis_info):
                if not isinstance(sample_axis, int):
                    for j, sub_module_saving_name in enumerate(self.__layers[i]):
                        if sub_module_saving_name is not None:
                            sample_axis_dict[sub_module_saving_name] = sample_axis[j]
                else:
                    sample_axis_dict[self.__layers[i]] = sample_axis
        elif isinstance(sample_axis_info, int):
            default_sample_axis = sample_axis_info
            sample_axis_dict = {}
        else:
            assert isinstance(sample_axis_info, dict)
            sample_axis_dict = sample_axis_info
        for layer in self.__layers:
            if layer not in sample_axis_dict:
                sample_axis_dict[layer] = default_sample_axis
        return sample_axis_dict

    def __call__(self, x) -> dict:
        return self.run(x)

    def run(self, x) -> dict:
        self._extractor.clear()
        if not isinstance(x, torch.Tensor):
            xt = torch.tensor(x[np.newaxis], device=self.__device)
        else:
            xt = x

        final_output = self._encoder.forward(xt)

        features = {}
        for feature, layer in zip(self._extractor.outputs, self.__layers):
            if isinstance(feature, tuple): # This is true for "module_in", and "module_out" for some types of layers such as attention layers
                assert isinstance(layer, tuple)
                assert len(feature) == len(layer), print(len(feature), len(layer))
                for i, sublayer in enumerate(layer):
                    if sublayer is None:
                        continue
                    assert feature[i] is not None
                    features[sublayer] = feature[i]
            else:
                features[layer] = feature

        if self.return_final_output:
            features[self.final_output_saving_name] = final_output
        if self.__detach:
            features = {
                k: v.cpu().detach().numpy()
                for k, v in features.items()
            }

        features = self.change_sample_axis(features)

        return features

    def change_sample_axis(self, features):
        if self.sample_axis_dict is None:
            return features
        else:
            for layer, sample_axis in self.sample_axis_dict.items():
                if layer in features:
                    if sample_axis != 0:
                        feature = features[layer]
                        ndim = feature.ndim
                        axes = list(range(ndim))
                        axes.remove(sample_axis)
                        axes.insert(0, sample_axis)
                        if self.__detach:
                            features[layer] = feature.transpose(*axes)
                        else:
                            features[layer] = feature.permute(*axes)
            return features


class FeatureExtractorHandle(object):
    def __init__(self):
        self.outputs = []

    def get_extraction_function(self, target: str):
        assert target in ['module_in', 'module_out']
        if target == 'module_in':
            return self.save_input
        else: # target == self.save_output
            return self.save_output

    def save_output(self, module, module_in, module_out):
        self.outputs.append(module_out)

    def save_input(self, module, module_in, module_out):
        self.outputs.append(module_in)

    def clear(self):
        self.outputs = []


class ImageDataset(torch.utils.data.Dataset):
    '''Pytoch dataset for images.'''

    def __init__(self, images, labels=None, label_dirname=False, resize=None, shape='chw', transform=None, scale=1, rgb_mean=None, preload=False, preload_limit=np.inf):
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
        transform : optional
            Transformers (applied after resizing, reshaping, ans scaling to [0, 1])
        scale : optional
            Image intensity is scaled to [0, scale] (default: 1).
        rgb_mean : list([r, g, b]), optional
            Image values are centered by the specified mean (after scaling) (default: None).
        preload : bool, optional
            Pre-load images (default: False).
        preload_limit : int
            Memory size limit of preloading in GiB (default: unlimited).

        Note
        ----
        - Images are converted to RGB. Alpha channels in RGBA images are ignored.
        '''

        self.transform = transform
        # Custom transforms
        self.__shape = shape
        self.__resize = resize
        self.__scale = scale
        self.__rgb_mean = rgb_mean

        self.__data = {}
        preload_size = 0
        image_labels = []
        for i, imf in enumerate(images):
            # TODO: validate the image file
            if label_dirname:
                image_labels.append(os.path.basename(os.path.dirname(imf)))
            else:
                image_labels.append(os.path.basename(imf))
            if preload:
                data = self.__load_image(imf)
                data_size = data.size * data.itemsize
                if preload_size + data_size > preload_limit * (1024 ** 3):
                    preload = False
                    continue
                self.__data.update({i: data})
                preload_size += data_size

        self.data_path = images
        if not labels is None:
            self.labels = labels
        else:
            self.labels = image_labels
        self.n_sample = len(images)

    def __len__(self):
        return self.n_sample

    def __getitem__(self, idx):
        if idx in self.__data:
            data = self.__data[idx]
        else:
            data = self.__load_image(self.data_path[idx])

        if not self.transform is None:
            data = self.transform(data)
        else:
            data = torch.Tensor(data)

        label = self.labels[idx]

        return data, label

    def __load_image(self, fpath):
        img = Image.open(fpath)

        # CMYK, RGBA --> RGB
        if img.mode == 'CMYK':
            img = img.convert('RGB')
        if img.mode == 'RGBA':
            bg = Image.new('RGB', img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            img = bg

        data = np.asarray(img)

        # Monotone to RGB
        if data.ndim == 2:
            data = np.stack([data, data, data], axis=2)

        # Resize the image
        if not self.__resize is None:
            data = np.array(Image.fromarray(data).resize(self.__resize, resample=2))  # bicubic

        # Reshape
        s2d = {'h': 0, 'w': 1, 'c': 2}
        data = data.transpose((s2d[self.__shape[0]],
                               s2d[self.__shape[1]],
                               s2d[self.__shape[2]]))

        # Scaling to [0, scale]
        data = (data / 255.) * self.__scale

        # Centering
        if not self.__rgb_mean is None:
            data[0] -= self.__rgb_mean[0]
            data[1] -= self.__rgb_mean[1]
            data[2] -= self.__rgb_mean[2]

        return data
