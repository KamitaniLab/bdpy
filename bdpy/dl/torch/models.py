"""Model definitions."""


from typing import Dict, Union, Optional, Sequence

import re
from functools import reduce

import torch
import torch.nn as nn


def layer_map(net: str) -> Dict[str, str]:
    """Get layer map for a given network.

    Parameters
    ----------
    net : str
        Network name. Currently, 'vgg19' and 'alexnet' are supported.

    Returns
    -------
    Dict[str, str]
        Layer map. Keys are human-readable layer names, and values are
        corresponding layer names in the network.
    """

    maps = {
        'vgg19': {
            'conv1_1': 'features[0]',
            'conv1_2': 'features[2]',
            'conv2_1': 'features[5]',
            'conv2_2': 'features[7]',
            'conv3_1': 'features[10]',
            'conv3_2': 'features[12]',
            'conv3_3': 'features[14]',
            'conv3_4': 'features[16]',
            'conv4_1': 'features[19]',
            'conv4_2': 'features[21]',
            'conv4_3': 'features[23]',
            'conv4_4': 'features[25]',
            'conv5_1': 'features[28]',
            'conv5_2': 'features[30]',
            'conv5_3': 'features[32]',
            'conv5_4': 'features[34]',
            'fc6':     'classifier[0]',
            'relu6':   'classifier[1]',
            'fc7':     'classifier[3]',
            'relu7':   'classifier[4]',
            'fc8':     'classifier[6]',
        },

        'alexnet': {
            'conv1': 'features[0]',
            'conv2': 'features[4]',
            'conv3': 'features[8]',
            'conv4': 'features[10]',
            'conv5': 'features[12]',
            'fc6':   'classifier[0]',
            'relu6': 'classifier[1]',
            'fc7':   'classifier[2]',
            'relu7': 'classifier[3]',
            'fc8':   'classifier[4]',
        },

        'reference_net': {
            'conv1': 'features[0]',
            'relu1': 'features[1]',
            'conv2': 'features[4]',
            'relu2': 'features[5]',
            'conv3': 'features[8]',
            'relu3': 'features[9]',
            'conv4': 'features[10]',
            'relu4': 'features[11]',
            'conv5': 'features[12]',
            'relu5': 'features[13]',
            'fc6':   'classifier[1]',
            'relu6': 'classifier[2]',
            'fc7':   'classifier[4]',
            'relu7': 'classifier[5]',
            'fc8':   'classifier[6]',
        }
    }
    return maps[net]


def _parse_layer_name(model: nn.Module, layer_name: str) -> nn.Module:
    """Parse layer name and return the corresponding layer object.

    Parameters
    ----------
    model : nn.Module
        Network model.
    layer_name : str
        Layer name. It accepts the following formats: 'layer_name',
        '[index]', 'parent_name.child_name', and combinations of them.

    Returns
    -------
    nn.Module
        Layer object.

    Examples
    --------
    >>> model = nn.Module()
    >>> model.layer1 = nn.Linear(10, 10)
    >>> model.layers = nn.Sequential(nn.Conv2d(3, 3, 3), nn.Conv2d(3, 3, 3))
    >>> _parse_layer_name(model, 'layer1')
    Linear(in_features=10, out_features=10, bias=True)
    >>> _parse_layer_name(model, 'layers[0]')
    Conv2d(3, 3, kernel_size=(3, 3), stride=(1, 1))
    """

    def _get_value_by_indices(array, indices):
        return reduce(lambda arr, index: arr[index], indices, array)

    if hasattr(model, layer_name):
        return getattr(model, layer_name)

    # parse layer name having parent name (e.g., 'features.conv1')
    if '.' in layer_name:
        top_most_layer_name, child_layer_name = layer_name.split('.', 1)
        model = _parse_layer_name(model, top_most_layer_name)
        return _parse_layer_name(model, child_layer_name)

    # parse layer name having index (e.g., '[0]', 'features[0]', 'backbone[0][1]')
    pattern = re.compile(r'^(?P<layer_name>[a-zA-Z_]+[a-zA-Z0-9_]*)?(?P<index>(\[(\d+)\])+)$')
    m = pattern.match(layer_name)
    if m is not None:
        layer_name: str | None = m.group('layer_name')  # NOTE: layer_name can be None
        index_str = m.group('index')

        indeces = re.findall(r'\[(\d+)\]', index_str)
        indeces = [int(i) for i in indeces]

        if isinstance(layer_name, str) and hasattr(model, layer_name):
            model = getattr(model, layer_name)
        return _get_value_by_indices(model, indeces)

    raise ValueError(
        f"Invalid layer name: '{layer_name}'. Either the syntax of '{layer_name}' is not supported, "
        f"or {type(model).__name__} object has no attribute '{layer_name}'.")


def model_factory(name: str) -> nn.Module:
    """Make a model instrance."""

    if name == "alexnet":
        return AlexNet()
    elif name == "referencenet":
        return ReferenceNet()
    elif name == "vgg19":
        return VGG19()
    elif name == "relu7generator":
        return AlexNetGenerator()
    elif name == "relu6generator":
        return AlexNetGenerator()
    elif name == "pool5generator":
        return AlexNetPool5Generator()
    elif name == "relu4generator":
        return AlexNetRelu4Generator()
    elif name == "relu3generator":
        return AlexNetRelu3Generator()
    elif name == "norm2generator":
        return AlexNetNorm2Generator()
    elif name == "norm1generator":
        return AlexNetNorm1Generator()
    else:
        raise ValueError(f"Unknwon model: {name}")


class VGG19(nn.Module):
    def __init__(self):

        super(VGG19, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 64, kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0,
                         dilation=1, ceil_mode=False),
            nn.Conv2d(64, 128, kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=False),
            nn.Conv2d(128, 128, kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0,
                         dilation=1, ceil_mode=False),
            nn.Conv2d(128, 256, kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0,
                         dilation=1, ceil_mode=False),
            nn.Conv2d(256, 512, kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0,
                         dilation=1, ceil_mode=False),
            nn.Conv2d(512, 512, kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0,
                         dilation=1, ceil_mode=False),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(in_features=25088, out_features=4096, bias=True),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=1000, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000) -> None:

        super(AlexNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(inplace=False),
            nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2),
            nn.ReLU(inplace=False),
            nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=False),
            nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=False),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=False),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class ReferenceNet(nn.Module):
    def __init__(self, num_classes: int = 1000) -> None:
        super(ReferenceNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75),
            nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=False),
            nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=False),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=False),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class AlexNetGenerator(nn.Module):

    def __init__(
            self, input_size: int = 4096, n_out_channel: int = 3,
            device: Optional[Union[str, Sequence[str]]] = None):

        super(AlexNetGenerator, self).__init__()

        if device is None:
            self.__device0 = 'cpu'
            self.__device1 = 'cpu'
        elif isinstance(device, str):
            self.__device0 = device
            self.__device1 = device
        else:
            self.__device0 = device[0]
            self.__device1 = device[1]

        self.defc7 = nn.Linear(input_size, 4096)
        self.relu_defc7 = nn.LeakyReLU(0.3, inplace=True)

        self.defc6 = nn.Linear(4096, 4096)
        self.relu_defc6 = nn.LeakyReLU(0.3, inplace=True)

        self.defc5 = nn.Linear(4096, 4096)
        self.relu_defc5 = nn.LeakyReLU(0.3, inplace=True)

        self.deconv5 = nn.ConvTranspose2d(
            256, 256, kernel_size=4, stride=2, padding=1, bias=True)
        self.relu_deconv5 = nn.LeakyReLU(0.3, inplace=True)

        self.conv5_1 = nn.ConvTranspose2d(
            256, 512, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu_conv5_1 = nn.LeakyReLU(0.3, inplace=True)

        self.deconv4 = nn.ConvTranspose2d(
            512, 256, kernel_size=4, stride=2, padding=1, bias=True)
        self.relu_deconv4 = nn.LeakyReLU(0.3, inplace=True)

        self.conv4_1 = nn.ConvTranspose2d(
            256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu_conv4_1 = nn.LeakyReLU(0.3, inplace=True)

        self.deconv3 = nn.ConvTranspose2d(
            256, 128, kernel_size=4, stride=2, padding=1, bias=True)
        self.relu_deconv3 = nn.LeakyReLU(0.3, inplace=True)

        self.conv3_1 = nn.ConvTranspose2d(
            128, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu_conv3_1 = nn.LeakyReLU(0.3, inplace=True)

        self.deconv2 = nn.ConvTranspose2d(
            128, 64, kernel_size=4, stride=2, padding=1, bias=True)
        self.relu_deconv2 = nn.LeakyReLU(0.3, inplace=True)

        self.deconv1 = nn.ConvTranspose2d(
            64, 32, kernel_size=4, stride=2, padding=1, bias=True)
        self.relu_deconv1 = nn.LeakyReLU(0.3, inplace=True)

        self.deconv0 = nn.ConvTranspose2d(
            32, n_out_channel, kernel_size=4, stride=2, padding=1, bias=True)

        self.defc = nn.Sequential(
            self.defc7,
            self.relu_defc7,
            self.defc6,
            self.relu_defc6,
            self.defc5,
            self.relu_defc5,
        ).to(self.__device0)

        self.deconv = nn.Sequential(
            self.deconv5,
            self.relu_deconv5,
            self.conv5_1,
            self.relu_conv5_1,
            self.deconv4,
            self.relu_deconv4,
            self.conv4_1,
            self.relu_conv4_1,
            self.deconv3,
            self.relu_deconv3,
            self.conv3_1,
            self.relu_conv3_1,
            self.deconv2,
            self.relu_deconv2,
            self.deconv1,
            self.relu_deconv1,
            self.deconv0,
        ).to(self.__device1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:

        f = self.defc(z)
        f = f.view(-1, 256, 4, 4)
        g = self.deconv(f)

        return g


class AlexNetPool5Generator(nn.Module):
    """From caffe pool5 generator of bvlc_reference_caffenet.
    The model trained by DeepSim using ILSVRC dataset.
    Provided by Alexey Dosovitskiy.
    """

    def __init__(
            self, device: Optional[Union[str, Sequence[str]]] = None):
        super(AlexNetPool5Generator, self).__init__()

        if device is None:
            self.__device0 = 'cpu'
            self.__device1 = 'cpu'
        elif isinstance(device, str):
            self.__device0 = device
            self.__device1 = device
        else:
            self.__device0 = device[0]
            self.__device1 = device[1]
        # input 256x6x6
        self.Rconv6 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.Rrelu6 = nn.LeakyReLU(0.3, inplace=True)
        self.Rconv7 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.Rrelu7 = nn.LeakyReLU(0.3, inplace=True)
        self.Rconv8 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0)
        self.Rrelu8 = nn.LeakyReLU(0.3, inplace=True)
        self.deconv5 = nn.ConvTranspose2d(
            512, 256, kernel_size=4, stride=2, padding=1, bias=True)
        self.relu_deconv5 = nn.LeakyReLU(0.3, inplace=True)
        self.conv5_1 = nn.ConvTranspose2d(
            256, 512, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu_conv5_1 = nn.LeakyReLU(0.3, inplace=True)
        self.deconv4 = nn.ConvTranspose2d(
            512, 256, kernel_size=4, stride=2, padding=1, bias=True)
        self.relu_deconv4 = nn.LeakyReLU(0.3, inplace=True)
        self.conv4_1 = nn.ConvTranspose2d(
            256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu_conv4_1 = nn.LeakyReLU(0.3, inplace=True)
        self.deconv3 = nn.ConvTranspose2d(
            256, 128, kernel_size=4, stride=2, padding=1, bias=True)
        self.relu_deconv3 = nn.LeakyReLU(0.3, inplace=True)
        self.conv3_1 = nn.ConvTranspose2d(
            128, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu_conv3_1 = nn.LeakyReLU(0.3, inplace=True)
        self.deconv2 = nn.ConvTranspose2d(
            128, 64, kernel_size=4, stride=2, padding=1, bias=True)
        self.relu_deconv2 = nn.LeakyReLU(0.3, inplace=True)
        self.deconv1 = nn.ConvTranspose2d(
            64, 32, kernel_size=4, stride=2, padding=1, bias=True)
        self.relu_deconv1 = nn.LeakyReLU(0.3, inplace=True)
        self.deconv0 = nn.ConvTranspose2d(
            32, 3, kernel_size=4, stride=2, padding=1, bias=True)
        self.rconv = nn.Sequential(
            self.Rconv6,
            self.Rrelu6,
            self.Rconv7,
            self.Rrelu7,
            self.Rconv8,
            self.Rrelu8,
        ).to(self.__device0)
        self.deconv = nn.Sequential(
            self.deconv5,
            self.relu_deconv5,
            self.conv5_1,
            self.relu_conv5_1,
            self.deconv4,
            self.relu_deconv4,
            self.conv4_1,
            self.relu_conv4_1,
            self.deconv3,
            self.relu_deconv3,
            self.conv3_1,
            self.relu_conv3_1,
            self.deconv2,
            self.relu_deconv2,
            self.deconv1,
            self.relu_deconv1,
            self.deconv0,
        ).to(self.__device1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        f = self.rconv(z)
        g = self.deconv(f)
        return g


class AlexNetRelu4Generator(nn.Module):
    """From caffe relu4 generator of bvlc_reference_caffenet.
    The model trained by DeepSim using ILSVRC dataset.
    Provided by Alexey Dosovitskiy.
    """

    def __init__(
            self, device: Optional[Union[str, Sequence[str]]] = None):
        super(AlexNetRelu4Generator, self).__init__()

        if device is None:
            self.__device0 = 'cpu'
            self.__device1 = 'cpu'
        elif isinstance(device, str):
            self.__device0 = device
            self.__device1 = device
        else:
            self.__device0 = device[0]
            self.__device1 = device[1]
        # input 384x13x13
        self.Rconv6 = nn.Conv2d(384, 384, kernel_size=3,
                                stride=1, padding=0)  # 384x11x11
        self.Rrelu6 = nn.LeakyReLU(0.3, inplace=True)
        self.Rconv7 = nn.Conv2d(384, 512, kernel_size=3,
                                stride=1, padding=0)  # 512x9x9
        self.Rrelu7 = nn.LeakyReLU(0.3, inplace=True)
        self.Rconv8 = nn.Conv2d(512, 512, kernel_size=2,
                                stride=1, padding=0)  # 512x8x8
        self.Rrelu8 = nn.LeakyReLU(0.3, inplace=True)
        self.deconv5 = nn.ConvTranspose2d(
            512, 256, kernel_size=4, stride=2, padding=1, bias=True)  # 256x12x12
        self.relu_deconv5 = nn.LeakyReLU(0.3, inplace=True)
        self.conv5_1 = nn.ConvTranspose2d(
            256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu_conv5_1 = nn.LeakyReLU(0.3, inplace=True)
        self.deconv4 = nn.ConvTranspose2d(
            256, 128, kernel_size=4, stride=2, padding=1, bias=True)
        self.relu_deconv4 = nn.LeakyReLU(0.3, inplace=True)
        self.conv4_1 = nn.ConvTranspose2d(
            128, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu_conv4_1 = nn.LeakyReLU(0.3, inplace=True)
        self.deconv3 = nn.ConvTranspose2d(
            128, 128, kernel_size=4, stride=2, padding=1, bias=True)
        self.relu_deconv3 = nn.LeakyReLU(0.3, inplace=True)
        self.conv3_1 = nn.ConvTranspose2d(
            128, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu_conv3_1 = nn.LeakyReLU(0.3, inplace=True)
        self.deconv2 = nn.ConvTranspose2d(
            128, 64, kernel_size=4, stride=2, padding=1, bias=True)
        self.relu_deconv2 = nn.LeakyReLU(0.3, inplace=True)
        self.conv2_1 = nn.Conv2d(
            64, 32, kernel_size=3, stride=1, padding=1, bias=True)  # 32x128x128
        self.relu_conv2_1 = nn.LeakyReLU(0.3, inplace=True)
        self.deconv1 = nn.ConvTranspose2d(
            32, 16, kernel_size=4, stride=2, padding=1, bias=True)  # 3x256x256
        self.relu_deconv1 = nn.LeakyReLU(0.3, inplace=True)
        self.conv1_1 = nn.Conv2d(
            16, 3, kernel_size=3, stride=1, padding=1, bias=True)  # 3x256x256
        self.conv1_1_tanh = nn.Tanh()  # Hyperbolic Tangent (Tanh) function  # 3x256x256
        self.rconv = nn.Sequential(
            self.Rconv6,
            self.Rrelu6,
            self.Rconv7,
            self.Rrelu7,
            self.Rconv8,
            self.Rrelu8,
        ).to(self.__device0)
        self.deconv = nn.Sequential(
            self.deconv5,
            self.relu_deconv5,
            self.conv5_1,
            self.relu_conv5_1,
            self.deconv4,
            self.relu_deconv4,
            self.conv4_1,
            self.relu_conv4_1,
            self.deconv3,
            self.relu_deconv3,
            self.conv3_1,
            self.relu_conv3_1,
            self.deconv2,
            self.relu_deconv2,
            self.conv2_1,
            self.relu_conv2_1,
            self.deconv1,
            self.relu_deconv1,
            self.conv1_1,
            self.conv1_1_tanh,
        ).to(self.__device1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        f = self.rconv(z)
        g = self.deconv(f)
        g = g * 255  # conv1_1_tanh_elt layer's scaling
        return g


class AlexNetRelu3Generator(nn.Module):
    """From caffe relu3 generator of bvlc_reference_caffenet.
    The model trained by DeepSim using ILSVRC dataset.
    Provided by Alexey Dosovitskiy.
    """

    def __init__(
            self, device: Optional[Union[str, Sequence[str]]] = None):
        super(AlexNetRelu3Generator, self).__init__()

        if device is None:
            self.__device0 = 'cpu'
            self.__device1 = 'cpu'
        elif isinstance(device, str):
            self.__device0 = device
            self.__device1 = device
        else:
            self.__device0 = device[0]
            self.__device1 = device[1]
        # input 384x13x13
        self.Rconv6 = nn.Conv2d(384, 384, kernel_size=3,
                                stride=1, padding=0)  # 384x11x11
        self.Rrelu6 = nn.LeakyReLU(0.3, inplace=True)
        self.Rconv7 = nn.Conv2d(384, 512, kernel_size=3,
                                stride=1, padding=0)  # 512x9x9
        self.Rrelu7 = nn.LeakyReLU(0.3, inplace=True)
        self.Rconv8 = nn.Conv2d(512, 512, kernel_size=2,
                                stride=1, padding=0)  # 512x8x8
        self.Rrelu8 = nn.LeakyReLU(0.3, inplace=True)
        self.deconv5 = nn.ConvTranspose2d(
            512, 256, kernel_size=4, stride=2, padding=1, bias=True)  # 256x12x12
        self.relu_deconv5 = nn.LeakyReLU(0.3, inplace=True)
        self.conv5_1 = nn.ConvTranspose2d(
            256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu_conv5_1 = nn.LeakyReLU(0.3, inplace=True)
        self.deconv4 = nn.ConvTranspose2d(
            256, 128, kernel_size=4, stride=2, padding=1, bias=True)
        self.relu_deconv4 = nn.LeakyReLU(0.3, inplace=True)
        self.conv4_1 = nn.ConvTranspose2d(
            128, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu_conv4_1 = nn.LeakyReLU(0.3, inplace=True)
        self.deconv3 = nn.ConvTranspose2d(
            128, 128, kernel_size=4, stride=2, padding=1, bias=True)
        self.relu_deconv3 = nn.LeakyReLU(0.3, inplace=True)
        self.conv3_1 = nn.ConvTranspose2d(
            128, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu_conv3_1 = nn.LeakyReLU(0.3, inplace=True)
        self.deconv2 = nn.ConvTranspose2d(
            128, 64, kernel_size=4, stride=2, padding=1, bias=True)
        self.relu_deconv2 = nn.LeakyReLU(0.3, inplace=True)
        self.conv2_1 = nn.Conv2d(
            64, 32, kernel_size=3, stride=1, padding=1, bias=True)  # 32x128x128
        self.relu_conv2_1 = nn.LeakyReLU(0.3, inplace=True)
        self.deconv1 = nn.ConvTranspose2d(
            32, 16, kernel_size=4, stride=2, padding=1, bias=True)  # 3x256x256
        self.relu_deconv1 = nn.LeakyReLU(0.3, inplace=True)
        self.conv1_1 = nn.Conv2d(
            16, 3, kernel_size=3, stride=1, padding=1, bias=True)  # 3x256x256
        self.conv1_1_tanh = nn.Tanh()  # Hyperbolic Tangent (Tanh) function  # 3x256x256
        self.rconv = nn.Sequential(
            self.Rconv6,
            self.Rrelu6,
            self.Rconv7,
            self.Rrelu7,
            self.Rconv8,
            self.Rrelu8,
        ).to(self.__device0)
        self.deconv = nn.Sequential(
            self.deconv5,
            self.relu_deconv5,
            self.conv5_1,
            self.relu_conv5_1,
            self.deconv4,
            self.relu_deconv4,
            self.conv4_1,
            self.relu_conv4_1,
            self.deconv3,
            self.relu_deconv3,
            self.conv3_1,
            self.relu_conv3_1,
            self.deconv2,
            self.relu_deconv2,
            self.conv2_1,
            self.relu_conv2_1,
            self.deconv1,
            self.relu_deconv1,
            self.conv1_1,
            self.conv1_1_tanh,
        ).to(self.__device1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        f = self.rconv(z)
        g = self.deconv(f)
        g = g * 255  # conv1_1_tanh_elt layer's scaling
        return g


class AlexNetNorm2Generator(nn.Module):
    """From caffe norm2 generator of bvlc_reference_caffenet.
    The model trained by DeepSim using ILSVRC dataset.
    Provided by Alexey Dosovitskiy.
    """

    def __init__(
            self, device: Optional[Union[str, Sequence[str]]] = None):
        super(AlexNetNorm2Generator, self).__init__()

        if device is None:
            self.__device0 = 'cpu'
            self.__device1 = 'cpu'
        elif isinstance(device, str):
            self.__device0 = device
            self.__device1 = device
        else:
            self.__device0 = device[0]
            self.__device1 = device[1]
        # input 256x13x13
        self.Rconv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=2)
        self.Rrelu6 = nn.LeakyReLU(0.3, inplace=True)
        self.Rconv7 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.Rrelu7 = nn.LeakyReLU(0.3, inplace=True)
        self.Rconv8 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.Rrelu8 = nn.LeakyReLU(0.3, inplace=True)
        self.deconv4 = nn.ConvTranspose2d(
            128, 128, kernel_size=4, stride=2, padding=1, bias=True)
        self.relu_deconv4 = nn.LeakyReLU(0.3, inplace=True)
        self.conv4_1 = nn.Conv2d(
            128, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu_conv4_1 = nn.LeakyReLU(0.3, inplace=True)
        self.deconv3 = nn.ConvTranspose2d(
            128, 64, kernel_size=4, stride=2, padding=1, bias=True)
        self.relu_deconv3 = nn.LeakyReLU(0.3, inplace=True)
        self.conv3_1 = nn.Conv2d(
            64, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu_conv3_1 = nn.LeakyReLU(0.3, inplace=True)
        self.deconv2 = nn.ConvTranspose2d(
            64, 32, kernel_size=4, stride=2, padding=1, bias=True)
        self.relu_deconv2 = nn.LeakyReLU(0.3, inplace=True)
        self.conv2_1 = nn.Conv2d(
            32, 32, kernel_size=3, stride=1, padding=1, bias=True)  # 32x128x128
        self.relu_conv2_1 = nn.LeakyReLU(0.3, inplace=True)
        self.deconv1 = nn.ConvTranspose2d(
            32, 16, kernel_size=4, stride=2, padding=1, bias=True)  # 3x256x256
        self.conv1_1 = nn.Conv2d(
            16, 3, kernel_size=3, stride=1, padding=1, bias=True)  # 3x256x256
        self.conv1_1_tanh = nn.Tanh()  # Hyperbolic Tangent (Tanh) function  # 3x256x256
        self.rconv = nn.Sequential(
            self.Rconv6,
            self.Rrelu6,
            self.Rconv7,
            self.Rrelu7,
            self.Rconv8,
            self.Rrelu8,
        ).to(self.__device0)
        self.deconv = nn.Sequential(
            self.deconv4,
            self.relu_deconv4,
            self.conv4_1,
            self.relu_conv4_1,
            self.deconv3,
            self.relu_deconv3,
            self.conv3_1,
            self.relu_conv3_1,
            self.deconv2,
            self.relu_deconv2,
            self.conv2_1,
            self.relu_conv2_1,
            self.deconv1,
            self.conv1_1,
            self.conv1_1_tanh,
        ).to(self.__device1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        f = self.rconv(z)
        g = self.deconv(f)
        g = g * 255  # conv1_1_tanh_elt layer's scaling
        return g


class AlexNetNorm1Generator(nn.Module):
    """From caffe norm2 generator of bvlc_reference_caffenet.
    The model trained by DeepSim using ILSVRC dataset.
    Provided by Alexey Dosovitskiy.
    """

    def __init__(
            self, device: Optional[Union[str, Sequence[str]]] = None):
        super(AlexNetNorm1Generator, self).__init__()

        if device is None:
            self.__device0 = 'cpu'
            self.__device1 = 'cpu'
        elif isinstance(device, str):
            self.__device0 = device
            self.__device1 = device
        else:
            self.__device0 = device[0]
            self.__device1 = device[1]
        # input 96x27x27
        self.Rconv6 = nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=2)
        self.Rrelu6 = nn.LeakyReLU(0.3, inplace=True)
        self.Rconv7 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.Rrelu7 = nn.LeakyReLU(0.3, inplace=True)
        self.Rconv8 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.Rrelu8 = nn.LeakyReLU(0.3, inplace=True)
        self.deconv4 = nn.ConvTranspose2d(
            128, 128, kernel_size=4, stride=2, padding=1, bias=True)
        self.relu_deconv4 = nn.LeakyReLU(0.3, inplace=True)
        self.conv4_1 = nn.Conv2d(
            128, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu_conv4_1 = nn.LeakyReLU(0.3, inplace=True)
        self.deconv3 = nn.ConvTranspose2d(
            128, 64, kernel_size=4, stride=2, padding=1, bias=True)
        self.relu_deconv3 = nn.LeakyReLU(0.3, inplace=True)
        self.conv3_1 = nn.Conv2d(
            64, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu_conv3_1 = nn.LeakyReLU(0.3, inplace=True)
        self.deconv2 = nn.ConvTranspose2d(
            64, 32, kernel_size=4, stride=2, padding=1, bias=True)
        self.relu_deconv2 = nn.LeakyReLU(0.3, inplace=True)
        self.conv2_1 = nn.Conv2d(
            32, 32, kernel_size=3, stride=1, padding=1, bias=True)  # 32x128x128
        self.relu_conv2_1 = nn.LeakyReLU(0.3, inplace=True)
        self.deconv1 = nn.ConvTranspose2d(
            32, 16, kernel_size=4, stride=2, padding=1, bias=True)  # 3x256x256
        self.conv1_1 = nn.Conv2d(
            16, 3, kernel_size=3, stride=1, padding=1, bias=True)  # 3x256x256
        self.conv1_1_tanh = nn.Tanh()  # Hyperbolic Tangent (Tanh) function  # 3x256x256
        self.rconv = nn.Sequential(
            self.Rconv6,
            self.Rrelu6,
            self.Rconv7,
            self.Rrelu7,
            self.Rconv8,
            self.Rrelu8,
        ).to(self.__device0)
        self.deconv = nn.Sequential(
            self.deconv4,
            self.relu_deconv4,
            self.conv4_1,
            self.relu_conv4_1,
            self.deconv3,
            self.relu_deconv3,
            self.conv3_1,
            self.relu_conv3_1,
            self.deconv2,
            self.relu_deconv2,
            self.conv2_1,
            self.relu_conv2_1,
            self.deconv1,
            self.conv1_1,
            self.conv1_1_tanh,
        ).to(self.__device1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        f = self.rconv(z)
        g = self.deconv(f)
        g = g * 255  # conv1_1_tanh_elt layer's scaling
        return g
