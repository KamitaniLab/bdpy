#!/usr/bin/python
'#-*- coding: utf-8 -*- '
import numpy as np
from tqdm import tqdm
from .metric import *
from .miscs.fastMotionEnergy.fastMotionEnergy import fastMotionEnergyModel
from .miscs.fastMotionEnergy.preprocWavelets_grid_GetMetaParams import preprocWavelets_gird_GetMetaParams

"""
StimEval class
    python package for evaluating
    the reconstructed/decoded image[video, other domin] with targets.
"""
metric_dict = {
    'profile correlation': corr,
    'pattern correlation': corr,
    'squared error': squarederror,
    'pairwise identification': pairwise_identification,
}

opts_dict = {'pattern correlation': {'var': 'row'}}


def rgb2gray(video):
    return video[0]*0.2989 + video[1] * 0.5870 + video[2] * 0.1140


def profile_correlation_ME(x, y, **opts):

    mec = MotionEnergyCalculator(metric='profile correlation', **opts)
    val = mec(x, y)
    return val


class MotionEnergyCalculator():

    def __init__(self, metric='profile correlation', **opts):

        self.metric = metric_dict[metric]
        if 'params' in opts:
            self.params = opts['params']
        else:
            self.params = preprocWavelets_gird_GetMetaParams(2)

        if 'img_mean_gray' in opts:
            self.img_mean_gray = opts['img_mean_gray']
        else:
            img_mean = [103.939, 116.779, 123.68]  # image net RGB
            self.img_mean_gray = rgb2gray(img_mean)

    def calcME(self, x, y):
        # Build model
        print('Build Motion Enegy Model...')
        bs, fr, h, w, ch = x.shape
        model = fastMotionEnergyModel((h, w, fr), self.params)

        # Convert gray
        print('Convert RGB to gray....')
        x_gray = self._convert_gray(x)
        y_gray = self._convert_gray(y)

        # Calculate M
        print('Calculate Motion Energy.....')
        x_ME = self._calc_ME(model, x_gray)
        y_ME = self._calc_ME(model, y_gray)
        print('Evaluation ......')
        val = self.metric(x_ME, y_ME)

        return val

    def __call__(self, x, y):

        return self.calcME(x, y)

    def _convert_gray(sefl, x):
        """Conbert color video into gray video with transpose.

        Args:
            x (np.array): video (bs, fr , height, width, ch)

        Return:
            np.array: gray_video for calculate ME (bs, height, widht, fr)
        """

        # transpose
        x_trans = x.transpose(0, 4, 2, 3, 1)

        x_gray = [rgb2gray(xi) for xi in x_trans]

        return np.array(x_gray)

    def _calc_ME(self, model, x):
        """[summary]

        Args:
            model (object): motionenergy model
            x (np.array): gray_video (bs, height, width, fr)
        """

        me = [model.calculate(xi) for xi in tqdm(x)]

        return np.array(me)


if __name__ == '__main__':
    from miscs.fastMotionEnergy.preprocWavelets_grid_GetMetaParams import preprocWavelets_gird_GetMetaParams

    params = preprocWavelets_gird_GetMetaParams(2)
    me = fastMotionEnergyModel((224, 224, 16), params)

    rand_vid = np.random.rand(1, 16, 224, 224, 3)
    # convert gray
    rand_gray_vid = np.mean([rand_vid[..., 0], rand_vid[..., 1],
                             rand_vid[..., 2], 0])
    rand_gray_vid_transpose = rand_gray_vid[0].transpose(1, 2, 0)
    print('start calculating')
    output = me.calculate(rand_gray_vid_transpose)

    print(np.mean(output))
