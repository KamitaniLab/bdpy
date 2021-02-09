#!/usr/bin/python
'#-*- coding: utf-8 -*- '

import numpy as np
from .metric import *
from .metric_ME import *

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
    'ME profile correlation': profile_correlation_ME
}

opts_dict = {'pattern correlation': {'var': 'row'}}


class StimEvaluater():
    """ Abstract 'stim evaluator class common for all types of stim evaluators/

    """
    def __init__(self, metric, **opts):
        """ Initialize evaluator class

        Args:
            metric (string): specify the metric
            opt_dict (dict): optional dict for calculate metrics
                             (this will be not well written)
        """
        self.metric = metric_dict[metric]
        self.opts = opts
        # select option if opts (dict) is empty
        if len(self.opts) == 0 and metric in opts_dict:
            self.opts = opts_dict[metric]

    def calc_metric(self, recon_stim, true_stim):
        """[summary]

        Args:
            recon_stim (np.array): [description]
            true_stim (np.array): [description]

        Returns:
            np.array (batch_size, *shape):
                The first element is the batch_size of the inputs
                For the second each elements is the metric calculated.
        """
        if type(true_stim) == list:
            true_stim = np.array(true_stim)
            recon_stim = np.array(recon_stim)

        # check for mismatch between true and recon
        try:
            true_stim.shape == recon_stim.shape
        except as e:
            raise('The shape is not matched between inputs')
        # return the caluculated values for every sample
        calculated_list = self.metric(recon_stim, rue_stim, **self.opts)
        return calculated_list

    def __call__(self, recon_stim, true_stim):
        calculated_list = self.calc_metric(recon_stim, true_stim)
        return calculated_list


class ImageEvaluator(StimEvaluater):
    """ calculate specified metric between two input images

    Args:
        StimEvaluater ([object]): Abstract class
    Returns:
        np.array (batch_size, height, width, channel):
            The first element is the batch_size of the inputs.
            For the second each elements is the metric calculated.
    """

    def __init__(self, img_metric='pixel correlation'):
        super().__init__(img_metric)

    def __call__(self, recon_img, true_img):
        """[summary]

        Args:
            recon_stim (np.array): [description]
            true_stim (np.array): [description]

        Returns:
            np.array (batch_size, *shape):
                The first element is the batch_size of the inputs.
                For the second each elements is the metric calculated.
        """

        # check image shape (bs, h, w, ch)
        if len(recon_img[0].shape) != 3:
            raise('The shape is not matched for image')
        return super().calc_metric(recon_img, true_img)


class FeatEvaluator(StimEvaluater):
    """ calculate specified metric between two input DNN features

    Args:
        StimEvaluater ([object]): Abstract class

    Returns:
        np.array (batch_size, *shape):
            The first element is the batch_size of the inputs.
            For the second each elements is the metric calculated.
    """
    def __init__(self, img_metric='pixel correlation'):
        super().__init__(img_metric)

    def __call__(self, decoded_feat, true_feat):
        return super().__call__(decoded_feat, true_feat)


class VideoEvaluator(StimEvaluater):
    """ Calculate specified metric between two input videos

    Args:
        StimEvaluater ([object]): Abstract class

    Returns:
        np.array (batch_size, *shape):
            The first element is the batch_size of the inputs
            For the second each elements is the metric calculated.
    """

    def __init__(self, vid_metric='pixel correlation'):
        super().__init__(vid_metric)

    def __call__(self, recon_vid, true_vid):
        """[summary]

        Args:
            recon_vid (np.array): [description]
            true_vid (np.array): [description]

        Returns:
            np.array (batch_size, *shape):
                The first element is the batch_size of the inputs.
                For the second each elements is the metric calculated.
        """
        # check image shape (bs, fr, h, w, ch)
        if len(true_vid[0].shape) != 4:
            raise('The shape is not matched for video')
        return super().__call__(recon_vid, true_vid)
