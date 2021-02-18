#!/usr/bin/python
'#-*- coding: utf-8 -*- '

import numpy as np
from .metrics import *


"""
Evaluator class
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


class BaseEvaluator():
    """ Abstract 'evaluator class/

    """
    def __init__(self, metric, preprocs=None, **opts):
        """ Initialize evaluator class

        Args:
            metric (string or function): specify the metric. 
                If variable is string, this class uses pre-provided function.
                If not (e.g, varable is function), use this funciton to evaluate.
            preprocs (list): preprocess list.  Each element should contains 
                            preprocess function before evaluation  
            opt_dict (dict): optional dict for calculate metrics
                             (this will be not well written)
        """
        if type(metric) == str:
            self.metric = metric_dict[metric]
        else:
            self.metric = metric
        self.preprocs = preprocs
        self.opts = opts

        # select option if opts (dict) is empty
        if len(self.opts) == 0 and metric in opts_dict:
            self.opts = opts_dict[metric]

    def validate(self, x, y):
        
        pass

    def calc_metric(self, x, y):
        """[summary]

        Args:
            x (np.array): Target variable (e.g. decoded, recon feature).
                          The first dimension should be samples
            y (np.array): Source variable (e.g. true feature)
                          The array should be the same as x

        Returns:
            np.array (batch_size, *shape):
                The first element is the batch_size of the inputs
                For the second each elements is the metric calculated.
        """
        if type(x) == list:
            x = np.array(x)
            x = np.array(y)

        # check for mismatch between true and recon
        try:
            y.shape == y.shape
        except as e:
            raise('The shape is not matched between inputs')
        # return the caluculated values for every sample
        calculated_list = self.metric(x, y, **self.opts)
        return calculated_list

    def __call__(self, x, y):

        self.validate(x, y)

        if self.preprocs is not None:
            for preproc in preporcs:
                x = preproc(x)
                y = preproc(y)
        calculated_list = self.calc_metric(x, y)
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

    def __init__(self, img_metric):
        super().__init__(img_metric)

    def validate(self, recon_img, true_img):
        # check image shape (bs, h, w, ch)
        if recon_img.dim != 4:
            raise('The shape is not matched for image')

    def __call__(self, recon_img, true_img):
        """[summary]

        Args:
            recon_img (np.array): [description]
            true_img(np.array): [description]

        Returns:
            np.array (batch_size, *shape):
                The first element is the batch_size of the inputs.
                For the second each elements is the metric calculated.
        """

        
        return super().calc_metric(recon_img, true_img)


class FeatureEvaluator(StimEvaluater):
    """ calculate specified metric between two input DNN features

    Args:
        StimEvaluater ([object]): Abstract class

    Returns:
        np.array (batch_size, *shape):
            The first element is the batch_size of the inputs.
            For the second each elements is the metric calculated.
    """
    def __init__(self, img_metric):
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

    def validate(self, recon_vid, true_vid):
        # check image shape (bs, fr, h, w, ch)
        if recon_vid.dim != 5:
            raise('The shape is not matched for video')


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
        return super().__call__(recon_vid, true_vid)
