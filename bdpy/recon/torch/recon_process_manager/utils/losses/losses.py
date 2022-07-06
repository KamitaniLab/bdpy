import torch
from torch import nn
from torch import Tensor
import numpy as np
import torch.nn.functional as F

import kornia.augmentation as K

import warnings


# tricks for loading bdpy files from working directory
if __file__ == '/home/eitoikuta/bdpy_update/bdpy/bdpy/recon/torch/recon_process_manager/utils/losses/losses.py':
    import importlib.util
    dl_torch_spec = importlib.util.spec_from_file_location('dl_torch', "/home/eitoikuta/bdpy_update/bdpy/bdpy/dl/torch/torch.py")
    dl_torch = importlib.util.module_from_spec(dl_torch_spec)
    dl_torch_spec.loader.exec_module(dl_torch)
    FeatureExtractor = dl_torch.FeatureExtractor

    recon_utils_spec = importlib.util.spec_from_file_location('recon_utils', "/home/eitoikuta/bdpy_update/bdpy/bdpy/recon/utils.py")
    recon_utils = importlib.util.module_from_spec(recon_utils_spec)
    recon_utils_spec.loader.exec_module(recon_utils)
    make_feature_masks = recon_utils.make_feature_masks
else:
    from bdpy.dl import FeatureExtractor
    from bdpy.recon.utils import make_feature_masks


### Typical loss based on encoder activations ---------- ###
# helper functions ---------- #
def convert_for_corrmat(feat):
    feat_shape = feat.shape
    feat_flatten = feat.view(feat_shape[0], feat_shape[1], -1)
    return feat_flatten


def cov_torch(m, y=None):
    if y is not None:
        m = torch.cat((m, y), dim=0)
    m_exp = torch.mean(m,dim=2)
    x = m.permute(0,2,1) - m_exp[:, None]
    x = x.permute(0,2,1)

    cov = 1 / (x.size(2) -1) * torch.einsum('bnm,bmk->bnk', x, x.permute(0,2,1))
    denom = 1. /torch.einsum('bn,bm->bnm',x.std(2),x.std(2))
    return cov * denom

# individual loss classes ---------- #
class CorrLoss():
    """
    return correlation coefficient multiplied by -1
    """

    def __init__(self):
        pass

    def __call__(self, act, feat):
        feat_shape = feat.shape
        act_flat = act.view(feat_shape[0], -1)
        feat_flat = feat.view(feat_shape[0], -1)
        x = act_flat
        y = feat_flat
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)
        corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx**2))* torch.sqrt(torch.sum(vy**2)))
        return  - corr

class MSEwithRegularization():
    """
    loss function for hand made
    In pytorch, hand made loss function requires two method, __init__ and forward
    """

    def __init__(self, vid, l_lambda=0.01):
        self.lam = l_lambda
        self.loss_fun = torch.nn.MSELoss(reduction='sum')
        self.vid = vid
    def __call__(self, act, feat):
        vid = self.vid
        loss = self.loss_fun(act, feat)
        img_mask = np.zeros_like(vid.detach().numpy())
        img_mask[-1] = 1
        diff = vid - torch.roll(vid, 1, 0)
        diff_masked = torch.masked_select(diff, torch.FloatTensor(img_mask).bool()).requires_grad_()

        loss += self.lam * torch.sum(diff_masked **2)

        return loss  # / (max_val * res_field_size **2

class FeatCorrLoss():
    """
    Loss function correlation coefficent replaced with MSE
    """
    def __init__(self):
        self.loss_fun = torch.nn.MSELoss()

    def __call__(self, act, feat):
        x = convert_for_corrmat(act)
        y = convert_for_corrmat(feat)
        x_mat = cov_torch(x)
        y_mat = cov_torch(y)

        return self.loss_fun(x_mat, y_mat)

# main loss class used for reconstruction ----------#
class ImageEncoderActivationLoss():
    def __init__(self, model, device, ref_features,
                 preprocess=None, given_as_BGR=False, model_inputs_are_RGB=True,
                 layer_mapping=None, targets=None, sample_axis_info=None,
                 include_model_output=False, model_output_saving_name='model_output',
                 input_image_shape=(224, 224), layer_weights=None,
                 loss_dicts=[{'loss_name': 'MSE', 'weight': 1}],
                 masks=None, channels=None, **args):
        self.model = model
        self.given_as_BGR = given_as_BGR
        self.model_inputs_are_RGB = model_inputs_are_RGB
        self.input_image_shape = input_image_shape
        self.device = device
        self.ref_features = ref_features
        if layer_weights is None:
            w = np.ones(len(ref_features.keys()), dtype=np.float32)
            w = w / w.sum()
            layer_weights = {layer: w[i] for i, layer in enumerate(ref_features.keys())}
        self.preprocess = preprocess
        self.layer_weights = layer_weights
        self.loss_dicts = loss_dicts
        self.layer_mapping = layer_mapping
        self.targets = targets
        self.include_model_output = include_model_output
        # self.sample_axis_info = sample_axis_info
        self.init_loss_funcs()
        self.feature_masks = make_feature_masks(ref_features, masks, channels)

        self.feature_extractor = FeatureExtractor(model, layers=list(layer_mapping.keys()),
                                                  layer_mapping=layer_mapping,
                                                  device=device, detach=False,
                                                  targets=targets, return_final_output=include_model_output,
                                                  final_output_saving_name=model_output_saving_name,
                                                  sample_axis_info=sample_axis_info)

    def init_loss_funcs(self):
        tmp_dicts = []
        for loss_dict in self.loss_dicts:
            if loss_dict['loss_name'] == 'MSE':
                # loss_dict['loss_func'] = torch.nn.MSELoss(reduction='sum')
                if 'params' not in loss_dict: loss_dict['params'] = {}
                loss_dict['loss_func'] = torch.nn.MSELoss(**loss_dict['params'])
            elif loss_dict['loss_name'] == 'CorrLoss':
                loss_dict['loss_func'] = CorrLoss()
            elif loss_dict['loss_func'] == 'MSEwithRegularization':
                # vid is the required parameter
                loss_dict['loss_func'] = MSEwithRegularization(**loss_dict['params'])
            elif loss_dict['loss_func'] == 'FeatCorrLoss':
                loss_dict['loss_func'] = FeatCorrLoss()
            else:
                warnings.warn('invalid loss type {} was ignored'.format(loss_dict['loss_name']))
            tmp_dicts.append(loss_dict)
        self.loss_dicts = tmp_dicts

    def __call__(self, image_batch: Tensor):
        '''
        image_batch needs to be deprocessed beforehand
        '''
        image_batch = F.interpolate(image_batch, tuple(list(self.input_image_shape)[:2]))
        if self.model_inputs_are_RGB and self.given_as_BGR:
            permute = [2, 1, 0]
            image_batch = image_batch[:, permute, :, :]
        if self.preprocess is not None:
            image_batch = self.preprocess(image_batch)
        # TODO: accept activations given in function arguments so that no redundant computation occurs
        current_features = self.feature_extractor(image_batch)
        return self.calc_losses(current_features)

    def calc_losses(self, current_features):
        if not self.include_model_output:
            eval('self.model.{}.zero_grad()'.format(list(self.layer_mapping.values())[-1]))
        else:
            eval('self.model.{}.zero_grad()'.format(list(self.layer_mapping.values())[-2]))
        layers = self.ref_features.keys()
        reversed_layer_list = reversed(layers) if not self.include_model_output else list(reversed(layers))[1:]
        loss = 0
        for loss_dict in self.loss_dicts:
            # TODO: show logs
            tmp_loss = 0
            for j, lay in enumerate(reversed_layer_list):
                act_j = current_features[lay].clone().to(self.device)
                feat_j = torch.tensor(self.ref_features[lay], device=self.device).clone()
                mask_j = torch.FloatTensor(self.feature_masks[lay]).to(self.device)
                weight_j = self.layer_weights[lay]
                masked_act_j = torch.masked_select(act_j, mask_j.bool()).view(act_j.shape)
                masked_feat_j = torch.masked_select(feat_j, mask_j.bool()).view(feat_j.shape)
                loss_j = loss_dict['loss_func'](masked_act_j, masked_feat_j) * weight_j
                tmp_loss += loss_j

            if self.include_model_output:
                tmp_loss = tmp_loss + loss_dict['loss_func'](current_features['model_output'], torch.tensor(self.ref_features['model_output'], device=self.device).clone()) * self.layer_weights['model_output']
            loss += loss_dict['weight'] * tmp_loss
        return loss


### ---------------------------------------------------- ###

### Loss based on CLIP similarity scores --------------- ###
# helper class ---------- #
class ImageAugs():
    def __init__(self, n_cutouts=10,
                 aug_dicts=[{'type': 'ColorJitter'}, {'type': 'RandomAffine'},
                            {'type': 'RandomResizedCrop'}, {'type': 'RandomErasing'}],
                 **args):
        self.n_cutouts = n_cutouts
        self.augs = self.instantiate_augs(aug_dicts)
    def __call__(self, image_batch: torch.Tensor):
        assert image_batch.dim() == 4
        # FIXME: currently only size-one image batch with BCHW is acceptable
        assert image_batch.shape[0] == 1
        image_batch = image_batch.repeat(self.n_cutouts, 1, 1, 1)
        image_batch = self.augs(image_batch)
        return image_batch
    def instantiate_augs(self, aug_dicts):
        augs = []
        # TODO: implement other types of augmentations
        for aug_dict in aug_dicts:
            aug_type = aug_dict['type']
            if 'params' not in aug_dict:
                update_dict = {}
            else:
                update_dict = aug_dict['params']
            if aug_type == 'ColorJitter':
                param_dict = {'brightness':(0,0.5), 'contrast':(0., 2.),
                              'saturation':(0.5, 1.5), 'hue':(-0.1, 0.1), 'p':1.0}
                param_dict.update(update_dict)
                augs.append(K.ColorJitter(**param_dict))
            elif aug_type == 'RandomAffine':
                param_dict = {'degrees':(-15, 15), 'translate':(0.125, 0.125),
                              'p':1.0, 'padding_mode':'zeros', 'keepdim':True}
                param_dict.update(update_dict)
                augs.append(K.RandomAffine(**param_dict))
            elif aug_type == 'RandomResizedCrop':
                param_dict = {'size':(224,224), 'scale':(0.8,1.2),
                              'ratio':(1,1), 'cropping_mode':'resample', 'p':1.0}
                param_dict.update(update_dict)
                augs.append(K.RandomResizedCrop(**param_dict))
            elif aug_type == 'RandomErasing':
                param_dict = {'scale':(.25, .25), 'ratio':(.3, 1/.3), 'p': 1.0}
                param_dict.update(update_dict)
                augs.append(K.RandomErasing(**param_dict))
            else:
                warnings.warn('Unknown augmentaion type {}'.format(aug_type))
        return nn.Sequential(*augs)

# main loss class used for reconstruction ----------#
class CLIPLoss():
    def __init__(self, clip_model, ref_features, device, given_as_BGR=False,
                 image_augmentation=False, image_aug=None):
        self.clip_model = clip_model.to(device)
        self.ref_features = torch.tensor(ref_features).to(device).float() # (1, 512)-shaped feature. maybe (n, 512) is also acceptable. Also, image feature can be used
        self.device = device
        self.given_as_BGR = given_as_BGR
        self.image_augmentation = image_augmentation
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(device)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(device)
        if image_augmentation:
            assert image_aug is not None
            self.image_aug = image_aug

    def __call__(self, image_batch: torch.Tensor):
        '''
        image_batch is expected `not` to be preprocessed/normalized
        '''
        image_encoder = self.clip_model.visual.float()
        if self.given_as_BGR:
            permute = [2, 1, 0]
            image_batch = image_batch[:, permute, :, :]
        image_batch = F.interpolate(image_batch, (224, 224))
        image_batch = image_batch / 255.
        if self.image_augmentation:
            image_batch = self.image_aug(image_batch)
        image_batch.sub_(self.mean[None, :, None, None]).div_(self.std[None, :, None, None])
        image_features = image_encoder(image_batch)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        ref_features = self.ref_features / self.ref_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.clip_model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ ref_features.T

        # to maximize similarity, -1 * score must be minimized
        loss = (-1.) * logits_per_image.mean(dim=0, keepdim=False)

        assert loss.shape == torch.Size([1])
        return loss[0]
### ---------------------------------------------------- ###