import yaml
import warnings

import copy

from scipy.io import loadmat
import numpy as np
import torch


from losses import ImageEncoderActivationLoss, CLIPLoss, ImageAugs
from feature_scaling_utils import normalize_features

__all__ = ['loss_dicts_to_loss_instances']

### General helper functions ---------- ###
def is_in_and_True(key, dictionary):
    return key in dictionary and dictionary[key]


### Helper functions for creating typical loss instances based on encoder activations ---------- ###
def calc_layer_weight_from_norm(features_dict, layers):
    feat_norms = np.array([np.linalg.norm(features_dict[layer])
                           for layer in layers],
                           dtype='float32')
    # Weight of each layer in the total loss function
    # Use the inverse of the squared norm of the DNN features as the
    # weight for each layer
    weights = 1. / (feat_norms ** 2)

    # Normalise the weights such that the sum of the weights = 1
    weights = weights / weights.sum()
    layer_weights = dict(zip(layers, weights))
    return layer_weights

def create_ImageEncoderActivationLoss_instance(loss_dict, model_info, features,
                                               hooked_module_names, module_saving_names,
                                               image_label, subject=None, roi=None,
                                               generator_BGR=True, device='cpu'):
    # FIXME: channels and masks are not used now
    num_layers = len(hooked_module_names)
    if 'num_layers_to_use' in loss_dict:
        num_layers_to_use = loss_dict['num_layers_to_use']
    else:
        num_layers_to_use = num_layers
    ref_feature_info = loss_dict['ref_feature_info']
    include_model_output = num_layers_to_use == num_layers and ref_feature_info['include_model_output']
    hooked_module_names = hooked_module_names[:num_layers_to_use]
    module_saving_names = module_saving_names[:num_layers_to_use]
    # layer_mapping = dict(zip(hooked_module_names, module_saving_names))
    layer_mapping = dict(zip(module_saving_names, hooked_module_names))
    module_loading_names = []
    for module_saving_name in module_saving_names:
        if isinstance(module_saving_name, str):
            module_loading_names.append(module_saving_name)
        else:
            module_loading_names.extend(module_saving_name)

    sample_axis_list = ref_feature_info['sample_axis_list'][:num_layers_to_use]

    if ref_feature_info['decoded']:
        ref_features = {layer: features.get(layer=layer, subject=subject, roi=roi, image=image_label)
                        for layer in module_loading_names}
    else:
        ref_features = {layer: features.get(layer=layer, image=image_label)
                        for layer in module_loading_names}

    # normalize features
    ref_feature_info = loss_dict['ref_feature_info']
    if is_in_and_True('normalize_feature', ref_feature_info):
        assert 'normalization_settings' in ref_feature_info
        normalization_settings = ref_feature_info['normalization_settings']
        for factor_settings in normalization_settings.values():
            if factor_settings['values'] not in ['mean', 'std'] and not isinstance(factor_settings['values'], dict):
                factor_settings['values'] = loadmat(factor_settings['values'])
        ref_features = normalize_features(ref_features, normalization_settings)

    # calculate layer weights
    if is_in_and_True('uniform_layer_weight', loss_dict):
        layer_weights = None
    else:
        layer_weights = calc_layer_weight_from_norm(ref_features, module_loading_names)
        print(layer_weights)

    # load model
    encoder_info = loss_dict['encoder_info']
    if is_in_and_True('image_encoder_only', encoder_info) and not is_in_and_True('image_encoder_only', model_info):
        model = model_info['model_instance'].visual.float()
    elif is_in_and_True('text_encoder_only', encoder_info) and not is_in_and_True('text_encoder_only', model_info):
        model = model_info['model_instance'].transformer.float()
    else:
        model = model_info['model_instance']
    preprocess = model_info['preprocess']

    loss_instance = ImageEncoderActivationLoss(model, device, ref_features,
                                               preprocess=preprocess,
                                               given_as_BGR=generator_BGR,
                                               layer_weights=layer_weights,
                                               **loss_dict['encoder_info'],
                                               loss_dicts=loss_dict['loss_dicts'],
                                               layer_mapping=layer_mapping,
                                               sample_axis_list=sample_axis_list,
                                               include_model_output=include_model_output)
    return loss_instance


### Helper functions for creating loss instances based on CLIP scores ---------- ###
class StartFromMiddle:
    '''
    start from the activations in the middle layer to get the final output
    - nothing will be returned
    - len(outputs)=len(layers) and outputs[i].shape=(Ci, Hi, Wi)
    '''
    def __init__(self, activation_input, device=None):
        self.activation_input = activation_input
        self.device = device

    def __call__(self, module, module_in, module_out):
        if isinstance(module_out, tuple):
            if isinstance(self.activation_input, tuple) or isinstance(self.activation_input, list):
                assert len(module_out) == len(self.activation_input)
            else:
                assert len(module_out) == 1
                self.activation_input = [self.activation_input]
            for module_out_i, activation in zip(module_out, self.activation_input):
                input_tensor = torch.tensor(activation)
                if module_out_i.shape != input_tensor.shape:
                    target_shape = module_out_i.shape
                    input_shape = input_tensor.shape
                    assert set(target_shape) == set(input_shape), print('the shapes are too different to automatically broadcast')
                    input_shape_array = np.array(input_shape)
                    indices = [np.argmax(input_shape_array == dim) for dim in target_shape]
                    input_tensor = input_tensor.permute(*indices)
                module_out_i.data = input_tensor.to(self.device)
        else:
            input_tensor = torch.tensor(self.activation_input)
            if module_out.shape != input_tensor.shape:
                target_shape = module_out.shape
                input_shape = input_tensor.shape
                assert set(target_shape) == set(input_shape), print('the shapes are too different to automatically broadcast')
                input_shape_array = np.array(input_shape)
                indices = [np.argmax(input_shape_array == dim) for dim in target_shape]
                input_tensor = input_tensor.permute(*indices)
            module_out.data = input_tensor.to(self.device)

    def clear(self):
        self.activation_input = None

def middle2final_activation(dummy_input, text_feature, hooked_module_name, model, device, clip_model):
    net = copy.deepcopy(model)
    net.to(device)
    start_from_middle = StartFromMiddle(text_feature, device)
    exec('net.'+ hooked_module_name + ".register_forward_hook(start_from_middle)")
    output = net(dummy_input.to(device))
    print(output.shape)
    del net
    net = copy.deepcopy(clip_model)
    net.to(device)
    output = output.permute(1, 0, 2)  # LND -> NLD
    output = net.ln_final(output).type(net.dtype)

    # FIXME: need the information of text length
    # output = output[torch.arange(output.shape[0]), text.argmax(dim=-1)] @ net.text_projection
    output = output[torch.arange(output.shape[0]), 12] @ net.text_projection.float()

    # print(output.shape)
    return output

def refine_text_features(text_features, module_saving_names, hooked_module_names, decoding_accuracies, model, device, dummy_input, clip_model):
    outputs = []
    decoding_accuracy_array = []
    for module_saving_name, hooked_module_name in zip(module_saving_names, hooked_module_names):
        if module_saving_name == 'output_layer': continue
        if isinstance(module_saving_name, tuple) or isinstance(module_saving_name, list):
            module_saving_name = [subname for subname in module_saving_name if subname in text_features.keys()]
            if len(module_saving_name) == 0:
                continue
            input_text_features = [text_features[module_saving_name_i] for module_saving_name_i in module_saving_name]
            outputs.append(middle2final_activation(dummy_input, input_text_features, hooked_module_name, model, device, clip_model).detach().cpu().numpy())
            tmp_size = 0
            tmp_acc = 0
            for module_saving_name_i in module_saving_name:
                tmp_acc += decoding_accuracies[module_saving_name_i]
                tmp_size += 1
            decoding_accuracy_array.append(tmp_acc / tmp_size)
        else:
            if module_saving_name not in text_features.keys():
                continue
            c_o = middle2final_activation(dummy_input, text_features[module_saving_name], hooked_module_name, model, device, clip_model).detach().cpu().numpy()
            outputs.append(c_o)
            decoding_accuracy_array.append(decoding_accuracies[module_saving_name])
    if 'output_layer' in text_features.keys():
        outputs.append(text_features['output_layer'])
        decoding_accuracy_array.append(decoding_accuracies['output_layer'])
    outputs = np.array(outputs)
    decoding_accuracy = np.array(decoding_accuracy_array)
    weight = np.exp(decoding_accuracy) / np.sum(np.exp(decoding_accuracy))
    broadcast_shape = [len(decoding_accuracy)]
    broadcast_shape.extend([1] * (outputs.ndim-1))
    return np.sum(outputs * weight.reshape(broadcast_shape), axis=0)

def create_CLIPLoss_instance(loss_dict, model_info, features,
                             hooked_module_names, module_saving_names,
                             image_label, subject=None, roi=None,
                             generator_BGR=True, device='cpu'):
    module_loading_names = []
    for module_saving_name in module_saving_names:
        if isinstance(module_saving_name, str):
            module_loading_names.append(module_saving_name)
        else:
            module_loading_names.extend(module_saving_name)

    ref_feature_info = loss_dict['ref_feature_info']
    text_features_yaml_file = ref_feature_info['text_features_yaml_file']
    with open(text_features_yaml_file) as file:
        text_feature_ROI_selection = yaml.safe_load(file)

    if ref_feature_info['decoded']:
        ref_features = {layer: features.get(layer=layer, subject=subject, roi=text_feature_ROI_selection[layer]['roi'], image=image_label)
                        for layer in module_loading_names if not text_feature_ROI_selection[layer]['roi'] is None}
        ref_features['output_layer'] = features.get(layer='output_layer', subject=subject, roi='whole_VC', image=image_label)
        print('ref_features["output_layer"]', ref_features['output_layer'].shape)
    else:
        # FIXME: dirty solution for mismatching image labels in true features
        if int(image_label) > 2185:
            tmp_image_label = str(int(image_label) - 2185)
        else:
            tmp_image_label = image_label
        ref_features = {layer: features.get(layer='', image=tmp_image_label)
                        for layer in module_loading_names if not text_feature_ROI_selection[layer]['roi'] is None}
        ref_features['output_layer'] = features.get(layer='', image=tmp_image_label)
    mean_accuracy_dict = {layer: text_feature_ROI_selection[layer]['accuracy']
                          for layer in module_loading_names if not text_feature_ROI_selection[layer]['roi'] is None}
    # FIXME: use actutal mean accuracy
    mean_accuracy_dict['output_layer'] = 1

    clip_model = model_info['model_instance']
    dummy_input = torch.zeros(tuple(loss_dict['encoder_info']['text_input_shape'])).to(torch.float32)
    ref_features = refine_text_features(ref_features,
                                        module_saving_names,
                                        hooked_module_names,
                                        mean_accuracy_dict,
                                        clip_model.transformer.float(),
                                        device,
                                        dummy_input,
                                        clip_model)

    image_augmentation_info = loss_dict['image_augmentation']
    perform_image_augmentation = image_augmentation_info['perform_augmentation']
    image_augs = None
    if perform_image_augmentation:
        image_augs = ImageAugs(**image_augmentation_info)
    loss_instance = CLIPLoss(clip_model, ref_features, device,
                             given_as_BGR=generator_BGR,
                             image_augmentation=perform_image_augmentation,
                             image_aug=image_augs)
    return loss_instance


### Helper function for creating one loss instance from one loss_dict  ---------- ###
def loss_dict_to_loss_instance(loss_dict, device='cpu', **args):
    # FIXME: if same model is required by multiple losses, do not load it agein
    loss_type = loss_dict['loss_type']
    if loss_type == 'ImageEncoderActivationLoss':
        loss_instance = create_ImageEncoderActivationLoss_instance(loss_dict, **args, device=device)
    elif loss_type == 'CLIPLoss':
        loss_instance = create_CLIPLoss_instance(loss_dict, **args, device=device)
    else:
        assert False, print('Unknown loss type is specified: {}'.format(loss_type))
    return loss_instance


### Helper function for creating various loss instances from list of loss_dicts  ---------- ###
def loss_dicts_to_loss_instances(loss_dicts, models_dict, features_dicts,
                                 image_label, subject='', rois_list=[],
                                 generator_BGR=True, device='cpu'):
    loss_func_dicts = []
    rois_list_index = 0
    for loss_dict in loss_dicts:
        loss_func_dict = {}
        options = {'image_label': image_label}
        if 'ref_feature_info' in loss_dict:
            options['subject'] = subject
            ref_feature_info = loss_dict['ref_feature_info']
            if ref_feature_info['decoded']:
                options['roi'] = rois_list[rois_list_index]
                rois_list_index += 1
                features = None
                if 'decoding_conf_path' in ref_feature_info:
                    decoding_conf_path = ref_feature_info['decoding_conf_path']
                    for features_dict in features_dicts:
                        if 'decoding_conf_path' in features_dict and features_dict['decoding_conf_path'] == decoding_conf_path:
                            features = features_dict['features_instance']
                            options['hooked_module_names'] = features_dict['hooked_module_names']
                            options['module_saving_names'] = features_dict['module_saving_names']
                            # options['layer_mapping'] = features_dicts['layer_mapping']
                else:
                    features_dir = ref_feature_info['features_dir']
                    for features_dict in features_dicts:
                        if features_dict['features_dir'] == features_dir:
                            features = features_dict['features_instance']
                            options['hooked_module_names'] = features_dict['hooked_module_names']
                            options['module_saving_names'] = features_dict['module_saving_names']
                            # options['layer_mapping'] = features_dicts['layer_mapping']
            else:
                features_dir = ref_feature_info['features_dir']
                for features_dict in features_dicts:
                    if features_dict['features_dir'] == features_dir:
                        features = features_dict['features_instance']
                        options['hooked_module_names'] = features_dict['hooked_module_names']
                        options['module_saving_names'] = features_dict['module_saving_names']
                        # options['layer_mapping'] = features_dicts['layer_mapping']
                # get features corresponding to image_label
            assert features is not None
            options['features'] = features

            options['model_info'] = models_dict[loss_dict['encoder_info']['network_name']]
            loss_instance = loss_dict_to_loss_instance(loss_dict, **options, generator_BGR=generator_BGR, device=device)
            loss_func_dict['loss_type'] = loss_dict['loss_type']
            loss_func_dict['loss_func'] = loss_instance
            loss_func_dict['weight'] = loss_dict['weight']
            loss_func_dicts.append(loss_func_dict)
        else:
            warnings.warn('loss {} is ignored'.format(loss_dict['loss_type']))

    return loss_func_dicts