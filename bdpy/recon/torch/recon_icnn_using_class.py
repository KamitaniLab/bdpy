import sys
import argparse
from pathlib import Path

import yaml
# import pickle
from natsort import os_sorted

from itertools import product

import gc
import torch

# Import from my own scripts
from recon_process_manager import loss_dicts_to_loss_instances, create_ReconProcess_from_conf, create_model_instance

# tricks for loading bdpy files from working directory
if __file__ == '/home/eitoikuta/bdpy_update/bdpy/bdpy/recon/torch/recon_icnn_using_class.py':
    import importlib.util
    datastore_spec = importlib.util.spec_from_file_location('datastore', "/home/eitoikuta/bdpy_update/bdpy/bdpy/dataform/datastore.py")
    datastore = importlib.util.module_from_spec(datastore_spec)
    datastore_spec.loader.exec_module(datastore)
    DecodedFeatures = datastore.DecodedFeatures
    GeneralFeatures = datastore.GeneralFeatures

    models_spec = importlib.util.spec_from_file_location('models', "/home/eitoikuta/bdpy_update/bdpy/bdpy/dl/torch/models.py")
    models = importlib.util.module_from_spec(models_spec)
    models_spec.loader.exec_module(models)
    layer_map = models.layer_map
else:
    # import bdpy
    from bdpy.dataform import DecodedFeatures, GeneralFeatures
    from bdpy.dl.torch import layer_map


def is_in_and_True(key, dictionary):
    return key in dictionary and dictionary[key]

def get_layer_mapping(conf, network_name):
    input_type = conf['input_type']
    if input_type == 'image':
        module_saving_names_key = 'module_saving_names'
        hooked_module_names_key = 'hooked_module_names'
    elif input_type == 'text':
        module_saving_names_key = 'text_features_module_saving_names'
        hooked_module_names_key = 'text_features_hooked_module_names'
    else:
        assert False, print('invalid input type for `get_layer_mapping`: {}'.format(input_type))
    # FIXME: current implementation does not allow using different layers for different loss instances
    if module_saving_names_key in conf.keys() and hooked_module_names_key not in conf.keys():
        layer_mapping = layer_map(network_name)
        module_saving_names = conf[module_saving_names_key]
        hooked_module_names = [layer_mapping[module_saving_name] for module_saving_name in module_saving_names]
    elif module_saving_names_key not in conf.keys():
        layer_mapping = layer_map(network_name)
        module_saving_names = list(layer_mapping.keys())
        hooked_module_names = list(layer_mapping.values())
    else:
        hooked_module_names = conf[hooked_module_names_key]
        module_saving_names = conf[module_saving_names_key]
        layer_mapping = dict(zip(module_saving_names, hooked_module_names))
    return hooked_module_names, module_saving_names, layer_mapping

def get_required_models(recon_conf):
    models_dict = {}
    device = recon_conf['general_settings']['device']

    # create generator instance if needed
    optimization_settings = recon_conf['optimization_settings']
    generator_settings = optimization_settings['generator']
    if generator_settings['use_generator']:
        model_name = generator_settings['network_name']
        model, preprocess = create_model_instance(model_name, device='cpu', **generator_settings)
        models_dict[model_name] = {'model_instance': model, 'preprocess': preprocess}

    required_models = {}
    loss_dicts = recon_conf['loss_settings']
    for loss_dict in loss_dicts:
        # FIXME: losses that do not require any model are not considered
        encoder_info = loss_dict['encoder_info']
        network_name = encoder_info['network_name']
        if network_name not in required_models:
            tmp_info = {}
            # FIXME: accept other CLIP models
            if network_name == 'CLIP_ViT-B_32':
                if is_in_and_True('image_encoder_only', encoder_info):
                    tmp_info['image_encoder_only'] = True
                elif is_in_and_True('text_encoder_only', encoder_info):
                    tmp_info['text_encoder_only'] = True
            if 'params_file' in encoder_info:
                tmp_info['params_file'] = encoder_info['params_file']
            required_models[network_name] = tmp_info
        else:
            if network_name == 'CLIP_ViT-B_32':
                if not is_in_and_True('image_encoder_only', encoder_info):
                    required_models[network_name]['image_encoder_only'] = False
                elif not is_in_and_True('text_encoder_only', encoder_info):
                    required_models[network_name]['text_encoder_only'] = False
            if 'params_file' in encoder_info:
                assert 'params_file' in required_models[network_name]
                assert tmp_info['params_file'] == required_models[network_name]['params_file']
    for network_name, model_info in required_models.items():
        model_info['model_instance'], model_info['preprocess'] = create_model_instance(network_name, device=device, **model_info)
        models_dict[network_name] = model_info
    return models_dict

def get_required_features(loss_dicts):
    features_dicts = []
    for loss_dict in loss_dicts:
        if 'ref_feature_info' not in loss_dict:
            continue
        tmp_dict = {}
        already_loaded = False
        network = loss_dict['encoder_info']['network_name']
        ref_feature_info = loss_dict['ref_feature_info']
        if ref_feature_info['decoded']:
            if 'decoding_conf' in ref_feature_info:
                decoding_conf_path = ref_feature_info['decoding_conf']
                for features_dict in features_dicts:
                    if 'decoding_conf_path' in features_dict and features_dict['decoding_conf_path'] == decoding_conf_path:
                        already_loaded = True
                        break
                if already_loaded: continue
                with open(decoding_conf_path, 'r') as f:
                    dec_conf = yaml.safe_load(f)
                features_dir = str(Path(dec_conf['decoded feature dir']) / dec_conf['analysis name'] / 'decoded_features' / network)
                # features_dir = os.path.join(dec_conf['decoded feature dir'], dec_conf['analysis name'], 'decoded_features', network)
                for features_dict in features_dicts:
                    if features_dict['features_dir'] == features_dir:
                        features_dict['decoding_conf_path'] = decoding_conf_path
                        features_dict['decoding_conf'] = dec_conf
                        already_loaded = True
                        break
                if already_loaded: continue
                tmp_dict['decoding_conf_path'] = decoding_conf_path
                tmp_dict['decoding_conf'] = dec_conf
            else:
                assert 'features_dir' in ref_feature_info
                features_dir = ref_feature_info['features_dir']
                for features_dict in features_dicts:
                    if features_dict['features_dir'] == features_dir:
                        already_loaded = True
                        break
                if already_loaded: continue
            # DecodedFeatures
            features_instance = DecodedFeatures(features_dir, squeeze=False)
        else:
            assert 'features_dir' in ref_feature_info
            features_dir = ref_feature_info['features_dir']
            for features_dict in features_dicts:
                if features_dict['features_dir'] == features_dir:
                    already_loaded = True
                    continue
            if already_loaded: break
            dirs_pattern = []
            if 'dirs_pattern' in ref_feature_info and ref_feature_info['dirs_pattern'] is not None:
                dirs_pattern = ref_feature_info['dirs_pattern']
            # Features
            features_instance = GeneralFeatures(features_dir, dirs_pattern=dirs_pattern)
        hooked_module_names, module_saving_names, _ = get_layer_mapping(ref_feature_info, network)
        tmp_dict['hooked_module_names'] = hooked_module_names
        tmp_dict['module_saving_names'] = module_saving_names

        tmp_dict['features_dir'] = features_dir
        tmp_dict['features_instance'] = features_instance
        features_dicts.append(tmp_dict)
    return features_dicts

def list_up_conditions(loss_dicts):
    subject_set = None
    roi_lists = []
    for loss_dict in loss_dicts:
        if 'ref_feature_info' not in loss_dict:
            continue
        ref_feature_info = loss_dict['ref_feature_info']
        if not ref_feature_info['decoded']:
            continue

        if subject_set is None:
            subject_set = set(ref_feature_info['subjects'])
        else:
            subject_set = subject_set & set(ref_feature_info['subjects'])

        roi_lists.append(ref_feature_info['rois'])

    return subject_set, roi_lists

def get_image_labels(features_dict):
    ref_features = features_dict['features_instance']
    image_labels = ref_features.labels
    return image_labels

def label_out_of_bound(image_label, label_lowerbound, label_upperbound, label_list):
    if label_list is not None:
        if image_label not in label_list:
            return True
    if label_lowerbound is not None:
        original_list = [label_lowerbound, image_label]
        sorted_list = os_sorted(original_list)
        if original_list != sorted_list:
            return True
    if label_upperbound is not None:
        original_list = [image_label, label_upperbound]
        sorted_list = os_sorted(original_list)
        if original_list != sorted_list:
            return True
    return False

def run_reconstruction(recon_conf):
    '''
    load features, create loss and reconprocess instances, and run reconstruction
    '''
    generator_settings = recon_conf['optimization_settings']['generator']
    generator_BGR = generator_settings['use_generator'] and generator_settings['output_BGR']
    recon_conf['optimization_settings']['generator']['generator_output_BGR'] = generator_BGR
    models_dict = get_required_models(recon_conf)
    features_dicts = get_required_features(recon_conf['loss_settings'])
    image_labels = os_sorted(get_image_labels(features_dicts[0]))
    print(image_labels)
    label_lowerbound = label_upperbound = label_list = None
    if 'label_lowerbound' in recon_conf['general_settings']:
        label_lowerbound = recon_conf['general_settings']['label_lowerbound']
    if 'label_upperbound' in recon_conf['general_settings']:
        label_upperbound = recon_conf['general_settings']['label_upperbound']
    if 'labels_list' in recon_conf['general_settings']:
        label_list = recon_conf['general_settings']['label_list']
    subject_set, roi_lists = list_up_conditions(recon_conf['loss_settings'])
    for image_label in image_labels:
        if label_out_of_bound(image_label, label_lowerbound, label_upperbound, label_list):
            continue
        if not subject_set is None:
            for subject in subject_set:
                for roi_for_each_loss in product(*roi_lists):
                    # prepare loss_instances and reconprocess
                    loss_lists = loss_dicts_to_loss_instances(recon_conf['loss_settings'], models_dict, features_dicts,
                                                              image_label=image_label, subject=subject, rois_list=roi_for_each_loss,
                                                              generator_BGR=generator_BGR, device=recon_conf['general_settings']['device'])
                    recon_process = create_ReconProcess_from_conf(image_label, models_dict, loss_lists, subject=subject, roi_for_each_loss=roi_for_each_loss, **recon_conf)
                    recon_process.optimize(print_logs=True)
                    del recon_process, loss_lists
                    torch.cuda.empty_cache()
                    gc.collect()
        else:
            loss_lists = loss_dicts_to_loss_instances(recon_conf['loss_settings'], models_dict, features_dicts,
                                                      image_label=image_label, subject='', rois_list=[],
                                                      generator_BGR=generator_BGR, device=recon_conf['general_settings']['device'])
            recon_process = create_ReconProcess_from_conf(image_label, models_dict, loss_lists, **recon_conf)
            recon_process.optimize(print_logs=True)
            torch.cuda.empty_cache()
            del recon_process, loss_lists
            gc.collect()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--recon_conf', type=str, required=True,
                        help='reconstruction configuration file')
    args = parser.parse_args()

    recon_conf_path = args.recon_conf
    with open(recon_conf_path, 'r') as f:
        recon_conf = yaml.safe_load(f)

    recon_conf['conf_file_name'] = str(Path(recon_conf_path).stem)

    run_reconstruction(recon_conf)

if __name__ == '__main__':
    main()