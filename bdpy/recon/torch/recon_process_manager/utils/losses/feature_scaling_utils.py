#coding:utf-8
import warnings

import numpy as np
import copy


def factor_settings_dict_to_factor(original_feature, current_feature, factor_settings: dict, layer: str):
    if 'target_layer' not in factor_settings or layer in factor_settings['target_layers']:
        if isinstance(factor_settings['values'], str):
            func_name = factor_settings['values']
            assert func_name in ['mean', 'std', 'original_mean', 'original_std']
            if 'original' in func_name:
                target_feature = original_feature
            else:
                target_feature = current_feature
            assert 'calculation_type' in factor_settings
            calculation_type = factor_settings['calculation_type']
            if calculation_type == 'positional':
                # if original_feature.ndim == 4:
                if target_feature.ndim >= 3:
                    axis = (0, 1)
                else:
                    axis = None
            else:
                assert calculation_type == 'all_units_to_one'
                axis = None
            if func_name == 'mean':
                return np.mean(target_feature, axis=axis, keepdims=True)
            else:
                if 'std_ddof' in factor_settings:
                    std_ddof = factor_settings['std_ddof']
                else:
                    print('std_ddof is set to 0')
                    std_ddof = 0
                return np.std(target_feature, axis=axis, ddof=std_ddof, keepdims=True)
        else:
            return factor_settings['values'][layer]
    return None

def prepare_normalization_factors(original_feature, current_feature, factor_settings, layer: str):
    if isinstance(factor_settings, dict):
        factor = factor_settings_dict_to_factor(original_feature, current_feature, factor_settings, layer)
    else:
        assert isinstance(factor_settings, list)
        factor = None
        for factor_settings_i in factor_settings:
            if factor is not None:
                break
            factor = factor_settings_dict_to_factor(original_feature, current_feature, factor_settings_i, layer)
    return factor

def normalize_features(target_features: dict,
                       normalization_settings: dict):
    '''
    returns (features - subtracthend) / divisor * multiplier + addend for each layer
    normalization_settings:
        {subtracthend: {'values': <ndarray>, 'calculation_type': 'unit-wise'},
         divisor: {'values': <ndarray>, 'calculation_type': 'all_units_to_one', 'std_ddof': 1},
         multiplier: {'values': <ndarray>, 'calculation_type': 'positional', 'std_ddof': 0},
         addend: {'values': <ndarray>, 'calculation_type': 'unit-wise'}}

        - If values is str (choices are ['std', 'mean']), values calculated from original given features will be used
        {subtracthend: {'values': 'mean', 'calculation_type': 'unit-wise'},
         divisor: {'values': <ndarray>, 'calculation_type': 'all_units_to_one', 'std_ddof': 1},
         multiplier: {'values': 'std', 'calculation_type': 'positional', 'std_ddof': 1},
         addend: {'values': <ndarray>, 'calculation_type': 'unit-wise'}}

        - None is acceptable for all elements
        {subtracthend: None,
         divisor: {'values': <ndarray>, 'calculation_type': 'all_units_to_one', 'std_ddof': 1},
         multiplier: None,
         addend: {'values': <ndarray>, 'calculation_type': 'unit-wise'}}

        - If 'target_layers' in the sub-dict, they will applied to only layers in the list.
        {subtracthend: {'values': <ndarray>, 'calculation_type': 'unit-wise', 'target_layers': <list of layers>},
         divisor: {'values': <ndarray>, 'calculation_type': 'all_units_to_one', 'std_ddof': 1},
         multiplier: {'values': <ndarray>, 'calculation_type': 'positional', 'std_ddof': 0},
         addend: {'values': <ndarray>, 'calculation_type': 'unit-wise'}}

        - If you want to use different values for some layers, use following:
        {subtracthend: [{'values': <ndarray>, 'target_layers': <list of layers>, 'calculation_type': 'unit-wise'},
                        {'values': <ndarray>, 'target_layers': <list of layers>, 'calculation_type': 'positional'}],
         divisor: None
         multiplier: None
         addend: None}
    '''

    for layer, feature in target_features.items():
        print(layer, end=':\n')
        original_feature = copy.deepcopy(feature)
        if normalization_settings['subtracthend'] is not None:
            subtracthend = prepare_normalization_factors(original_feature, feature, normalization_settings['subtracthend'], layer)
            if subtracthend is not None:
                feature = feature - subtracthend
                print('subtracthend:', subtracthend.shape, np.mean(subtracthend))
            else:
                print('subtracthend: None')

        if normalization_settings['dividor'] is not None:
            dividor = prepare_normalization_factors(original_feature, feature, normalization_settings['dividor'], layer)
            if dividor is not None:
                if np.any(dividor == 0):
                    warnings.warn('there are {} 0 values in the dividor'.format(np.sum(dividor == 0)))
                    dividor = np.where(dividor == 0, 1, dividor)
                feature = feature / dividor
                print('dividor:', dividor.shape, np.mean(dividor))
            else:
                print('dividor: None')

        if normalization_settings['multiplier'] is not None:
            multiplier = prepare_normalization_factors(original_feature, feature, normalization_settings['multiplier'], layer)
            if multiplier is not None:
                if np.any(multiplier == 0):
                    warnings.warn('there are {} 0 values in the multiplier'.format(np.sum(multiplier == 0)))
                feature = feature * multiplier
                print('multiplier:', multiplier.shape, np.mean(multiplier))
            else:
                print('multiplier: None')

        if normalization_settings['addend'] is not None:
            addend = prepare_normalization_factors(original_feature, feature, normalization_settings['addend'], layer)
            if addend is not None:
                feature = feature + addend
                print('addend:', addend.shape, np.mean(addend))
            else:
                print('addend: None')

        target_features[layer] = feature

    return target_features
