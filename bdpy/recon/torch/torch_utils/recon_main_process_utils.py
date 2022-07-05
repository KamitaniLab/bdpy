from pathlib import Path

import numpy as np

from ..recon_main_process import ReconProcess
# from recon_main_process import ReconProcess
from image_processing import image_deprocess_in_tensor
# from recon_utils import image_preprocess, image_deprocess, image_deprocess_in_tensor

def create_ReconProcess_from_conf(image_label, models_dict, loss_lists, subject='', roi_for_each_loss=[''], **recon_conf):
    general_settings = recon_conf['general_settings']
    output_settings = recon_conf['output_settings']
    optimization_settings = recon_conf['optimization_settings']
    generator_settings = optimization_settings['generator']

    model_instance = None
    if generator_settings['use_generator']:
        model_name = generator_settings['network_name']
        model_instance = models_dict[model_name]['model_instance']

    # output_root = output_settings['output_root']
    roi_names = ''
    for roi in roi_for_each_loss:
        roi_names += '_' + roi
    roi_names = roi_names[1:]
    output_dir = Path(output_settings['output_root']) / recon_conf['conf_file_name'] / subject / roi_names
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_settings['save_snapshot']:
        snapshot_dir = output_dir / 'snapshots' / image_label
        snapshot_dir.mkdir(parents=True, exist_ok=True)

    image_mean = None
    if 'image_mean_file' in optimization_settings:
        image_mean = np.load(optimization_settings['image_mean_file'])
        image_mean = np.float32([image_mean[0].mean(), image_mean[1].mean(), image_mean[2].mean()])
    # image_deprocess
    generator_deprocess = None
    if generator_settings['use_generator']:
        generator_deprocess =\
            lambda img_tensor: image_deprocess_in_tensor(img_tensor,
                                                         image_mean=np.float32(generator_settings['deprocess']['mean']),
                                                         image_std=np.float32(generator_settings['deprocess']['std']))
    # image_postprocess optimization_settings['image_postprocess']
    # snapshot_postprocess optimization_settings['snapshot_postprocess']
    return ReconProcess(loss_lists, **general_settings, **output_settings,
                        **optimization_settings, **generator_settings,
                        output_dir=output_dir, snapshot_dir=snapshot_dir,
                        generator_model=model_instance,
                        image_deprocess=generator_deprocess,
                        image_mean=image_mean)