import re
from pathlib import Path
import warnings

from PIL import Image
import numpy as np
import torch
import gc

# Import from bdpy
from bdpy.dataform import save_array
from bdpy.recon.utils import gaussian_blur, normalize_image, clip_extreme
from .utils import optim_name2class, image_deprocess_in_tensor, get_image_deprocess_fucntion

from scipy.io import savemat
import matplotlib.pyplot as plt

__all__ = ['ReconProcess', 'create_ReconProcess_from_conf']

def print_with_verbose(*args, verbose=False, **kwargs):
    if verbose:
        print(*args, **kwargs)

def is_in_and_not_None(key, target_dict):
    return key in target_dict and target_dict[key] is not None

class ReconProcess:
    # TODO: `reset` function
    def __init__(self, loss_func_dicts,
                 initial_image=None, initial_feature=None,
                 image_mean=None,
                 generator_model=None, optimizer_info=[],
                 image_shape=None, feature_shape=None,
                 image_deprocess=None,
                 inter_step_processes=[], stabilizing_processes=[],
                 normalize_gradients=True,
                 n_iter=100, device='cpu',
                 use_generator=False, output_dir=None,
                 snapshot_dir=None, snapshot_interval=50,
                 save_generator_feature=False, save_loss_hist=False,
                 snapshot_postprocess=None, image_postprocess=None,
                 snapshot_ext='.tiff', result_image_ext='.tiff',
                 loss_history_ext='.png',
                 result_saving_pattern_head='<image_label>_<iteration(:0>6)>',
                 snapshot_saving_pattern_head='<iteration(:0>6)>',
                 result_image_pattern_tail='_normalized',
                 snapshot_image_pattern_tail='',
                 loss_history_pattern_tail='_loss_history',
                 feature_saving_pattern_tail='_feature',
                 image_label='',
                 generator_output_BGR=False, **args):
        '''
        initialization
        '''
        self.device = device
        # FIXME: image_shape should be fed to loss calculation classes
        # -> image_shape in this class is only used for optimization in the pixel space
        self.image_shape = image_shape
        assert loss_func_dicts is not None
        self.loss_func_dicts = loss_func_dicts

        self.n_iter = n_iter
        self.current_iteration = 1
        if use_generator:
            assert generator_model is not None
            self.image_array = None
            self.feature_array = self.initialize_feature_array(initial_feature, feature_shape)
            self.feature_tensor = torch.tensor(self.feature_array[None], device=self.device, requires_grad=True).float()
            self.generator_model = generator_model
            self.use_generator = True
            self.image_deprocess = image_deprocess
            self.optimizer = optim_name2class(optimizer_info['optimizer_name'])([self.feature_tensor], lr=2.0) # lr will be configurated in each step
        else:
            self.generator_model = None
            self.use_generator = False
            self.image_array = self.initialize_image_array(initial_image, image_shape, image_mean)
            self.image_tensor = torch.tensor(self.image_array.transpose(2,0,1)[None], device=self.device, requires_grad=True)
            self.optimizer = optim_name2class(optimizer_info['optimizer_name'])([self.image_tensor], lr=2.0) # lr will be configurated in each step
        self.loss_history = np.zeros(n_iter, dtype=np.float32)
        self.optimizer_param_dicts = optimizer_info['param_dicts']
        self.normalize_gradients = normalize_gradients
        self.inter_step_process_dicts = inter_step_processes
        self.stabilizing_process_dicts = stabilizing_processes

        # TODO: deal with None case
        self.output_dir = output_dir
        self.snapshot_dir = snapshot_dir
        self.save_intervals = snapshot_interval
        self.save_feature = save_generator_feature

        self.save_loss_hist = save_loss_hist
        self.snapshot_postprocess = snapshot_postprocess
        self.image_postprocess = image_postprocess
        self.snapshot_ext = snapshot_ext
        self.result_image_ext = result_image_ext
        self.result_saving_pattern_head = result_saving_pattern_head
        self.snapshot_saving_pattern_head = snapshot_saving_pattern_head
        self.loss_history_ext = loss_history_ext
        self.result_image_pattern_tail = result_image_pattern_tail
        self.snapshot_image_pattern_tail = snapshot_image_pattern_tail
        self.loss_history_pattern_tail = loss_history_pattern_tail
        self.feature_saving_pattern_tail = feature_saving_pattern_tail
        self.image_label = image_label
        self.generator_output_BGR = generator_output_BGR

    def initialize_image_array(self, initial_image, image_shape, image_mean, value_range=(0, 256)):
        if initial_image is None:
            if image_mean is not None:
                initial_image = np.zeros(image_shape, dtype='float32')
                # since image_mean was BGR, we needed to reverse the order to make it RGB
                # for i in range(3):
                #     initial_image[:, :, i] = image_mean[2-i].copy()
                # TODO: make sure the image_mean is given in the RGB order
                for i in range(3):
                    initial_image[:, :, i] = image_mean[i].copy()
                return initial_image
            else:
                return np.random.randint(value_range[0], value_range[1], image_shape)
        else:
            assert isinstance(initial_image, np.ndarray), print(type(initial_image))
            assert initial_image.shape == tuple(image_shape)
            return initial_image
    def initialize_feature_array(self, initial_feature, feature_size, value_range=(0, 1), dtype=np.float32):
        if initial_feature is None:
            return np.random.normal(value_range[0], value_range[1], feature_size).astype(dtype)
        else:
            assert isinstance(initial_feature, np.ndarray)
            assert initial_feature.shape == tuple(feature_size)
            return initial_feature

    def get_current_image(self):
        '''
        return current image as tensor
        '''
        # TODO: DA
        if self.use_generator:
            image_tensor = self.generate_image()
            if self.image_deprocess is not None:
                image_tensor = self.image_deprocess(image_tensor)
            self.image_tensor = image_tensor
        else:
            self.image_tensor.data = torch.tensor(self.image_array.transpose(2,0,1)[None], device=self.device)
        return self.image_tensor

    def generate_image(self):
        '''
        generate image from feature tensor using the generator
        '''
        # TODO: DA
        self.feature_tensor.data = torch.tensor(self.feature_array[None], device=self.device).float()
        self.generator_model.zero_grad()
        self.generator_model.to(self.device)
        self.generator_model = self.generator_model.float()
        image_tensor = self.generator_model(self.feature_tensor)

        return image_tensor

    def configurate_optimizer(self, new_optimizer_param_dicts=None, print_logs=False):
        '''
        configurate optimizer's parameters and grad
        '''
        if new_optimizer_param_dicts is not None:
            tmp_optimizer_param_dicts = new_optimizer_param_dicts
            print_with_verbose('update params using values given by functional args', verbose=print_logs)
        else:
            tmp_optimizer_param_dicts = self.optimizer_param_dicts
        for optimizer_param_dict in tmp_optimizer_param_dicts:
            param_name = optimizer_param_dict['param_name']
            if param_name in self.optimizer.param_groups[0]:
                param_values = optimizer_param_dict['param_values']
                if not isinstance(param_values, (int, float)):
                    if len(param_values) > 1:
                        param_it = float(param_values[0]) + (self.current_iteration - 1) * (float(param_values[1]) - float(param_values[0])) / (self.n_iter- 1)
                    else:
                        param_it = float(param_values[0])
                else:
                    param_it = float(param_values)
                print_with_verbose('parameter {} is updated from {} to {}'.format(param_name, self.optimizer.param_groups[0][param_name], param_it), verbose=print_logs)
                self.optimizer.param_groups[0][param_name] = param_it
            else:
                warnings.warn('parameter {} is specified but not in optimizer.param_gropus'.format(param_name))
        self.optimizer.zero_grad()

    def gradient_normalization(self, print_logs=False):
        print_with_verbose('normalize gradients', verbose=print_logs)
        if self.use_generator:
            grad_mean = torch.abs(self.feature_tensor.grad).mean().cpu().detach()
            if grad_mean > 0:
                self.feature_tensor.grad /= grad_mean
        else:
            grad_mean = torch.abs(self.image_tensor.grad).mean().cpu().detach()
            if grad_mean > 0:
                self.image_tensor.grad /= grad_mean

    def inter_step_process(self):
        for inter_step_process_dict in self.inter_step_process_dicts:
            process_name = inter_step_process_dict['process_name']
            if process_name == 'L2_decay':
                decay_values = inter_step_process_dict['param_values']
                decay_it = float(decay_values[0]) + (self.current_iteration - 1) * (float(decay_values[1]) - float(decay_values[0])) / self.n_iter
                if not self.use_generator:
                    self.image_array = (1 - decay_it) * self.image_array
                else:
                    self.feature_array = (1 - decay_it) * self.feature_array
            elif process_name == 'image_blurring' and not self.use_generator:
                sigma_values = inter_step_process_dict['param_values']
                sigma_it = float(sigma_values[0]) + (self.current_iteration - 1) * (float(sigma_values[1]) - float(sigma_values[0])) / self.n_iter
                self.image_array = gaussian_blur(self.image_array, sigma_it)
            elif process_name == 'feature_clipping' and self.use_generator:
                feature_lower_bound, feature_upper_bound = inter_step_process_dict['param_values']
                if feature_lower_bound is not None:
                    if not isinstance(feature_lower_bound, np.ndarray):
                        feature_lower_bound = np.loadtxt(feature_lower_bound)
                        inter_step_process_dict['param_values'][0] = feature_lower_bound
                    self.feature_array = np.maximum(self.feature_array, feature_lower_bound)
                if feature_upper_bound is not None:
                    if not isinstance(feature_upper_bound, np.ndarray):
                        feature_upper_bound = np.loadtxt(feature_upper_bound)
                        inter_step_process_dict['param_values'][1] = feature_upper_bound
                    self.feature_array = np.minimum(self.feature_array, feature_upper_bound)
                self.feature_array.astype(np.float32)
            else:
                warnings.warn('Unknown inter-step process: {}'.format(process_name))

    def stabilizing_process(self, backward=False):
        '''
        techniques to supress noise or make optimization stable
        currently only jittering is expected
        '''
        if not backward:
            tmp_dicts = []
            for stabilizing_process_dict in self.stabilizing_process_dicts:
                process_name = stabilizing_process_dict['process_name']
                if process_name == 'jittering' and not self.use_generator:
                    jitter_size = stabilizing_process_dict['jitter_size']
                    ox, oy = np.random.randint(-jitter_size, jitter_size + 1, 2)
                    self.image_array = np.roll(np.roll(self.image_array, ox, -1), oy, -2)
                    stabilizing_process_dict['x_jitter_size'] = ox
                    stabilizing_process_dict['y_jitter_size'] = oy
                    tmp_dicts.append(stabilizing_process_dict)
                else:
                    warnings.warn('Unknown stabilizing process: {}'.format(process_name))
            self.stabilizing_process_dicts = tmp_dicts
        else:
            for stabilizing_process_dict in self.stabilizing_process_dicts:
                process_name = stabilizing_process_dict['process_name']
                if process_name == 'jittering' and not self.use_generator:
                    ox, oy = stabilizing_process_dict['x_jitter_size'], stabilizing_process_dict['y_jitter_size']
                    self.image_array = np.roll(np.roll(self.image_array, -ox, -1), -oy, -2)
                else:
                    warnings.warn('Unknown stabilizing process: {}'.format(process_name))

    def forward(self, new_optimizer_param_dicts=None, print_logs=False):
        '''
        one optimization step. if you want to change optimizer params in some specific ways, you should call `forward` iteratively.
        otherwise, you just need to call `optimize`
        '''
        self.configurate_optimizer(new_optimizer_param_dicts, print_logs)
        self.inter_step_process()
        self.stabilizing_process()
        image_batch = self.get_current_image()

        loss = 0
        print_with_verbose('iteration {}'.format(self.current_iteration), verbose=print_logs)
        for loss_func_dict in self.loss_func_dicts:
            c_loss = loss_func_dict['loss_func'](image_batch)
            # TODO: format the log
            print_with_verbose('{}: {}'.format(loss_func_dict['loss_type'], c_loss.item()), verbose=print_logs)
            loss += c_loss * loss_func_dict['weight']
        print_with_verbose('total: {}'.format(loss.item()), verbose=print_logs)
        # TODO: change here to save each loss value respectively
        self.loss_history[self.current_iteration-1] = loss.cpu().detach().numpy()
        loss.backward()
        del loss

        if self.normalize_gradients:
            self.gradient_normalization()
        if self.use_generator:
            self.image_array = image_batch.clone().detach().cpu().numpy()[0].transpose(1,2,0)
        self.optimizer.step()

        if self.use_generator:
            self.feature_array = self.feature_tensor.detach().cpu().numpy()[0]
        else:
            self.image_array = image_batch.detach().cpu().numpy()[0].transpose(1,2,0)
        self.stabilizing_process(backward=True)
        self.current_iteration += 1
        del image_batch
        gc.collect()

    def optimize(self, print_logs=False):
        while self.current_iteration <= self.n_iter:
            if (self.current_iteration - 1) % self.save_intervals == 0:
                self.save_results(print_logs=print_logs)
            self.forward(print_logs=print_logs)
        self.save_results(print_logs=print_logs, final=True)

    def filename_formatting(self, output_type='snapshot') -> str:
        if output_type == 'snapshot':
            saving_name = self.snapshot_saving_pattern_head
        else:
            assert output_type == 'final'
            saving_name = self.result_saving_pattern_head
        saving_name = saving_name.replace('<image_label>', self.image_label)
        iteration_tokens = re.findall('(<iteration[^<]*>)', saving_name)
        if len(iteration_tokens) > 0:
            for iteration_token in iteration_tokens:
                formatting_option = re.findall('\((.*)\)', iteration_token)
                if len(formatting_option) > 0:
                    formatting_option = '{' + formatting_option[0] + '}'
                else:
                    formatting_option = '{}'
                saving_name = saving_name.replace(iteration_token, formatting_option).format(self.current_iteration - 1)
        if output_type == 'snapshot':
            saving_name = self.snapshot_dir / saving_name
        else:
            saving_name = self.output_dir / saving_name
        return str(saving_name)

    def save_results(self, print_logs=False, final=False):
        if not final:
            saving_name = self.filename_formatting('snapshot')
            if self.image_array is not None:
                snapshot = self.image_array.copy()
                if self.image_postprocess is not None:
                    snapshot = self.image_postprocess(snapshot)
                if self.snapshot_postprocess is not None:
                    snapshot = self.snapshot_postprocess(snapshot)
                if self.generator_output_BGR:
                    print('BGR to RGB')
                    snapshot = snapshot[:,:,::-1]
                snapshot = snapshot.clip(min=0, max=255)

                image_saving_name = Path(saving_name + self.snapshot_image_pattern_tail).with_suffix(self.snapshot_ext)
                Image.fromarray(snapshot.astype(np.uint8)).save(image_saving_name)
                print_with_verbose("saved snapshot: {}".format(saving_name), verbose=print_logs)
            if (self.current_iteration - 1) == 0 and self.use_generator and self.save_feature:
                feature_saving_name = saving_name + self.feature_saving_pattern_tail + '.mat'
                save_array(feature_saving_name, self.feature_array, key='initial_gen_feat')
                print_with_verbose("saved initial feature: {}".format(Path(feature_saving_name).with_suffix('.mat')), verbose=print_logs)
        else:
        # use (self.current_iteration - 1) for saving name
            saving_name = self.filename_formatting('final')
            image = self.image_array.copy()
            if self.image_postprocess is not None:
                image = self.image_postprocess(image)
            if self.generator_output_BGR:
                image = image[:,:,::-1]
            image_saving_name = Path(saving_name + self.result_image_pattern_tail).with_suffix(self.result_image_ext)
            image = Image.fromarray(normalize_image(clip_extreme(image, pct=4)))
            image.save(image_saving_name)
            print_with_verbose("saved the last image as {}".format(image_saving_name), verbose=print_logs)
            if self.save_loss_hist:
                loss_history_saving_name = Path(saving_name + self.loss_history_pattern_tail).with_suffix(self.loss_history_ext)
                figure = plt.figure()
                plt.plot(self.loss_history)
                plt.savefig(loss_history_saving_name)
                del figure
                print_with_verbose("saved loss history: {}".format(loss_history_saving_name), verbose=print_logs)
            if self.use_generator and self.save_feature:
                feature_saving_name = saving_name + self.feature_saving_pattern_tail + '.mat'
                savemat(feature_saving_name, {'final_generator_feature': self.feature_array})
                print_with_verbose('saved the final feature: {}'.format(feature_saving_name))

def create_ReconProcess_from_conf(image_label, models_dict, loss_lists, subject='', roi_for_each_loss=[''], **recon_conf):
    general_settings = recon_conf['general_settings']
    output_settings = recon_conf['output_settings']
    optimization_settings = recon_conf['optimization_settings']
    generator_settings = optimization_settings['generator']

    model_instance = None
    if generator_settings['use_generator']:
        model_name = generator_settings['network_name']
        model_instance = models_dict[model_name]['model_instance']

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
    if 'image_mean_file' in optimization_settings and optimization_settings['image_mean_file'] is not None:
        image_mean_file_info = optimization_settings['image_mean_file']
        image_mean = np.load(image_mean_file_info['file_path'])
        if image_mean_file_info['BGR']:
            image_mean = np.float32([image_mean[2].mean(), image_mean[1].mean(), image_mean[0].mean()])
        else:
            image_mean = np.float32([image_mean[0].mean(), image_mean[1].mean(), image_mean[2].mean()])

    # image_deprocess for generator
    generator_deprocess = None
    if generator_settings['use_generator'] and is_in_and_not_None('deprocess', generator_settings):
        generator_deprocess =\
            lambda img_tensor: image_deprocess_in_tensor(img_tensor,
                                                         image_mean=np.float32(generator_settings['deprocess']['mean']),
                                                         image_std=np.float32(generator_settings['deprocess']['std']))
    image_postprocess = None
    if is_in_and_not_None('image_postprocess', output_settings):
        if callable(output_settings['image_postprocess']):
            image_postprocess = output_settings['image_postprocess']
        else:
            image_postprocess = get_image_deprocess_fucntion(output_settings['image_postprocess']['mean'],
                                                             output_settings['image_postprocess']['std'],
                                                             BGR2RGB=False)
        # output_settings['image_postprocesss_mean'] = np.float32(output_settings['image_postprocess']['mean'])
        # output_settings['image_postprocess_std'] = np.float32(output_settings['image_postprocess']['std'])
        # # def image_postprocess(img_array):
        # #     return image_deprocess(img_array,
        # #                            image_mean=np.float32(output_settings['image_postprocess']['mean']),
        # #                            image_std=np.float32(output_settings['image_postprocess']['std']),
        # #                            BGR2RGB=False)
        # image_postprocess =\
        #     lambda img_array: image_deprocess(img_array,
        #                                       image_mean=output_settings['image_postprocess_mean'],
        #                                       image_std=output_settings['image_postprocess_std'],
        #                                       BGR2RGB=False)
    output_settings['image_postprocess'] = image_postprocess
    snapshot_postprocess = None
    if is_in_and_not_None('snapshot_postprocess', output_settings):
        if callable(output_settings['snapshot_postprocess']):
            snapshot_postprocess = output_settings['snapshot_postprocess']
        else:
            snapshot_postprocess = get_image_deprocess_fucntion(output_settings['snapshot_postprocess']['mean'],
                                                                output_settings['snapshot_postprocess']['std'],
                                                                BGR2RGB=False)
        # output_settings['snapshot_postprocesss_mean'] = np.float32(output_settings['snapshot_postprocess']['mean'])
        # output_settings['snapshot_postprocess_std'] = np.float32(output_settings['snapshot_postprocess']['std'])
        # # def image_postprocess(img_array):
        # #     return image_deprocess(img_array,
        # #                            image_mean=np.float32(output_settings['image_postprocess']['mean']),
        # #                            image_std=np.float32(output_settings['image_postprocess']['std']),
        # #                            BGR2RGB=False)
        # snapshot_postprocess =\
        #     lambda img_array: image_deprocess(img_array,
        #                                       image_mean=output_settings['snapshot_postprocess_mean'],
        #                                       image_std=output_settings['snapshot_postprocess_std'],
        #                                       BGR2RGB=False)
        # # def snapshot_postprocess(img_array):
        # #     return image_deprocess(img_array,
        # #                            image_mean=np.float32(output_settings['snapshot_postprocess']['mean']),
        # #                            image_std=np.float32(output_settings['snapshot_postprocess']['std']),
        # #                            BGR2RGB=False)
        # snapshot_postprocess=\
        #     lambda img_array: image_deprocess(img_array,
        #                                       image_mean=np.float32(output_settings['snapshot_postprocess']['mean']),
        #                                       image_std=np.float32(output_settings['snapshot_postprocess']['std']),
        #                                       BGR2RGB=False)
    output_settings['snapshot_postprocess'] = snapshot_postprocess

    return ReconProcess(loss_lists, **general_settings, **output_settings,
                        **optimization_settings, **generator_settings,
                        output_dir=output_dir, snapshot_dir=snapshot_dir,
                        generator_model=model_instance,
                        image_deprocess=generator_deprocess,
                        image_mean=image_mean, image_label=image_label)