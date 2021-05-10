'''PyTorch implementation of iCNN reconstruction.

Contributions:

Shen Guo-Hua developed the original Caffe implementation of iCNN
(https://github.com/KamitaniLab/icnn). Ken Shirakawa developed initial PyTorch
implementation of iCNN. Shuntaro Aoki refactored the initial PyTorch
implementation as well as fixed bugs and improved performance.

Developed and tested on Python 3.8 + PyTorch 1.7.1.

'''


import os

import numpy as np
from PIL import Image
import torch
import torch.optim as optim

from bdpy.dl.torch import FeatureExtractor
from bdpy.recon.utils import make_feature_masks, gaussian_blur
from bdpy.dataform import save_array
from bdpy.util import makedir_ifnot


def reconstruct(features,
                encoder,
                layer_mapping=None,
                generator=None,
                n_iter=200,
                loss_func=torch.nn.MSELoss(reduction='sum'),
                optimizer=optim.SGD,
                lr=(2., 1e-10),
                momentum=(0.9, 0.9),
                decay=(0.2, 1e-10),
                layer_weights=None,
                masks=None, channels=None,
                image_size=(224, 224, 3),
                initial_image=None,
                feature_size=(4096,),
                initial_feature=None,
                preproc=None,
                postproc=None,
                gradient_normalization=True,
                jittering=False, jitter_size=4,
                blurring=False,
                sigma=(2., 0.5),
                feature_upper_bound=None,
                feature_lower_bound=None,
                return_loss=False,
                output_dir='./output',
                save_snapshot=True,
                snapshot_dir='./output/snapshots',
                snapshot_ext='tiff',
                snapshot_interval=1,
                snapshot_postprocess=None,
                disp_interval=1,
                device='cpu'):
    '''
    Reconstruction an image.

    Parameters
    ----------
    features : dict
      Target DNN features. Keys are layer names and values are feature values.

      Example:

      features = {
        'conv1': np.array([...]),
        'conv2': np.array([...]),
        ...
      }

    encoder : torch.nn.Module
      Encoder network.

    generator : torch.nn.Module, optional
      Generator network.

    n_iter : int, optional
      The total number of iterations.

    loss_function : func, optional
      Loss function.

    optimizer : torch.optim.Optimizer
      Optimizer.

    lr : tuple, optional
      Learning rate.
      The learning rate will linearly decrease from lr[0] to lr[1] over iterations.

    momentum : tuple, optional
      Momentum.
      The momentum will linearly decrease from momentum[0] to momentum[1] over iterations.

    decay : tuple, optional
      Decay reate of images or features.
      The decay rate will linearly decrease from decay[0] to decay[1] over iterations.

    layer_weights : dict, optional
      Weights of layers in the loss function.
      If None, equal weights are used for all layers.

    masks : dict, optional
      Masks of DNN features.

    channels : dict, optional
      The channel numbers of each layer used in the loss function.

    image_size : tuple, optional
      Size of the image (h x w x c).

    initial_image : numpy.ndarar, optionaly

    preprocm, postproc : func, optional
      Pre- and post-processing functions on images.

    gradient_normalization : bool, optional

    jittering : bool, optional
      If True, reconstructed images are randomly shifted in each iteration.

    jitter_size : int, optional
      The number of pixels shifted in the jittering.

    blurring : bool, optional
      If True, Gaussian smoothing is applied on the reconstructed images in each iteration.

    sigma : tuple, optional
      The size of Gaussian kernel in the blurring.
      The sigma rate will linearly decrease from sigma[0] to sigma[1] over iterations.

    feature_size : tuple, optional
      Size of features fed to the generator.

    initial_feature : numpy.ndarary, optional
      Initial generator features.

    feature_upper_bound, feature_lower_bound : scalar, optional
      Upper and lower bound of generator features.

    return_loss : bool, optional
      If True, the function returns loss history.

    output_dir : str, optional
      Path to output directory.

    save_snapshot : bool, optional
      If True, snapshots (intermediate reconstructed images) will be saved.

    snapshot_dir : str, optional
      Path to the directory to save snapshots.

    snapshot_ext : str, optional
      File extension (e.g., 'jpg', 'tiff') of snapshots.

    snapshot_interval : int, optional
      Save snapshots for every N iterations.

    snapshot_postprocess : func, optional
      Postprocessing function applied on the snapshots.

    display_interval : int, optional
      Display information for every N iterations.

    device : str, optional (default: 'cpu')
      PyTorch device (e.g., 'cuda:0', 'cpu').

    Returns
    -------
    numpy.ndarray
      A reconstructed image.
    list, optional
      Loss history.


    Note
    ----
    The original Caffe implementation of icnn was written by Shen Guo-Hua.
    The initial PyTorch implementation was written by Ken Shirakawa.

    Reference
    ---------
    Shen et al. (2019) Deep image reconstruction from human brain activity.
      PLOS Computational Biology. https://doi.org/10.1371/journal.pcbi.1006633
    '''

    use_generator = generator is not None

    # Directory setup
    makedir_ifnot(output_dir)
    if save_snapshot:
        makedir_ifnot(snapshot_dir)

    # Noise image
    # noise_image = np.random.randint(0, 256, image_size)
    # image_norm0 = np.linalg.norm(noise_image) / 2.

    # Initial image/features
    if initial_image is None:
        initial_image = np.random.randint(0, 256, image_size)

    if initial_feature is None:
        initial_feature = np.random.normal(0, 1, feature_size).astype(np.float32)

    if save_snapshot:
        if use_generator:
            # Save the initial generator feature
            save_name = 'initial_gen_feat'
            save_array(os.path.join(snapshot_dir, save_name),
                       initial_feature, key='initial_gen_feat')
        else:
            # Save the initial image
            save_name = 'initial_image.' + snapshot_ext
            Image.fromarray(np.uint8(initial_image)).save(
                os.path.join(snapshot_dir, save_name)
            )

    # Layer list
    layers = list(features.keys())
    n_layers = len(layers)
    print('Layers: {}'.format(layers))

    # Layer weights
    if layer_weights is None:
        w = np.ones(n_layers, dtype=np.float32)
        w = w / w.sum()
        layer_weights = {
            layer: w[i]
            for i, layer in enumerate(layers)
        }

    # Feature masks
    feature_masks = make_feature_masks(features,
                                       masks=masks,
                                       channels=channels)

    # Main -------------------------------------------------------------------

    if use_generator:
        f = initial_feature.copy().astype(np.float32)
    else:
        x = initial_image.copy().astype(np.float32)  # Is this copy necessary?
        if preproc is not None:
            x = preproc(x)

    loss_history = np.zeros(n_iter, dtype=np.float32)

    feature_extractor = FeatureExtractor(encoder, layers, layer_mapping,
                                         device=device, detach=False)

    encoder.zero_grad()
    if use_generator:
        generator.zero_grad()

    # Optimizer setup
    if use_generator:
        ft = torch.tensor(f[np.newaxis], device=device, requires_grad=True)
        op = optimizer([ft], lr=lr[0])
    else:
        xt = torch.tensor(x[np.newaxis], device=device, requires_grad=True)
        op = optimizer([xt], lr=lr[0])

    op.zero_grad()

    for it in range(n_iter):

        if use_generator:
            encoder.zero_grad()
            generator.zero_grad()

        # Learning parameters
        lr_it = lr[0] + it * (lr[1] - lr[0]) / n_iter
        momentum_it = momentum[0] + it * (momentum[1] - momentum[0]) / n_iter
        decay_it = decay[0] + it * (decay[1] - decay[0]) / n_iter
        sigma_it = sigma[0] + it * (sigma[1] - sigma[0]) / n_iter

        for g in op.param_groups:
            if 'lr' in g:
                g['lr'] = lr_it
            if 'momentum' in g:
                g['momentum'] = momentum_it

        # Jittering
        if jittering:
            ox, oy = np.random.randint(-jitter_size, jitter_size + 1, 2)
            x = np.roll(np.roll(x, ox, -1), oy, -2)

        if use_generator:
            # Generate an image
            ft.data = torch.tensor(f[np.newaxis], device=device)
            xt = generator.forward(ft)
            # xt.retain_grad()

            # Crop the generated image
            gen_image_size = (xt.shape[2], xt.shape[3])
            top_left = ((gen_image_size[0] - image_size[0]) // 2,
                        (gen_image_size[1] - image_size[1]) // 2)

            image_mask = np.zeros(xt.shape)
            image_mask[:, :,
                       top_left[0]:top_left[0] + image_size[0],
                       top_left[1]:top_left[1] + image_size[1]] = 1
            image_mask_t = torch.FloatTensor(image_mask).to(device)

            xt = torch.masked_select(xt, image_mask_t.bool()).view(
                (1, image_size[2], image_size[0], image_size[1])
            )
        else:
            # Convert x to torch.tensor
            xt.data = torch.tensor(x[np.newaxis], device=device)

        # Forward (calculate features)
        activations = feature_extractor.run(xt)

        # Backward
        err = 0.
        loss = 0.

        eval('encoder.{}.zero_grad()'.format(layer_mapping[layers[-1]]))

        for j, lay in enumerate(reversed(layers)):

            act_j = activations[lay].clone().to(device)
            feat_j = torch.tensor(features[lay], device=device).clone()

            mask_j = torch.FloatTensor(feature_masks[lay]).to(device)

            weight_j = layer_weights[lay]

            masked_act_j = torch.masked_select(act_j, mask_j.bool()).view(act_j.shape)
            masked_feat_j = torch.masked_select(feat_j, mask_j.bool()).view(feat_j.shape)

            loss_j = loss_func(masked_act_j, masked_feat_j) * weight_j
            loss += loss_j

        loss.backward()

        # Normalize gradient
        if gradient_normalization:
            if use_generator:
                grad_mean = torch.abs(ft.grad).mean().cpu().detach()
                if grad_mean > 0:
                    ft.grad /= grad_mean
            else:
                grad_mean = torch.abs(xt.grad).mean().cpu().detach()
                if grad_mean > 0:
                    xt.grad /= grad_mean

        # Image update
        x_buf = xt.clone().cpu().detach().numpy()[0]
        op.step()
        if use_generator:
            f = ft.cpu().detach().numpy()[0]
            x = x_buf
        else:
            x = xt.cpu().detach().numpy()[0]

        err += loss
        loss_history[it] = loss.cpu().detach().numpy()

        # Unshift
        if jittering:
            x = np.roll(np.roll(x, -ox, -1), -oy, -2)

        # L2 decay
        if use_generator:
            f = (1 - decay_it) * f
        else:
            x = (1 - decay_it) * x

        # Gaussian bluring
        if blurring:
            x = gaussian_blur(x, sigma_it)

        # Clipping features (upper/lower bound)
        if use_generator:
            if feature_lower_bound is not None:
                f = np.maximum(f, feature_lower_bound)
            if feature_upper_bound is not None:
                f = np.minimum(f, feature_upper_bound)
            f = f.astype(np.float32)

        # Disp info
        if (it + 1) % disp_interval == 0:
            print('iteration = {}; err = {}'.format(it + 1, err))

        # Save snapshot
        if save_snapshot and ((it + 1) % snapshot_interval == 0):
            snapshot = x.copy()
            save_path = os.path.join(snapshot_dir, '%06d.%s' % (it + 1, snapshot_ext))
            if snapshot_postprocess is not None and postproc is not None:
                snapshot = snapshot_postprocess(postproc(snapshot))
            elif snapshot_postprocess is not None and postproc is None:
                snapshot = snapshot_postprocess(snapshot)
            elif snapshot_postprocess is None and postproc is not None:
                snapshot = postproc(snapshot)

            Image.fromarray(snapshot.astype(np.uint8)).save(save_path)

    # Postprocessing
    if postproc is not None:
        x = postproc(x)

    # Returns
    if return_loss:
        return x, loss_history
    else:
        return x
