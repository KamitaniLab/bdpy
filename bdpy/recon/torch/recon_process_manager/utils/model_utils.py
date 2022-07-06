import torch
from torchvision import transforms
if __file__ == '/home/eitoikuta/bdpy_update/bdpy/bdpy/recon/torch/recon_process_manager/utils/model_utils.py':
    # from importlib.machinery import SourceFileLoader
    # bdpy = SourceFileLoader("bdpy","/home/eitoikuta/bdpy_update/bdpy/bdpy/__init__.py").load_module()
    import importlib.util
    spec = importlib.util.spec_from_file_location('dl', "/home/eitoikuta/bdpy_update/bdpy/bdpy/dl/torch/models.py")
    dl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dl)
else:
    import bdpy.dl as dl

def create_model_instance(model_name, device='cpu', training=False, **args):
    if model_name == 'CLIP_ViT-B_32':
        import clip
        model, preprocess = clip.load('ViT-B/32', device=device)
        if not training:
            model.eval()
        if 'image_encoder_only' in args and args['image_encoder_only']:
            model = model.visual.float()
        elif 'text_encoder_only' in args and args['text_encoder_only']:
            model = model.transformer.float()
        # FIXME: the option `preprocess_from_PIL` does not work now
        if not 'preprocess_from_PIL' in args or not args['preprocess_from_PIL']:
            preprocess = transforms.Normalize((0.48145466*255, 0.4578275*255, 0.40821073*255), (0.26862954*255, 0.26130258*255, 0.27577711*255))
        return model, preprocess
    elif model_name == 'AlexNetGenerator_ILSVRC2012_Training_relu7':
        # from bdpy.dl import AlexNetGenerator
        model = dl.AlexNetGenerator().to(device)
        if 'params_file' in args:
            model.load_state_dict(torch.load(args['params_file']))
        if not training:
            model.eval()
        return model, None
    # TODO: check whether following two models work correctly
    elif model_name == 'VGG19':
        # from bdpy.dl import VGG19
        model = dl.VGG19().to(device)
        if 'params_file' in args:
            model.load_state_dict(torch.load(args['params_file']))
        if not training:
            model.eval()
        return model, None
    elif model_name == 'AlexNet':
        # from bdpy.dl import AlexNet
        # TODO: accept different number of classes
        model = dl.AlexNet().to(device)
        if 'params_file' in args:
            model.load_state_dict(torch.load(args['params_file']))
        if not training:
            model.eval()
        return model, None
    else:
        assert False, print('Unknown model name is specified: {}'.format(model_name))