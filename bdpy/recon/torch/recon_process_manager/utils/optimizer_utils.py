import torch.optim as optim

__all__ = ['optim_name2class']

def optim_name2class(optim_name):
    if optim_name == 'SGD':
        return optim.SGD
    elif optim_name == 'Adam':
        return optim.Adam
    elif optim_name == 'Adadelta':
        return optim.Adadelta
    elif optim_name == 'Adagrad':
        return optim.Adagrad
    elif optim_name == 'AdamW':
        return optim.AdamW
    elif optim_name == 'SparseAdam':
        return optim.SparseAdam
    elif optim_name == 'Adamax':
        return optim.Adamax
    elif optim_name == 'ASGD':
        return optim.ASGD
    elif optim_name == 'RMSprop':
        return optim.RMSprop
    elif optim_name == 'Rprop':
        return optim.Rprop
    else:
        print('invalid optimizer selection! use Adam instead')
        return optim.Adam