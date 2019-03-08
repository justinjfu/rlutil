import numpy as np
import torch

from rlutil.serializable import Serializable

USE_GPU = torch.cuda.is_available()
CUDA_DEVICE = torch.device('cuda')
CPU_DEVICE = torch.device('cpu')

def set_gpu(enable=True):
    global USE_GPU
    USE_GPU = enable

def get_device():
    if USE_GPU:
        return CUDA_DEVICE
    else:
        return CPU_DEVICE

def tensor(array, pin=False, dtype=torch.float32):
    if array is None:
        return None
    if isinstance(array, torch.Tensor):
        return array
    device = get_device()
    var = torch.tensor(array, dtype=dtype, device=device)
    if pin:
        var.pin_memory()
    return var

def all_tensor(arrays, **kwargs):
    return [tensor(arr, **kwargs) for arr in arrays]

def to_numpy(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor
    if USE_GPU:
        return tensor.detach().cpu().numpy()
    else:
        return tensor.detach().numpy()

def initialize_network(network):
    if USE_GPU:
        network.cuda()

def logsumexp(input, dim, keepdim=False, alpha=1.0):
    if alpha == 1.0:
        return torch.logsumexp(input, dim, keepdim=keepdim)
    else:
        return alpha * torch.logsumexp( input/alpha, dim, keepdim=keepdim)

def one_hot(y, n_dims=None):
    """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
    y_tensor = y.view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims, device=get_device()).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return y_one_hot


def copy_params_from_to(source, target):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class Module(torch.nn.Module, Serializable):
    def get_param_values(self):
        return self.state_dict()

    def set_param_values(self, param_values):
        self.load_state_dict(param_values)

    def get_param_values_np(self):
        state_dict = self.state_dict()
        np_dict = OrderedDict()
        for key, tensor in state_dict.items():
            np_dict[key] = to_numpy(tensor)
        return np_dict

    def set_param_values_np(self, param_values):
        torch_dict = OrderedDict()
        for key, tensor_ in param_values.items():
            torch_dict[key] = tensor(tensor_)
        self.load_state_dict(torch_dict)

    def copy(self):
        copy = Serializable.clone(self)
        copy_params_from_to(self, copy)
        return copy

    def save_init_params(self, locals):
        """
        Should call this FIRST THING in the __init__ method if you ever want
        to serialize or clone this network.

        Usage:
        ```
        def __init__(self, ...):
            self.init_serialization(locals())
            ...
        ```
        :param locals:
        :return:
        """
        Serializable.quick_init(self, locals)

    def __getstate__(self):
        d = Serializable.__getstate__(self)
        d["params"] = self.get_param_values()
        return d

    def __setstate__(self, d):
        Serializable.__setstate__(self, d)
        self.set_param_values(d["params"])

    def regularizable_parameters(self):
        """
        Return generator of regularizable parameters. Right now, all non-flat
        vectors are assumed to be regularizabled, presumably because only
        biases are flat.

        :return:
        """
        for param in self.parameters():
            if len(param.size()) > 1:
                yield param

    def eval_np(self, *args, **kwargs):
        """
        Eval this module with a numpy interface

        Same as a call to __call__ except all Variable input/outputs are
        replaced with numpy equivalents.

        Assumes the output is either a single object or a tuple of objects.
        """
        torch_args = tuple(torch_ify(x) for x in args)
        torch_kwargs = {k: torch_ify(v) for k, v in kwargs.items()}
        outputs = self.__call__(*torch_args, **torch_kwargs)
        if isinstance(outputs, tuple):
            return tuple(np_ify(x) for x in outputs)
        else:
            return np_ify(outputs)
