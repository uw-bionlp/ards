

import torch


def get_activation(activation):

    if isinstance(activation, str):

        map = {
        "relu": torch.nn.ReLU,
        "relu6": torch.nn.ReLU6,
        "elu": torch.nn.ELU,
        "prelu": torch.nn.PReLU,
        "leaky_relu": torch.nn.LeakyReLU,
        "threshold": torch.nn.Threshold,
        "hardtanh": torch.nn.Hardtanh,
        "sigmoid": torch.nn.Sigmoid,
        "tanh": torch.nn.Tanh,
        "log_sigmoid": torch.nn.LogSigmoid,
        "softplus": torch.nn.Softplus,
        "softshrink": torch.nn.Softshrink,
        "softsign": torch.nn.Softsign,
        "tanhshrink": torch.nn.Tanhshrink,
        "selu": torch.nn.SELU,
        "gelu": torch.nn.GELU,
        }

        activation = map[activation]()

    return activation
