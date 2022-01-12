import torch
import torch.cuda as cuda


def create_reduced_causes(net):
    reduced_causes = []

    for l in range(net.n_layer - 1):
        b = net.cause_sz[l] * net.cause_sz[l]

        if cuda.is_available():
            with cuda.device(net.cause_device[l]):
                reduced_causes += [cuda.FloatTensor(b, net.n_channel[l], 1)]
        else:
            reduced_causes += [torch.FloatTensor(b, net.n_channel[l], 1)]

    return reduced_causes


def reset_reduced_causes(reduced_causes):
    for rc in reduced_causes:
        rc.zero_()
