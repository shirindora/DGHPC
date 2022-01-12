import torch.optim as optim
import numpy as np


def create_weight_optimizers(net, w_lr):
    w_optimizer = []

    for l in range(net.n_layer - 1):
        w_optimizer += [optim.SGD([net.w[l]], lr=w_lr[l])]

    return w_optimizer


def create_cause_optimizers(inp, cause_lr):
    cause_optimizer = []

    for l in range(len(inp)):
        cause_optimizer += [optim.SGD([inp[l]], lr=cause_lr[l])]

    return cause_optimizer


def reset_cause_grads(cause_optimizer):
    n_optimizers = len(cause_optimizer)

    for l in range(n_optimizers):
        cause_optimizer[l].zero_grad()


def reset_w_grads(w_optimizer):
    n_optimizers = len(w_optimizer)

    for l in range(n_optimizers):
        w_optimizer[l].zero_grad()


def apply_cause_updates(cause_optimizer):
    n_optimizers = len(cause_optimizer)

    for l in range(n_optimizers):
        cause_optimizer[l].step()


def apply_w_updates(w_optimizer):
    n_optimizers = len(w_optimizer)

    for l in range(n_optimizers):
        w_optimizer[l].step()


def update_targets(n_layer, predictor, target):
    for l in range(1, n_layer - 1):
        target[l].copy_(predictor[l - 1])


def update_sample_targets(n_layer, sample, predictor, target):
    for l in range(1, n_layer - 1):
        target[l][sample, :, :, :].copy_(predictor[l - 1][sample, :, :, :])


def save_loss(epoch, epoch_loss, losses, n_sample):
    for l in range(len(losses)):
        epoch_loss[epoch, l] += (losses[l].cpu()[0] / n_sample)


def save_model(save_path, net, predictor_causes, epoch_loss):
    for l in range(len(predictor_causes)):
        np.save(save_path + 'causes/' + str(l + 1), predictor_causes[l].cpu().numpy())
        np.save(save_path + 'filters/' + str(l + 1), net.w[l].data.cpu().numpy())

    np.save(save_path + 'results', epoch_loss)


def print_loss(epoch, epoch_loss):
    print(str(epoch) + ': ', end='')
    for l in range(epoch_loss.shape[1]):
        if l < (epoch_loss.shape[1] - 1):
            print('l' + str(l) + ':' + str(epoch_loss[epoch, l]) + ', ', end='')
        else:
            print('l' + str(l) + ':' + str(epoch_loss[epoch, l]))
