import numpy as np
import torch
import torch.cuda as cuda
from torch.autograd import Variable
import operator
from functools import reduce


def compute_loss(update_for, criterion, inp, net, take_indices, target_cause, predicted_cause, reg_type,
                 reduced_cause=None, cause_reg=None, w_reg=None):
    losses = []
    n_losses = len(predicted_cause)

    if update_for == 'causes':
        for l in range(n_losses):
            if l == (n_losses - 1):  # top layer has no top-down input
                losses += [criterion(predicted_cause[l].unsqueeze(0),
                                     Variable(torch.take(target_cause[l], take_indices[l]).unsqueeze(0))) +
                           (cause_reg[l] * torch.norm(inp[l], reg_type) / reduce(operator.mul, list(inp[l].shape)))]
            else:
                losses += [criterion(predicted_cause[l].unsqueeze(0),
                                     Variable(torch.take(target_cause[l], take_indices[l]).unsqueeze(0))) +
                           criterion(inp[l].unsqueeze(0), Variable(reduced_cause[l + 1].unsqueeze(0))) +
                           (cause_reg[l] * torch.norm(inp[l], reg_type) / reduce(operator.mul, list(inp[l].shape)))]
    elif update_for == 'weights':
        for l in range(n_losses):
            losses += [criterion(predicted_cause[l], Variable(torch.take(target_cause[l], take_indices[l]))) +
                       (w_reg[l] * torch.norm(net.w[l], reg_type) / reduce(operator.mul, list(net.w[l].shape)))]

    return losses


def compute_gradients(losses):
    n_layer = len(losses)

    for l in range(n_layer):
        losses[l].backward()


def create_aux_indices(n_layer, cause_sz, filter_sz, n_channel, cause_device):
    take_indices = []
    reduction_indices = []

    for l in range(n_layer - 1):
        b = cause_sz[l + 1] * cause_sz[l + 1]
        layer_take_indices = np.zeros([b, filter_sz[l] * filter_sz[l] * n_channel[l], 1], dtype=np.int)
        layer_reduction_indices = np.zeros(b * filter_sz[l] * filter_sz[l], dtype=np.int)

        # start filling take indices
        n_contiguous_items = filter_sz[l] * n_channel[l]
        for x in range(b):
            row = x // cause_sz[l + 1]
            col = x % cause_sz[l + 1]

            for f in range(filter_sz[l]):
                # create matrix of indices for calling torch.take
                start_target_idx = (((row + f) * cause_sz[l]) + col) * n_channel[l]
                start_take_idx = f * filter_sz[l] * n_channel[l]

                layer_take_indices[x, start_take_idx:(start_take_idx + n_contiguous_items), :] = \
                    np.expand_dims(np.arange(start_target_idx, (start_target_idx + n_contiguous_items), 1), axis=1)

                # create matrix of indices for calling torch.Tensor.index_add_()
                pixel_coordinate = ((row + f) * cause_sz[l]) + col
                start_reduction_idx = (x * filter_sz[l] * filter_sz[l]) + (f * filter_sz[l])

                layer_reduction_indices[start_reduction_idx:(start_reduction_idx + filter_sz[l])] = \
                    np.arange(pixel_coordinate, (pixel_coordinate + filter_sz[l]))

        if cuda.is_available():
            take_indices += [torch.from_numpy(layer_take_indices).cuda(cause_device[l])]
            reduction_indices += [torch.from_numpy(layer_reduction_indices).cuda(cause_device[l])]
        else:
            take_indices += [torch.from_numpy(layer_take_indices)]
            reduction_indices += [torch.from_numpy(layer_reduction_indices)]

    return take_indices, reduction_indices


def create_overlap_matrix(n_layer, cause_sz, filter_sz, cause_device):
    overlap_matrix = []

    for l in range(n_layer - 1):
        layer_overlap_matrix = np.zeros(cause_sz[l] * cause_sz[l], dtype=np.float32)

        for x in range(cause_sz[l]):
            for y in range(cause_sz[l]):
                right_offset = filter_sz[l] if x >= (filter_sz[l] - 1) else (x + 1)
                left_offset = filter_sz[l] if (cause_sz[l] - x) >= filter_sz[l] else (cause_sz[l] - x)
                column_span = right_offset + left_offset - filter_sz[l]

                top_offset = filter_sz[l] if y >= (filter_sz[l] - 1) else (y + 1)
                bottom_offset = filter_sz[l] if (cause_sz[l] - y) >= filter_sz[l] else (cause_sz[l] - y)
                row_span = top_offset + bottom_offset - filter_sz[l]

                layer_overlap_matrix[(x * cause_sz[l]) + y] = column_span * row_span

        # None has been added to ensure that overlap_matrix can be broadcasted to the desired dimension while being used
        if cuda.is_available():
            overlap_matrix += [torch.from_numpy(layer_overlap_matrix[:, None, None]).cuda(cause_device[l])]
        else:
            overlap_matrix += [torch.from_numpy(layer_overlap_matrix[:, None, None])]

    return overlap_matrix
