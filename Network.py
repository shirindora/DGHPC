import torch
import torch.nn as nn
import torch.cuda as cuda
import numpy as np


class Network (nn.Module):
    def __init__(self, n_devices, n_layer, input_sz, filter_sz, n_channel, save_path, cont_training):
        super(Network, self).__init__()

        self.n_devices = n_devices
        self.n_layer = n_layer
        self.input_sz = input_sz
        self.filter_sz = filter_sz
        self.n_channel = n_channel
        self.cont_training = cont_training

        # build the network
        self.w = []
        # self.causes = []
        self.cause_device = []
        self.cause_sz = [self.input_sz]
        for l in range(self.n_layer - 1):
            # compute the size of each layer based on the input and filter_sz
            self.cause_sz += [self.cause_sz[l] - self.filter_sz[l] + 1]

            # accordingly initialize the weights and causes for each layer
            b = self.cause_sz[l + 1] * self.cause_sz[l + 1]
            if cuda.is_available():
                sel_device = l % self.n_devices
                self.cause_device += [sel_device]

                with cuda.device(sel_device):
                    if cont_training:
                        loaded_weights = np.load(save_path + 'filters/' + str(l + 1) + '.npy')
                        self.w += [nn.Parameter(torch.from_numpy(loaded_weights)).cuda()]
                    else:
                        self.w += [nn.Parameter(cuda.FloatTensor(b, self.n_channel[l] * self.filter_sz[l] * self.filter_sz[l],
                                                                 self.n_channel[l + 1]).normal_())]
            else:
                if cont_training:
                    loaded_weights = np.load(save_path + 'filters/' + str(l + 1) + '.npy')
                    self.w += [nn.Parameter(torch.from_numpy(loaded_weights))]
                else:
                    self.w += [nn.Parameter(torch.randn(b, self.n_channel[l] * self.filter_sz[l] * self.filter_sz[l],
                                           self.n_channel[l + 1]))]

    def forward(self, x):
        predicted_cause = []
        for l in range(self.n_layer - 1):
            # self.causes[l].data = x[l]
            # predicted_cause += [torch.bmm(self.w[l], self.causes[l])]
            predicted_cause += [torch.bmm(self.w[l], x[l])]

        return predicted_cause

    def reduce_predicted_causes(self, level, predicted_cause, reduced_cause, reduction_indices, overlap_matrix):
        reshaped_predicted_cause = predicted_cause.view(-1, self.n_channel[level], 1)
        reduced_cause.index_add_(0, reduction_indices, reshaped_predicted_cause)
        reduced_cause.div_(overlap_matrix)


