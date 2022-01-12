import torch
import torch.cuda as cuda
import numpy as np


def load_predictor_causes(cont_training, cause_device, n_layer, tr_data, cause_sz, n_channel, save_path=None):
    n_sample = tr_data.shape[0]
    predictor_causes = []
    target_causes = []

    for l in range(n_layer - 1):
        b_predictor = cause_sz[l + 1] * cause_sz[l + 1]
        b_target = cause_sz[l] * cause_sz[l]

        if cont_training is False:
            if cuda.is_available():
                with cuda.device(cause_device[l]):
                    predictor_causes += [cuda.FloatTensor(n_sample, b_predictor, n_channel[l + 1], 1).normal_()]

                    if l == 0:  # layer 1
                        target_causes += [tr_data]
                    else:
                        # I think I can remove the call to copy_ below. But I am not sure owing to my limited
                        # knowledge of pytorch internals. But if the call to copy_ can be removed, it implies that I
                        # don't have to update target causes manually after learning. This would be a useful
                        # optimization
                        target_data = cuda.FloatTensor(n_sample, b_target, n_channel[l], 1)
                        target_causes += [target_data.copy_(predictor_causes[l - 1])]
            else:
                predictor_causes += [torch.randn(n_sample, b_predictor, n_channel[l + 1], 1)]

                # load targets
                if l == 0:  # layer 1
                    target_causes += [torch.from_numpy(tr_data)]
                else:
                    target_causes += [predictor_causes[l - 1].clone()]
        else:
            if cuda.is_available():
                with cuda.device(cause_device[l]):
                    loaded_causes = torch.from_numpy(np.load(save_path + 'causes/' + str (l + 1) + '.npy'))
                    predictor_causes += [loaded_causes.cuda()]

                    # load targets
                    if l == 0:  # layer 1
                        target_causes += [tr_data]
                    else:
                        # I think I can remove the call to copy_ below. But I am not sure owing to my limited
                        # knowledge of pytorch internals. But if the call to copy_ can be removed, it implies that I
                        # don't have to update target causes manually after learning. This would be a useful
                        # optimization
                        target_data = cuda.FloatTensor(n_sample, b_target, n_channel[l], 1)
                        target_causes += [target_data.copy_(predictor_causes[l - 1])]
            else:
                predictor_causes += [torch.from_numpy(np.load(save_path + 'causes/' + str(l + 1) + '.npy'))]

                # load targets
                if l == 0:  # layer 1
                    target_causes += [torch.from_numpy(tr_data)]
                else:
                    target_causes += [predictor_causes[l - 1].clone()]

    return predictor_causes, target_causes


def load_w():
    # TODO Add code for loading learned weights
    pass
