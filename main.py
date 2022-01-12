import torch
import torch.cuda as cuda
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from cifar_loader import load_cifar
from Network import Network
from Loader import load_predictor_causes
from Loss import compute_loss, create_aux_indices, create_overlap_matrix, compute_gradients
from Reducer import create_reduced_causes, reset_reduced_causes
import Util


if __name__ == '__main__':
    # seed the CPUs and GPUs
    torch.manual_seed(0)
    cuda.manual_seed_all(0)

    # model architecture and parameters
    n_layer = 3
    input_sz = 32
    filter_sz = [7, 7, 7, 7, 8]
    n_channel = [3, 32, 64, 128, 256, 512]
    cont_training = False
    completed_epochs = 0

    # load data
    cifar_data, cifar_label = load_cifar(1, return_labels=True)
    horse_samples = (cifar_label == 7)
    ship_samples = (cifar_label == 8)
    cifar_data = cifar_data[horse_samples + ship_samples][:2].astype(np.float32)
    cifar_data = np.reshape(cifar_data, [2, 1024, 3, 1])

    # training parameters
    n_epoch = 2
    cause_epoch = 100
    reg_type = 2  # type of regularization used for weights and causes
    cause_lr = [0.1, 0.1]
    cause_reg = [0.1, 0.1]
    w_lr = [0.1, 0.1]
    w_reg = [0.1, 0.1]
    save_path = '/Users/shirin/projects/torch_PC/'

    # find number of devices if CUDA is available
    if cuda.is_available():
        n_devices = cuda.device_count()

        # move target for layer 1 neurons to the device 0
        cifar_data = torch.from_numpy(cifar_data).cuda(0)
    else:
        n_devices = 0
    # create the network
    net = Network(n_devices, n_layer, input_sz, filter_sz, n_channel, save_path, cont_training)
    reduced_causes = create_reduced_causes(net)

    # initialize/load causes for all layers
    predictor_causes, target_causes = load_predictor_causes(cont_training, net.cause_device, n_layer, cifar_data,
                                                            net.cause_sz, n_channel, save_path)

    # tracking the loss of the model
    epoch_loss = np.zeros([n_epoch + completed_epochs, n_layer - 1])
    if cont_training:
        epoch_loss[:completed_epochs, :] = np.load(save_path + 'results.npy')

    # TRAINING
    criterion = nn.MSELoss()
    w_optimizer = Util.create_weight_optimizers(net, w_lr)
    # pre-compute auxiliary indices for computing top-down and bottom-up losses
    take_indices, reduction_indices = create_aux_indices(n_layer, net.cause_sz, filter_sz, n_channel, net.cause_device)
    overlap_matrix = create_overlap_matrix(n_layer, net.cause_sz, filter_sz, net.cause_device)
    for e in range(n_epoch):
        for s in range(cifar_data.shape[0]):
            reset_reduced_causes(reduced_causes)  # reset top-down input for each layer
            inp = [Variable(x[s], requires_grad=True) for x in predictor_causes]
            cause_optimizer = Util.create_cause_optimizers(inp, cause_lr)

            # update the causes cause_epoch number of times
            for ce in range(cause_epoch):
                Util.reset_cause_grads(cause_optimizer)  # reset cause gradients

                predicted_cause = net(inp)
                if ce == 0:
                    # reduced predicted causes at each layer (except the bottom layer)
                    for l in range(1, n_layer - 1):
                        net.reduce_predicted_causes(l, predicted_cause[l].data, reduced_causes[l],
                                                    reduction_indices[l], overlap_matrix[l])

                target_cause = [x[s] for x in target_causes]
                losses = compute_loss('causes', criterion, inp, net, take_indices, target_cause, predicted_cause, reg_type,
                                      reduced_cause=reduced_causes, cause_reg=cause_reg)
                compute_gradients(losses)

                Util.apply_cause_updates(cause_optimizer)  # update causes

            # update weights once
            Util.reset_w_grads(w_optimizer)  # reset w gradients

            inp = [Variable(x[s]) for x in predictor_causes]
            predicted_cause = net(inp)

            target_cause = [x[s] for x in target_causes]
            losses = compute_loss('weights', criterion, inp, net, take_indices, target_cause, predicted_cause, reg_type,
                                  w_reg=w_reg)
            compute_gradients(losses)

            Util.apply_w_updates(w_optimizer)  # update weights

            # save loss in this iteration
            Util.save_loss(e + completed_epochs, epoch_loss, losses, cifar_data.shape[0])

        Util.update_targets(n_layer, predictor_causes, target_causes)
        Util.print_loss(e + completed_epochs, epoch_loss)

    Util.save_model(save_path, net, predictor_causes, epoch_loss)
    print('Done')