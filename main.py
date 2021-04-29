
__author__ = "Yuan Zi"
__email__ = "yzi2@central.uh.edu"
__version__ = "1.0.0"

from utiles import gprMaxInversion
from utiles import plot_final_section
from utiles import plot_record
from utiles import GaussianSmoothing
import torch.nn as nn
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import os
import h5py

def main():
    # manully build gt model
    """..."""
    # forward
    inverser = gprMaxInversion()
    inverser.path_gprMax = '/home/yzi/research/gprMax'
    for i in range(inverser.num_shot):
        inverser.forward(in_folder='/home/yzi/research/gprMax-FWI/output/forward/forward_gt',
                         in_filename='cross_well_cylinder_B_scan_shot_'+str(i)+'.in')
    # observation update
    inverser.update_obs(mode='ground truth observation')

    loss_record = []
    steps = []
    gradients_scale = []
    # build initial model
    p_tensor = torch.zeros((50, 50)) + 5.5
    c_tensor = torch.zeros((50, 50)) + 0.0028


    for epoch in range(10):
        for i in range(inverser.num_shot):
            inverser.build_model(permittivity_tensor=p_tensor, conductivity_tensor=c_tensor,
                                 gt_gprmaxin_file='/home/yzi/research/gprMax-FWI/output/forward/forward_gt/cross_well_cylinder_B_scan_shot_'+str(i)+'.in'
                                 , folder='/home/yzi/research/gprMax-FWI/output/forward/forward_curr')
            # forward with initial model
            inverser.forward(in_folder='/home/yzi/research/gprMax-FWI/output/forward/forward_curr',
                             in_filename='cross_well_cylinder_B_scan_shot_'+str(i)+'.in')

        # current observation update
        inverser.update_obs(mode='current observation')
        # update residual observation
        inverser.update_residual_obs()
        # time reverse of the observation
        inverser.update_time_reverse_sum()
        # write residual source hd5f file
        inverser.make_res_source_hd5_file(folder='/home/yzi/research/gprMax-FWI/output/forward/forward_curr')
        # make back propagation gprMax file (replace shot line with residual shots)
        inverser.make_backward_gprMax_infile(
            forward_gprmaxin_file='/home/yzi/research/gprMax-FWI/output/forward/forward_curr/cross_well_cylinder_B_scan_shot_0.in'
            , folder='/home/yzi/research/gprMax-FWI/output/forward/inverse')
        # backpropagation
        inverser.backpropogation(in_folder='/home/yzi/research/gprMax-FWI/output/forward/inverse',
                                 in_filename='backpropogation.in')

        # loss calculated
        loss_record.append(inverser.check_loss())
        # gradient calculated
        inverser.calc_gradient()
        # step length calculation
        # update parameters

        # p_log = torch.clone(torch.log(p_tensor[20:31, 15:36]))
        # c_log = torch.clone(torch.log(c_tensor[20:31, 15:36]))


        # # gradient gaussian
        # smoothing = GaussianSmoothing(1, 3, 0.6)
        # input_unsqueeze = inverser.p_grad_fields[5:-5, :].unsqueeze(0).unsqueeze(0)
        # input_unsqueeze = F.pad(input_unsqueeze, (1, 1, 1, 1), mode='reflect')
        # inverser.p_grad_fields[5:-5, :] = smoothing(input_unsqueeze).squeeze(0).squeeze(0)
        # input = inverser.c_grad_fields[5:-5, :].unsqueeze(0).unsqueeze(0)
        # input = F.pad(input, (1, 1, 1, 1), mode='reflect')
        # inverser.c_grad_fields[5:-5, :] = smoothing(input).squeeze(0).squeeze(0)


        # # gradient to log gradient
        # inverser.p_grad_fields = inverser.p_grad_fields * p_tensor[15:36, 15:36]
        # inverser.c_grad_fields = inverser.c_grad_fields * c_tensor[15:36, 15:36]

        if epoch == 0:
            # permittivity
            inverser.p_conju_grad = inverser.p_grad_fields
            inverser.c_conju_grad = inverser.c_grad_fields
        else:
            # conductivity
            inverser.p_conju_grad = inverser.p_grad_fields + inverser.p_conju_grad_cache * (
                        inverser.p_grad_fields * (inverser.p_grad_fields - inverser.p_grad_cache)) / (
                                                inverser.p_grad_cache * inverser.p_grad_cache)
            # inverser.c_conju_grad = inverser.c_grad_fields + inverser.c_conju_grad_cache * (
            #             inverser.c_grad_fields * (inverser.c_grad_fields - inverser.c_grad_cache)) / (
            #                                     inverser.c_grad_cache * inverser.c_grad_cache)

        p_step = inverser.calc_step(p_tensor=p_tensor, c_tensor=c_tensor, kappa_p=1e-9, mode='p')
        # c_step = inverser.calc_step(p_tensor=p_tensor, c_tensor=c_tensor, kappa_p=1e-12, mode='c')
        steps.append(p_step)
        gradients_scale.append(inverser.p_conju_grad[5:-5, :].mean().numpy())
        #print('c_step:', c_step)

        p_tensor[20:31, 15:36] = p_tensor[20:31, 15:36] - p_step * inverser.p_conju_grad[5:-5, :]
        # c_tensor[20:31, 15:36] = c_tensor[20:31, 15:36] - c_step * inverser.c_conju_grad[5:-5, :]

        # p_tensor[20:31, 15:36] = torch.exp(p_log - p_step * inverser.p_conju_grad[5:-5, :])
        # c_tensor[20:31, 15:36] = torch.exp(c_log - c_step * inverser.c_conju_grad[5:-5, :])


        # p_tensor[15:36, 15:36] = p_tensor[15:36, 15:36] - step * inverser.conju_grad
        # # p_tensor[15:36, 15:36] =  torch.exp(p_log - step * inverser.conju_grad*p_tensor[15:36, 15:36])
        # # mean filter
        # mean_filter = nn.Sequential(
        #     nn.Conv2d(1, 1, 3, stride=1,padding=1)
        # )
        # p_tensor_nn_input_l = p_tensor[15:20].unsqueeze(0).unsqueeze(0)
        # p_tensor_nn_input_r = p_tensor[30:36].unsqueeze(0).unsqueeze(0)
        # p_tensor[15:20] = mean_filter(p_tensor_nn_input_l).squeeze(0).squeeze(0).detach()
        # p_tensor[30:36] = mean_filter(p_tensor_nn_input_r).squeeze(0).squeeze(0).detach()

        # store gradient to cache
        inverser.p_grad_cache = torch.clone(inverser.p_grad_fields)
        inverser.p_conju_grad_cache = torch.clone(inverser.p_conju_grad)
        # inverser.c_grad_cache = torch.clone(inverser.c_grad_fields)
        # inverser.c_conju_grad_cache = torch.clone(inverser.c_conju_grad)

        plot_final_section(path_figures='/home/yzi/research/gprMax-FWI/output/figures',
                           para_matrix=p_tensor, epoch=epoch, mode='p')
        plot_final_section(path_figures='/home/yzi/research/gprMax-FWI/output/figures',
                           para_matrix=c_tensor, epoch=epoch, mode='c')
    plot_record(path_figures='/home/yzi/research/gprMax-FWI/output/figures',
              record=loss_record, mode='loss')
    plot_record(path_figures='/home/yzi/research/gprMax-FWI/output/figures',
              record=steps, mode='step')
    plot_record(path_figures='/home/yzi/research/gprMax-FWI/output/figures',
              record=gradients_scale, mode='grad_scale')



if __name__ == "__main__":
    main()


