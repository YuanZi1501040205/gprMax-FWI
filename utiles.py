# %%
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import os
import h5py
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors
import os
import cv2

class gprMaxInversion():
    def __init__(self):
        self.path_gprMax = ''
        self.forward_inputfilename = ''
        self.forward_inputdirectory = ''
        self.forward_outputdirectory = ''
        self.back_inputfilename = ''
        self.back_inputdirectory = ''
        self.back_outputdirectory = ''
        self.para_value_tensor = torch.tensor([])


        self.x_len = 21
        self.y_len = 21
        self.sig_len = 425
        self.num_rceivers = 5
        self.num_shot = 5

        self.forward_fields = torch.zeros((self.num_shot, self.x_len, self.y_len, self.sig_len))
        self.backward_fields = torch.zeros((self.x_len, self.y_len, self.sig_len))
        # permittivity
        self.p_grad_fields = torch.zeros((self.x_len, self.y_len))
        self.p_grad_fields_cache = torch.zeros((self.x_len, self.y_len))
        self.p_conju_grad = torch.zeros((self.x_len, self.y_len))
        self.p_conju_grad_cache = torch.zeros((self.x_len, self.y_len))
        # conductivity
        self.c_grad_fields = torch.zeros((self.x_len, self.y_len))
        self.c_grad_fields_cache = torch.zeros((self.x_len, self.y_len))
        self.c_conju_grad = torch.zeros((self.x_len, self.y_len))
        self.c_conju_grad_cache = torch.zeros((self.x_len, self.y_len))

        self.gt_obs_fields = torch.zeros((self.num_shot, self.num_rceivers, self.sig_len))
        self.curr_obs_fields = torch.zeros((self.num_shot, self.num_rceivers, self.sig_len))
        self.res_obs_fields = torch.zeros((self.num_shot, self.num_rceivers, self.sig_len))
        self.reverse_res_obs_fields = torch.zeros((self.num_shot, self.num_rceivers, self.sig_len))
        self.disturb_obs_fields = torch.zeros((self.num_shot, self.num_rceivers, self.sig_len))

    def calc_step(self, p_tensor, c_tensor, kappa_p, mode='p'):
        if mode == 'p':
            para_tensor_disturb = torch.clone(p_tensor)
            para_tensor_disturb[15:36, 15:36] = para_tensor_disturb[15:36, 15:36] + kappa_p * self.p_conju_grad
            c_tensor = c_tensor
            for i in range(self.num_shot):
                self.build_model(permittivity_tensor=para_tensor_disturb, conductivity_tensor=c_tensor,
                             gt_gprmaxin_file='/home/yzi/research/gprMax-FWI/output/forward/forward_gt/cross_well_cylinder_B_scan_shot_'+str(i)+'.in',
                             folder='/home/yzi/research/gprMax-FWI/output/forward/inverse_disturb')
                self.forward(in_folder='/home/yzi/research/gprMax-FWI/output/forward/inverse_disturb',
                                 in_filename='cross_well_cylinder_B_scan_shot_'+str(i)+'.in')
            self.update_obs(mode='disturb observation')
            step = kappa_p*((self.disturb_obs_fields - self.curr_obs_fields) * self.res_obs_fields).sum()/\
                   ((self.disturb_obs_fields - self.curr_obs_fields)**2).sum()
        elif mode == 'c':
            para_tensor_disturb = torch.clone(c_tensor)
            para_tensor_disturb[15:36, 15:36] = para_tensor_disturb[15:36, 15:36] + kappa_p * self.c_conju_grad
            p_tensor = p_tensor
            for i in range(self.num_shot):
                self.build_model(permittivity_tensor=p_tensor, conductivity_tensor=para_tensor_disturb,
                             gt_gprmaxin_file='/home/yzi/research/gprMax-FWI/output/forward/forward_gt/cross_well_cylinder_B_scan_shot_'+str(i)+'.in',
                             folder='/home/yzi/research/gprMax-FWI/output/forward/inverse_disturb')
                self.forward(in_folder='/home/yzi/research/gprMax-FWI/output/forward/inverse_disturb',
                                 in_filename='cross_well_cylinder_B_scan_shot_'+str(i)+'.in')
            self.update_obs(mode='disturb observation')
            step = kappa_p*((self.disturb_obs_fields - self.curr_obs_fields) * self.res_obs_fields).sum()/\
                   ((self.disturb_obs_fields - self.curr_obs_fields)**2).sum()
        else:
            exit('Step calculate stop')
        return step


    def check_loss(self):
        return (self.res_obs_fields**2).sum().numpy()/2

    def calc_gradient(self):
        time_partial_forward_fields = torch.zeros((self.num_shot, self.x_len, self.y_len, self.sig_len))
        for i in range(self.sig_len):
            if i == 0:
                time_partial_forward_fields[:, :, :, i]=0
            else:
                time_partial_forward_fields[:, :, :, i] = self.forward_fields[:, :, :, i] - self.forward_fields[:, :, :, i - 1]
        self.p_grad_fields = (time_partial_forward_fields * self.backward_fields).sum(dim=-1).sum(dim=0)
        self.c_grad_fields = (self.forward_fields * self.backward_fields).sum(dim=-1).sum(dim=0)


    def make_backward_gprMax_infile(self, forward_gprmaxin_file, folder):
        f1 = open(forward_gprmaxin_file, 'r')
        f1_Lines = f1.readlines()
        head_lines = f1_Lines[:4]
        waveform_lines = []
        shot_loc_lines = []
        shot_start = 1.5
        shot_interval = 0.5
        for i in range(self.num_rceivers):
            shot_loc = shot_start + i*shot_interval
            shot_loc = ("%.1f"%shot_loc)
            waveform_lines.append('#waveform: '+'resshot_'+str(i)+' 1.0 5e7 mysource_'+str(i) +'\n')
            shot_loc_lines.append('#hertzian_dipole: z 3.5 '+str(shot_loc)+' 0.0 mysource_'+str(i)+'\n')
        end_lines = f1_Lines[6:]
        new_file = folder + '/' + 'backpropogation.in'
        file = open(new_file, "w")
        file.writelines(head_lines)
        file.writelines(waveform_lines)
        file.writelines(shot_loc_lines)
        file.writelines(end_lines)

    def backpropogation(self, in_folder, in_filename):
        in_file = in_folder + '/' + in_filename
        os.system('eval "$(conda shell.bash hook)"\n'
                  # + 'conda activate work\n'
                  + 'cd ' + self.path_gprMax + '\n'
                  + 'python -m gprMax ' + in_file + '\n')
        field_file_path = in_file.split('.in')[0] + '.out'
        h5_file = h5py.File(field_file_path, 'r')
        rxs = h5_file['rxs']
        for _ in range(rxs.__len__()):
            rx = rxs['rx' + str(_ + 1)]
            i = int(_ / self.y_len)
            j = _ % self.y_len
            sig_tensor = torch.tensor(rx['Ez'][()])
            self.backward_fields[i][j][:] = sig_tensor  # ([x,y,z,t])
        # os.system('cd ' + in_folder + '\d'
        #           + 'rm -f ' + field_file_path)

    def make_res_source_hd5_file(self, folder):
        name_data = 'residual_source'
        f = h5py.File(folder + '/' + name_data + '.h5', 'w')
        X = self.reverse_res_obs_fields.numpy()
        f.create_dataset('X', data=X)
        f.close()

    def update_time_reverse_sum(self):
        self.reverse_res_obs_fields = torch.flip(self.res_obs_fields, dims=[2]).sum(dim=0)

    def update_obs(self, mode):
        for i in range(self.num_shot):
            for j in range(self.num_rceivers):
                if mode == 'ground truth observation':
                    self.gt_obs_fields[i][j] = self.forward_fields[i][-1][j*5][:]
                elif mode == 'current observation':
                    self.curr_obs_fields[i][j] = self.forward_fields[i][-1][j*5][:]
                elif mode == 'disturb observation':
                    self.disturb_obs_fields[i][j] = self.forward_fields[i][-1][j*5][:]
                else:
                    exit('wrong obs update')

    def update_residual_obs(self):
        self.res_obs_fields = self.curr_obs_fields - self.gt_obs_fields

    def forward(self, in_folder, in_filename):
        in_file = in_folder + '/' + in_filename
        index_shot = int(in_filename.split('_')[-1].split('.in')[0])
        os.system('eval "$(conda shell.bash hook)"\n'
                  # + 'conda activate work\n'
                  + 'cd ' + self.path_gprMax + '\n'
                  + 'python -m gprMax ' + in_file + '\n')
        field_file_path = in_file.split('.in')[0] + '.out'
        h5_file = h5py.File(field_file_path, 'r')
        rxs = h5_file['rxs']
        for _ in range(rxs.__len__()):
            rx = rxs['rx' + str(_ + 1)]
            i = int(_ / self.y_len)
            j = _%self.y_len
            sig_tensor = torch.tensor(rx['Ez'][()])
            self.forward_fields[index_shot][i][j][:] = sig_tensor  # ([x,y,t])
        # os.system('cd ' + in_folder + '\d'
        #           + 'rm -f ' + field_file_path)

    def build_model(self, permittivity_tensor, conductivity_tensor, gt_gprmaxin_file, folder):
        f1 = open(gt_gprmaxin_file, 'r')
        f1_Lines = f1.readlines()
        head_lines = f1_Lines[:9]
        material_lines = []
        geo_model_lines = []
        for i in range(permittivity_tensor.shape[0]):
            for j in range(permittivity_tensor.shape[1]):
                x_l = i * 0.1
                x_r = (i + 1) * 0.1
                y_l = j * 0.1
                y_r = (j + 1) * 0.1
                x_l = ("%.2f" % x_l)
                x_r = ("%.2f" % x_r)
                y_l = ("%.2f" % y_l)
                y_r = ("%.2f" % y_r)
                index_pix = "pix_" + str(i) + "_" + str(j)
                if i < 15 or i > 34 or j<15 or j>34:
                    pass
                else:
                    permittivity_value = ("%.16f" % permittivity_tensor[i][j].numpy())
                    conductivity_value = ("%.16f" % conductivity_tensor[i][j].numpy())
                    material_lines.append("#material: " + str(permittivity_value) + " " + str(
                        conductivity_value) + " 1.0 0.0 " + index_pix + "\n")
                    geo_model_lines.append(
                        "#box: " + x_l + " " + y_l + " 0.0 " + x_r + " " + y_r + " 0.1 " + index_pix + "\n")
        geo_view_line = ['#geometry_view: 0.0 0.0 0.0 5.0 5.0 0.1 0.1 0.1 0.1 cross_well_half_space n']
        new_file = folder + '/' + gt_gprmaxin_file.split('/')[-1]
        file = open(new_file, "w")
        file.writelines(head_lines)
        file.writelines(material_lines)
        file.writelines(geo_model_lines)
        file.writelines(geo_view_line)


def plot_final_section(path_figures, para_matrix, epoch, mode):
    # plot p_tensor

    image = cv2.rotate(para_matrix.numpy(), cv2.ROTATE_90_COUNTERCLOCKWISE)
    fig, axes = plt.subplots(1, 1, sharex=True, sharey=True)
    max_v = np.percentile(np.abs(image), 99)
    min_v = np.percentile(np.abs(image), 1)
    norm = colors.Normalize(vmin=min_v, vmax=max_v, clip=True)

    im1 = axes.imshow(image, aspect='auto', norm=norm, cmap='seismic')
    axes.set_title('permittivity')
    axes.set_xlabel('x')
    axes.set_ylabel('y')
    fig.colorbar(im1)
    title = 'FWI ' + mode + ' '+str(epoch)
    fig.suptitle(title, verticalalignment='center')
    plt.savefig(os.path.join(path_figures, title + '.png'))
    plt.close()
    plt.cla()

def plot_record(path_figures, record, mode):
    # plot p_tensor
    plt.plot(record)
    if mode == 'loss':
        title = 'FWI loss'
    elif mode == 'step':
        title = 'step'
    elif mode == 'grad_scale':
        title = 'grad_scale'
    plt.savefig(os.path.join(path_figures, title + '.png'))
    plt.close()
    plt.cla()




import math
import numbers
import torch
from torch import nn
from torch.nn import functional as F

class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)
# # %% epoch update
# import numpy as np
# from scipy.sparse.linalg import cg
# import time
#
#
# def conjugate_grad(A, b, x=None):
#     """
#     Description
#     -----------
#     Solve a linear equation Ax = b with conjugate gradient method.
#     Parameters
#     ----------
#     A: 2d numpy.array of positive semi-definite (symmetric) matrix
#     b: 1d numpy.array
#     x: 1d numpy.array of initial point
#     Returns
#     -------
#     1d numpy.array x such that Ax = b
#     """
#     n = len(b)
#     if not x:
#         x = np.ones(n)
#     r = np.dot(A, x) - b
#     p = - r
#     r_k_norm = np.dot(r, r)
#     for i in xrange(2*n):
#         Ap = np.dot(A, p)
#         alpha = r_k_norm / np.dot(p, Ap)
#         x += alpha * p
#         r += alpha * Ap
#         r_kplus1_norm = np.dot(r, r)
#         beta = r_kplus1_norm / r_k_norm
#         r_k_norm = r_kplus1_norm
#         if r_kplus1_norm < 1e-5:
#             print 'Itr:', i
#             break
#         p = beta * p - r
#     return x
#
# # x_ph = tf.placeholder('float32', [None, None])
# # r = tf.matmul(A, x_ph) - b
#
# if __name__ == '__main__':
#     n = 1000
#     P = np.random.normal(size=[n, n])
#     A = np.dot(P.T, P)
#     b = np.ones(n)
#
#     t1 = time.time()
#     print 'start'
#     x = conjugate_grad(A, b)
#     t2 = time.time()
#     print t2 - t1
#     x2 = np.linalg.solve(A, b)
#     t3 = time.time()
#     print t3 - t2
#     x3 = cg(A, b)
#     t4 = time.time()
#     print t4 - t3
# %%
# # %%
#

# model = Geo_Model()
#
# def MSE(pred, label):
#     return ((pred - label) ** 2)
#
# # %% work result of 4/14 night!
# a = Variable(torch.randn(10, 10))
# gradient = torch.randn((10, 10))
# b = Variable(a, requires_grad=True)
# c = torch.sum(b * gradient)
# c.backward()
# # %%
# for x, y in zip(xb, yb):
#     optimizer.zero_grad()
#     pred = model(x)
#     loss_value = criterion(pred, y)
#     loss_value.backward()
#     optimizer.step()
#
# # %%
# in_file = '/home/yzi/research/gprMax-FWI/output/forward/forward_input_gt/cross_well_cylinder_B_scan_shot_1.in'
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# optimizer = optim.Adam([var1, var2], lr=0.0001)
#
# class resnet(nn.Module):
#     def __init__(self, in_channel, out_channel, verbose=False):
#         super(resnet, self).__init__()
#
# model = resnet(1, 110)
#
# loss_func = nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), weight_decay=0.01, lr=0.0001, momentum=0.9)
#
# for epoch in range(1, 81):
#     train_loss, valid_loss, test_loss = [], [], []
#
#     for i, (x, y) in enumerate(train_dl):
#         optimizer.zero_grad()
#         x = x.unsqueeze(1).to(device)
#         y = y.unsqueeze(1).to(device)
#
#         # 1. forward propagation
#         y_pred = model(x)
#
#         y_pred = y_pred.unsqueeze(1)
#
#         # 2. loss calculation
#         loss = loss_func(y_pred, y)
#
#         # 3. backward propagation
#         loss.backward()
#
#         # 4. weight optimization
#         optimizer.step()
#
#         train_loss.append(loss.item())
#
# def initial(self, para_value_tensor, gt_gprMaxin_file):
#
#     return gprMax_infile
# def residual_observation(self, ):
#     return res_obs
