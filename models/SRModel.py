import os
from collections import OrderedDict
# from torchviz import make_dot
# from torch.autograd import Variable

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from utils.tc import summary as tc_summary

from models.modules.loss import TVLoss,discriminator_loss,GeneratorLossW,GANLoss,ms_Loss,NT_Xent

from .networks import create_model
from .base_solver import BaseSolver
from .networks import init_weights
import torch.nn.functional as F
from torchvision.models import resnet50

import torchvision.models as models
from torchsummaryX import summary as summaryX

from ptflops import get_model_complexity_info


class SRModel(BaseSolver):
    def __init__(self, opt):
        super(SRModel, self).__init__(opt)
        self.train_opt = opt['train']
        self.use_curriculum = False
        self.LR = self.Tensor()
        self.HR = self.Tensor()
        self.SR = None

        self.results = {'training_loss': [],
                        'val_loss': [],
                        'psnr': [],
                        'ssim': []}

        # if opt['mode'] == 'sr':
        # self.model = create_model(opt)
        # else:
        # assert 'Invalid opt.mode [%s] for SRModel class!'
        with torch.no_grad():
            self.model = create_model(opt)

        # TODO
        # self.load()

        if self.is_train:
            self.model.train()
            loss_type = self.train_opt['pixel_criterion']
            if loss_type == 'l1':
                size_average = False if self.train_opt['type'].upper() == "SGD" else True
                self.criterion_pix = nn.L1Loss(size_average=size_average)

                # self.criterion_pix = nn.L1Loss()
            elif loss_type == 'l2':
                size_average = False if self.train_opt['type'].upper() == "SGD" else True
                self.criterion_pix = nn.MSELoss(size_average=size_average)

            elif loss_type == 'l2_tv':
                size_average = False if self.train_opt['type'].upper() == "SGD" else True
                self.criterion_pix = nn.MSELoss().to(self.device)
                self.cri_tv = TVLoss().to(self.device)
                self.tvloss_paremater = 1e-5

            elif loss_type == 'l1_tv':
                size_average = False if self.train_opt['type'].upper() == "SGD" else True
                self.criterion_pix = nn.L1Loss()
                self.cri_tv = TVLoss().to()
                self.tvloss_paremater = 1e-5

            elif loss_type == 'l1_ms':
                size_average = False if self.train_opt['type'].upper() == "SGD" else True
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                self.criterion_pix = ms_Loss().to(device)
                # self.cri_tv = TVLoss().to()
                # self.tvloss_paremater = 1e-5


                # self.criterion_pix = nn.MSELoss()
            else:
                raise NotImplementedError('[ERROR] Loss type [%s] is not implemented!' % loss_type)
            #
            # if self.use_gpu:
            #     self.criterion_pix = self.criterion_pix.cuda()
            self.criterion_pix_weight = self.train_opt['pixel_weight']

            weight_decay = self.train_opt['weight_decay_G'] if self.train_opt['weight_decay_G'] else 0
            optim_type = self.train_opt['type'].upper()
            if optim_type == "SGD":
                self.optimizer = optim.SGD(self.model.parameters(), lr=self.train_opt['lr_G'],
                                           momentum=self.train_opt['beta1_G'], weight_decay=weight_decay)
            elif optim_type == "ADAM":
                self.optimizer = optim.Adam(self.model.parameters(), lr=self.train_opt['lr_G'],
                                            weight_decay=weight_decay)
            else:
                raise NotImplementedError('[ERROR] Loss type [%s] is not implemented!' % optim_type)

            if self.train_opt['lr_scheme'].lower() == 'multisteplr':
                self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, self.train_opt['lr_steps'],
                                                                self.train_opt['lr_gamma'])
            else:
                raise NotImplementedError('[ERROR] Only MultiStepLR scheme is supported!')

            self.log_dict = OrderedDict()
            print('[Model Initialized]')

    def name(self):
        return 'SRModel'

    def net_init(self, init_type='kaiming'):
        init_weights(self.model, init_type)

    """
    New Version
    Variable usage is disabled
    """

    def feed_data(self, batch):
        input, target = batch['LR'], batch['HR']
        self.LR.resize_(input.size()).copy_(input)
        self.HR.resize_(target.size()).copy_(target)

    def summary(self, input_size):
        print('========================= Model Summary ========================')
        print(self.model)
        print('================================================================')
        print('Input Size: %s' % str(input_size))
        # tc_summary(self.model, input_size)
        # exit()
        print('================================================================')



        total_num = sum(p.numel() for p in self.model.parameters())
        trainable_num = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('Number of params: %.2fK' % (total_num/1e3))
        print('trainable_num:', trainable_num)
        print('=====================================================================')


    def train_step(self):
        self.optimizer.zero_grad()
        # SR = self.model(self.var_LR)
        # SR = self.model(self.LR)
        loss_type = self.train_opt['pixel_criterion']
        if loss_type == 'l1':
            # SR = self.model(self.var_LR)
            # self.weight11 = nn.Parameter(torch.ones(3))
            # weighti = F.softmax(self.weight11, 0)
            SR = self.model(self.LR)
            loss_pix = self.criterion_pix_weight * self.criterion_pix(SR, self.HR)
            # TODO: skip_threshold
            if loss_pix < self.skip_threshold * self.last_epoch_loss:
                loss_pix.backward()
                # print(self.train_opt['clip_grad'])
                # exit()
                if self.train_opt['clip_grad']:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.train_opt['clip_grad'])
                self.optimizer.step()
            else:
                print('Skip this batch! (Loss: {})'.format(loss_pix))
        elif loss_type == 'l2':
            # SR = self.model(self.var_LR)
            SR = self.model(self.LR)

            loss_pix = self.criterion_pix_weight * self.criterion_pix(SR, self.HR)

            # TODO: skip_threshold
            if loss_pix < self.skip_threshold * self.last_epoch_loss:
                loss_pix.backward()

                if self.train_opt['clip_grad']:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.train_opt['clip_grad'])
                self.optimizer.step()
            else:
                print('Skip this batch! (Loss: {})'.format(loss_pix))
        elif loss_type == 'l1_tv':
            SR = self.model(self.LR)
            #loss_pix0 = self.criterion_pix_weight * self.criterion_pix(SR_M, self.HR)
            loss_pix1 = self.criterion_pix_weight * self.criterion_pix(SR, self.HR)
            loss_pix = loss_pix1 + self.tvloss_paremater*self.cri_tv(SR)

        elif  loss_type == 'l1_ms':
            SR = self.model(self.LR)
            loss_pix = self.criterion_pix(SR, self.HR)
            # TODO: skip_threshold
            if loss_pix < self.skip_threshold * self.last_epoch_loss:
                loss_pix.backward()
                if self.train_opt['clip_grad']:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.train_opt['clip_grad'])
                self.optimizer.step()
            else:
                print('Skip this batch! (Loss: {})'.format(loss_pix))


        loss_step = loss_pix
        return loss_step


    def test(self, use_chop):
        self.model.eval()
        with torch.no_grad():
            if use_chop:
                self.SR = self._overlap_crop_forward(self.LR, use_curriculum=self.use_curriculum)
            else:  # use entire image
                if self.use_curriculum:
                    self.SR = self.model(self.LR)[-1]
                else:
                    self.SR = self.model(self.LR)

        if self.is_train:
            # print(self.SR.shape)
            # print(self.HR.shape)
            loss_pix = self.criterion_pix(self.SR, self.HR)
            return loss_pix

        self.model.eval()

    def _overlap_crop_forward(self, x, shave=10, min_size=160000, use_curriculum=False):
        # print('++++++++++++++++++++++++++++++++++++++++++++++++')
        # print(x.shape)
        # print('++++++++++++++++++++++++++++++++++++++++++++++++')
        n_GPUs = 2  # TODO
        scale = self.scale
        b, c, h, w = x.size()
        h_half, w_half = h // 2, w // 2
        h_size, w_size = h_half + shave, w_half + shave
        #
        # h_size = (h_size //2) * 2
        # w_size = (w_size // 2) * 2

        lr_list = [
            x[:, :, 0:h_size, 0:w_size],
            x[:, :, 0:h_size, (w - w_size):w],
            x[:, :, (h - h_size):h, 0:w_size],
            x[:, :, (h - h_size):h+1, (w - w_size):w+1]]

        if w_size * h_size < min_size:
            sr_list = []
            for i in range(0, 4, n_GPUs):
                lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
                if use_curriculum:
                    sr_batch = self.model(lr_batch)[-1]
                else:
                    sr_batch = self.model(lr_batch)
                sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
        else:
            sr_list = [
                self._overlap_crop_forward(patch, shave=shave, min_size=min_size) \
                for patch in lr_list
            ]

        h, w = scale * h, scale * w


        h_half, w_half = scale * h_half, scale * w_half
        h_size, w_size = scale * h_size, scale * w_size
        shave *= scale

        output = x.new(b, c, h, w)
        output[:, :, 0:h_half, 0:w_half] \
            = sr_list[0][:, :, 0:h_half, 0:w_half]
        output[:, :, 0:h_half, w_half:w] \
            = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
        output[:, :, h_half:h, 0:w_half] \
            = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
        output[:, :, h_half:h, w_half:w] \
            = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

        # print('==============================================================')
        # print('output',output.shape)
        # print('==============================================================')

        return output

    def save(self, epoch, is_best):
        # TODO: best checkpoint
        filename = os.path.join(self.checkpoint_dir, 'checkpoint.pth')
        print('[Saving checkpoint to %s ...]' % filename)
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_prec': self.best_prec,
            'results': self.results
        }
        torch.save(state, filename, _use_new_zipfile_serialization=False)
        if is_best:
            torch.save(state, os.path.join(self.checkpoint_dir, 'best_checkpoint.pth'))
        print(['=> Done.'])

    def load(self):

        checkpoint_path = os.path.join(self.checkpoint_dir, 'best_checkpoint.pth')
        print('[Loading checkpoint from %s...]' % checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch'] + 1  # Because the last state had been saved
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.best_prec = checkpoint['best_prec']
        self.results = checkpoint['results']
        print('=> Done.')

        return start_epoch

    def current_loss(self):
        pass

    def get_current_visual(self, need_HR=True):
        out_dict = OrderedDict()

        out_dict['LR'] = self.LR.data[0].float().cpu()
        out_dict['SR'] = self.SR.data[0].float().cpu()
        if need_HR:
            out_dict['HR'] = self.HR.data[0].float().cpu()
        return out_dict

    def current_learning_rate(self):
        return self.optimizer.param_groups[0]['lr']

    def update_learning_rate(self, epoch):
        self.scheduler.step(epoch)

    # def save_network(self):
    # def load_network(self):
