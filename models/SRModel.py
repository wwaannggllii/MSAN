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
#

from flopth import flopth
from torchstat import stat
#
from ptflops import get_model_complexity_info
import thop
from thop import profile
class SRModel(BaseSolver):
    def __init__(self, opt):
        super(SRModel, self).__init__(opt)
        self.train_opt = opt['train']
        self.use_curriculum = False
        self.LR = self.Tensor()
        self.HR = self.Tensor()
        self.SR = None
        self.SR_M = None
        self.SR_L = None

        self.results = {'training_loss': [],
                        'val_loss': [],
                        'psnr': [],
                        'ssim': [],
                        'psnr_M': [],
                        'ssim_M': [],
                        'psnr_L': [],
                        'ssim_L': []
                        }

        # if opt['mode'] == 'sr':
        # self.model = create_model(opt)
        # else:
        # assert 'Invalid opt.mode [%s] for SRModel class!'
        self.model = create_model(opt)

        # TODO
        # self.load()

        if self.is_train:
            self.model.train()
            loss_type = self.train_opt['pixel_criterion']  # pixel_criterion=L1  ADAM
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
                self.cri_tv = TVLoss()
                self.tvloss_paremater = 1e-5



                # self.criterion_pix = nn.MSELoss()
            else:
                raise NotImplementedError('[ERROR] Loss type [%s] is not implemented!' % loss_type)

            if self.use_gpu:
                self.criterion_pix = self.criterion_pix.cuda()
            self.criterion_pix_weight = self.train_opt['pixel_weight']  # pixel_weight

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

            if self.train_opt['lr_scheme'].lower() == 'multisteplr':  #
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
        tc_summary(self.model, input_size)
        print('================================================================')
        print('==================================================================')

        total_num = sum(p.numel() for p in self.model.parameters())
        trainable_num = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('Number of params: %.2fK' % (total_num / 1e3))
        print('trainable_num:', trainable_num)
        print('=====================================================================')

        model_name = 'yolov3 cut asff'
        flops, params = get_model_complexity_info(self.model, (3,32,32), as_strings=True, print_per_layer_stat=True)
        print("%s |%s |%s" % (model_name, flops, params))

        # stat(self.model, (3,32,32))
        exit()
        # exit()
        # exit()

        # model_name = 'yolov3 cut asff'
        #
        # # flops, params = get_model_complexity_info(self.model, (3,32,32), as_strings=True, print_per_layer_stat=True)
        # #
        # print("%s |%s |%s" % (model_name, flops, params))
        stat(self.model, (3,128,72))
        # exit()



    def train_step(self):
        self.optimizer.zero_grad()
        loss_type = self.train_opt['pixel_criterion']
        if loss_type == 'l1':
            # SR = self.model(self.var_LR)
            # self.weight11 = nn.Parameter(torch.ones(3))
            # weighti = F.softmax(self.weight11, 0)

            SR, SR_M,SR_L = self.model(self.LR)
            loss_pix0 = self.criterion_pix_weight * self.criterion_pix(SR, self.HR)
            loss_pix1 = self.criterion_pix_weight * self.criterion_pix(SR_M, self.HR)
            loss_pix2 = self.criterion_pix_weight * self.criterion_pix(SR_L, self.HR)
            loss_pix = 0.9 * loss_pix0 + 0.05 * loss_pix1 +  0.05 *loss_pix2
            # TODO: skip_threshold
            if loss_pix < self.skip_threshold * self.last_epoch_loss:
                loss_pix.backward()
                if self.train_opt['clip_grad']:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.train_opt['clip_grad'])
                self.optimizer.step()
            else:
                print('Skip this batch! (Loss: {})'.format(loss_pix))
        elif loss_type == 'l2':
            # SR = self.model(self.var_LR)
            SR, SR_M = self.model(self.LR)
            loss_pix0 = self.criterion_pix_weight * self.criterion_pix(SR_M, self.HR)
            loss_pix1 = self.criterion_pix_weight * self.criterion_pix(SR, self.HR)
            loss_pix = 0.8 * loss_pix1 + 0.2 * loss_pix0
            # TODO: skip_threshold
            if loss_pix < self.skip_threshold * self.last_epoch_loss:
                loss_pix.backward()
                if self.train_opt['clip_grad']:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.train_opt['clip_grad'])
                self.optimizer.step()
            else:
                print('Skip this batch! (Loss: {})'.format(loss_pix))
        elif loss_type == 'l1_tv':
            SR, SR_M = self.model(self.LR)
            #loss_pix0 = self.criterion_pix_weight * self.criterion_pix(SR_M, self.HR)
            loss_pix1 = self.criterion_pix_weight * self.criterion_pix(SR, self.HR)
            loss_pix =  loss_pix1 + self.tvloss_paremater*self.cri_tv(SR)
        loss_step = loss_pix
        return loss_step

    def test(self, use_chop):
        self.model.eval()
        with torch.no_grad():
            if use_chop:
                self.SR, self.SR_M, self.SR_L = self._overlap_crop_forward(self.LR, use_curriculum=self.use_curriculum)
            else:  # use entire image
                if self.use_curriculum:
                    self.SR, self.SR_M, self.SR_L = self.model(self.LR)[-1]
                else:
                    self.SR, self.SR_M, self.SR_L = self.model(self.LR)

        if self.is_train:
            loss_pix0 = self.criterion_pix_weight * self.criterion_pix(self.SR_M, self.HR)
            loss_pix1 = self.criterion_pix_weight * self.criterion_pix(self.SR, self.HR)
            loss_pix = loss_pix1
            return loss_pix

        self.model.eval()

    def _overlap_crop_forward(self, x, shave=10, min_size=160000, use_curriculum=False):
        n_GPUs = 1  # TODO
        scale = self.scale
        b, c, h, w = x.size()
        h_half, w_half = h // 2, w // 2
        h_size, w_size = h_half + shave, w_half + shave
        lr_list = [
            x[:, :, 0:h_size, 0:w_size],
            x[:, :, 0:h_size, (w - w_size):w],
            x[:, :, (h - h_size):h, 0:w_size],
            x[:, :, (h - h_size):h, (w - w_size):w]]

        if w_size * h_size < min_size:
            sr_list = []
            sr_list_M = []
            sr_list_L = []
            for i in range(0, 4, n_GPUs):
                lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
                if use_curriculum:
                    sr_batch, sr_batch_M,sr_batch_L = self.model(lr_batch)[-1]
                else:
                    sr_batch, sr_batch_M, sr_batch_L = self.model(lr_batch)
                sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
                sr_list_M.extend(sr_batch_M.chunk(n_GPUs, dim=0))
                sr_list_L.extend(sr_batch_L.chunk(n_GPUs, dim=0))
        else:
            sr_list = [self._overlap_crop_forward(patch, shave=shave, min_size=min_size) \
                       for patch in lr_list]
            sr_list_M = [self._overlap_crop_forward(patch, shave=shave, min_size=min_size) \
                         for patch in lr_list]
            sr_list_L = [self._overlap_crop_forward(patch, shave=shave, min_size=min_size) \
                         for patch in lr_list]

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

        output_M = x.new(b, c, h, w)
        output_M[:, :, 0:h_half, 0:w_half] \
            = sr_list_M[0][:, :, 0:h_half, 0:w_half]
        output_M[:, :, 0:h_half, w_half:w] \
            = sr_list_M[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
        output_M[:, :, h_half:h, 0:w_half] \
            = sr_list_M[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
        output_M[:, :, h_half:h, w_half:w] \
            = sr_list_M[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

        output_L = x.new(b, c, h, w)
        output_L[:, :, 0:h_half, 0:w_half] \
            = sr_list_L[0][:, :, 0:h_half, 0:w_half]
        output_L[:, :, 0:h_half, w_half:w] \
            = sr_list_L[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
        output_L[:, :, h_half:h, 0:w_half] \
            = sr_list_L[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
        output_L[:, :, h_half:h, w_half:w] \
            = sr_list_L[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]


        return output, output_M, output_L

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
        torch.save(state, filename)
        if is_best:
            torch.save(state, os.path.join(self.checkpoint_dir, 'best_checkpoint.pth'))
        print(['=> Done.'])

    def load(self):
        checkpoint_path = os.path.join(self.checkpoint_dir, 'checkpoint.pth')
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
        out_dict = OrderedDict()  # 实现了对字典对象中元素的排序
        out_dict['LR'] = self.LR.data[0].float().cpu()
        out_dict['SR'] = self.SR.data[0].float().cpu()
        out_dict['SR_M'] = self.SR_M.data[0].float().cpu()
        out_dict['SR_L'] = self.SR_L.data[0].float().cpu()
        if need_HR:
            out_dict['HR'] = self.HR.data[0].float().cpu()
        return out_dict

    def current_learning_rate(self):
        return self.optimizer.param_groups[0]['lr']

    def update_learning_rate(self, epoch):
        self.scheduler.step(epoch)

    # def save_network(self):
    # def load_network(self):


# TODO: Can be merged into SRModel
class SRModelCurriculum(SRModel):
    def __init__(self, opt):
        super(SRModelCurriculum, self).__init__(opt)
        self.curriculum_gamma = self.train_opt['curriculum_gamma']
        self.curriculum_weights = self.train_opt['curriculum_weights']
        self.use_curriculum = True

    def name(self):
        return 'SRModelCurriculum'

    def net_init(self, init_type='kaiming'):
        super(SRModelCurriculum, self).net_init()
        # print('[Regular weight initialization done, initializing conv layer before nn.PixelShuffle...]')
        # self.model.module.icnr_weight_init()

    def train_step(self):
        self.optimizer.zero_grad()
        outputs = self.model(self.LR)
        # assert len(outputs)==len(self.curriculum_weights), 'Number of outputs and curriculum weights do not match!'
        # SR = outputs[-1]
        losses_pix = [self.criterion_pix(sr, self.HR) for sr in outputs]
        loss_pix = 0.0
        for it in range(len(losses_pix)):
            loss_pix += (self.curriculum_gamma ** it) * losses_pix[it]
            # loss_pix += self.curriculum_weights[it] * losses_pix[it]
        if loss_pix < self.skip_threshold * self.last_epoch_loss:
            loss_pix.backward(retain_graph=True)
            if self.train_opt['clip_grad'] and (self.train_opt['type'].upper() == "SGD"):
                nn.utils.clip_grad_norm_(self.model.parameters(), self.train_opt['clip_grad'])
            self.optimizer.step()
        else:
            print('Skip this batch! (Loss: {})'.format(loss_pix))

        loss_step = loss_pix
        # self.model.reset_state()
        return loss_step


class SRModel1(BaseSolver):
    def __init__(self, opt):
        super(SRModel1, self).__init__(opt)
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

        summaryX(self.model, torch.zeros((1,3,426,240)))
        # exit()
        # print('=====================================================================')
        # input = torch.randn(1,3,64,36)
        # flops, params = profile(self.model, inputs=(input,))
        # print("FLOPs :{:.4f}, Params : {:.4f}".format(flops / 1e9, params / 1e6))  # flops单位G，para单位M
        # exit()

        #
        model_name = 'MSAN'
        flops, params = get_model_complexity_info(self.model, (3,426,240), as_strings=True, print_per_layer_stat=True)
        print("%s |%s |%s" % (model_name, flops, params))
        print('=====================================================================')
        # print(flopth(self.model, in_size=[3, 64, 64]))
        # exit()
        # # #
        # # # # print('=====================================================================')
        # stat(self.model, (3,42,24))
        # exit()
        # for p in self.model.named_parameters():
        #     print(len(p))
        #     print(p)



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



class SRModel2(BaseSolver):
    def __init__(self, opt):
        super(SRModel2, self).__init__(opt)
        self.train_opt = opt['train']
        self.use_curriculum = False
        self.LR = self.Tensor()
        self.HR = self.Tensor()
        self.SR = None
        self.SR1 = None
        self.SR2 = None
        self.SR3 = None


        self.results = {'training_loss': [],
                        'val_loss': [],
                        'psnr': [],
                        'ssim': []}

        # if opt['mode'] == 'sr':
        # self.model = create_model(opt)
        # else:
        # assert 'Invalid opt.mode [%s] for SRModel class!'
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
                self.cri_pix = nn.MSELoss().to(self.device)
                self.cri_tv = TVLoss().to(self.device)
                self.tvloss_paremater = 1e-5



                # self.criterion_pix = nn.MSELoss()
            else:
                raise NotImplementedError('[ERROR] Loss type [%s] is not implemented!' % loss_type)

            if self.use_gpu:
                self.criterion_pix = self.criterion_pix.cuda()
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
        tc_summary(self.model, input_size)
        print('================================================================')

        print('==================================================================')

        total_num = sum(p.numel() for p in self.model.parameters())
        trainable_num = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('Number of params: %.2fK' % (total_num/1e3))
        print('trainable_num:', trainable_num)
        print('=====================================================================')
        # exit()




    def train_step(self):
        self.optimizer.zero_grad()
        # SR = self.model(self.var_LR)
        SR, SR1, SR2, SR3 = self.model(self.LR)
        loss_pix0 = self.criterion_pix_weight * self.criterion_pix(SR, self.HR)
        # loss_pix1 = self.criterion_pix_weight * self.criterion_pix(SR1, self.HR)
        # loss_pix2 = self.criterion_pix_weight * self.criterion_pix(SR2, self.HR)
        # loss_pix3 = self.criterion_pix_weight * self.criterion_pix(SR3, self.HR)
        loss_pix1 = SR1
        loss_pix2 = SR2
        loss_pix3 = SR3
        loss_pix = loss_pix0 #+ 0.01*loss_pix1+ 0.01*loss_pix2+ 0.01*loss_pix3
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
                self.SR, self.SR1, self.SR2, self.SR3 = self._overlap_crop_forward(self.LR, use_curriculum=self.use_curriculum)
            else:  # use entire image
                if self.use_curriculum:
                    self.SR, self.SR1, self.SR2, self.SR3 = self.model(self.LR)[-1]
                else:
                    self.SR, self.SR1, self.SR2, self.SR3 = self.model(self.LR)

        if self.is_train:
            loss_pix = self.criterion_pix_weight * self.criterion_pix(self.SR, self.HR)
            return loss_pix

        self.model.eval()

    def _overlap_crop_forward(self, x, shave=10, min_size=160000, use_curriculum=False):
        n_GPUs = 2  # TODO
        scale = self.scale
        b, c, h, w = x.size()
        h_half, w_half = h // 2, w // 2
        h_size, w_size = h_half + shave, w_half + shave
        lr_list = [
            x[:, :, 0:h_size, 0:w_size],
            x[:, :, 0:h_size, (w - w_size):w],
            x[:, :, (h - h_size):h, 0:w_size],
            x[:, :, (h - h_size):h, (w - w_size):w]]

        if w_size * h_size < min_size:
            sr_list = []
            sr_list1 = []
            sr_list2 = []
            sr_list3 = []
            for i in range(0, 4, n_GPUs):
                lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
                if use_curriculum:
                    sr_batch,sr_batch1,sr_batch2,sr_batch3 = self.model(lr_batch)[-1]
                else:
                    sr_batch, sr_batch1, sr_batch2, sr_batch3 = self.model(lr_batch)
                sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
                sr_list1.extend(sr_batch.chunk(n_GPUs, dim=0))
                sr_list2.extend(sr_batch.chunk(n_GPUs, dim=0))
                sr_list3.extend(sr_batch.chunk(n_GPUs, dim=0))
        else:
            sr_list = [self._overlap_crop_forward(patch, shave=shave, min_size=min_size) for patch in lr_list  ]
            sr_list1 = [self._overlap_crop_forward(patch, shave=shave, min_size=min_size) for patch in lr_list ]
            sr_list2 = [self._overlap_crop_forward(patch, shave=shave, min_size=min_size) for patch in lr_list ]
            sr_list3 = [ self._overlap_crop_forward(patch, shave=shave, min_size=min_size) for patch in lr_list]

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

        output1 = x.new(b, c, h, w)
        output1[:, :, 0:h_half, 0:w_half] \
            = sr_list1[0][:, :, 0:h_half, 0:w_half]
        output1[:, :, 0:h_half, w_half:w] \
            = sr_list1[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
        output1[:, :, h_half:h, 0:w_half] \
            = sr_list1[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
        output1[:, :, h_half:h, w_half:w] \
            = sr_list1[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

        output2 = x.new(b, c, h, w)
        output2[:, :, 0:h_half, 0:w_half] \
            = sr_list2[0][:, :, 0:h_half, 0:w_half]
        output2[:, :, 0:h_half, w_half:w] \
            = sr_list2[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
        output2[:, :, h_half:h, 0:w_half] \
            = sr_list2[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
        output2[:, :, h_half:h, w_half:w] \
            = sr_list2[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]


        output3 = x.new(b, c, h, w)
        output3[:, :, 0:h_half, 0:w_half] \
            = sr_list3[0][:, :, 0:h_half, 0:w_half]
        output3[:, :, 0:h_half, w_half:w] \
            = sr_list3[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
        output3[:, :, h_half:h, 0:w_half] \
            = sr_list3[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
        output3[:, :, h_half:h, w_half:w] \
            = sr_list3[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

        return output, output1, output2, output3

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
        torch.save(state, filename)
        if is_best:
            torch.save(state, os.path.join(self.checkpoint_dir, 'best_checkpoint.pth'))
        print(['=> Done.'])

    def load(self):
        checkpoint_path = os.path.join(self.checkpoint_dir, 'checkpoint.pth')
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
        out_dict['SR1'] = self.SR1.data[0].float().cpu()
        out_dict['SR2'] = self.SR2.data[0].float().cpu()
        out_dict['SR3'] = self.SR3.data[0].float().cpu()
        if need_HR:
            out_dict['HR'] = self.HR.data[0].float().cpu()
        return out_dict

    def current_learning_rate(self):
        return self.optimizer.param_groups[0]['lr']

    def update_learning_rate(self, epoch):
        self.scheduler.step(epoch)

    # def save_network(self):
    # def load_network(self):


class SRModelGAN(BaseSolver):
    def __init__(self, opt):
        super(SRModelGAN, self).__init__(opt)
        self.train_opt = opt['train']
        self.use_curriculum = False
        self.LR = self.Tensor()
        self.HR = self.Tensor()
        self.SR = None

        self.results = {'training_loss': [],
                        'val_loss': [],
                        'psnr': [],
                        'ssim': []}

        self.cri_gan = GANLoss(opt['train']['gan_type'], 1.0, 0.0)


        # if opt['mode'] == 'sr':
        # self.model = create_model(opt)
        # else:
        # assert 'Invalid opt.mode [%s] for SRModel class!'
        with torch.no_grad():
            self.model_g,self.model_d = create_model(opt)

        # TODO
        # self.load()

        if self.is_train:
            self.model_g.train()
            self.model_d.train()
            self.g_loss_module = GeneratorLossW(**vars(opt))


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
                self.optimizer_d = optim.Adam(self.model_d.parameters(), lr=self.train_opt['lr_G'],
                                            weight_decay=weight_decay)
                self.optimizer_g = optim.Adam(self.model_g.parameters(), lr=self.train_opt['lr_G'],
                                            weight_decay=weight_decay)
            else:
                raise NotImplementedError('[ERROR] Loss type [%s] is not implemented!' % optim_type)

            if self.train_opt['lr_scheme'].lower() == 'multisteplr':
                self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer_d, self.train_opt['lr_steps'],
                                                                self.train_opt['lr_gamma'])
                self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer_g, self.train_opt['lr_steps'],
                                                                self.train_opt['lr_gamma'])

            else:
                raise NotImplementedError('[ERROR] Only MultiStepLR scheme is supported!')

            self.log_dict = OrderedDict()
            print('[Model Initialized]')

    def name(self):
        return 'SRModel'

    def net_init(self, init_type='kaiming'):
        init_weights(self.model_g, init_type)
        init_weights(self.model_d, init_type)

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
        print(self.model_g)
        print('================================================================')
        print(self.model_d)
        print('Input Size: %s' % str(input_size))
        # tc_summary(self.model, input_size)
        print('================================================================')
        # exit()


        # total_num = sum(p.numel() for p in self.model.parameters())
        # trainable_num = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        # print('Number of params: %.2fK' % (total_num/1e3))
        # print('trainable_num:', trainable_num)
        # print('=====================================================================')
        # #
        # # summaryX(self.model, torch.zeros((1,3,40,10)))
        # print('=====================================================================')
        # # input = torch.randn(1,3,32,32)
        # macs, params = profile(model, inputs=(input,))
        # print(flops)
        # print(params)
        # print('=====================================================================')

        #
        model_name = 'MSAN'
        flops, params = get_model_complexity_info(self.model_g, (3,42,24), as_strings=True, print_per_layer_stat=True)
        print("%s |%s |%s" % (model_name, flops, params))
        print('=====================================================================')

        # print(flopth(self.model, in_size=[3, 100, 100]))
        # print('=====================================================================')
        # stat(self.model, (3,42,24))
        # exit()


    def train_step(self):
        self.optimizer_g.zero_grad()
        self.optimizer_d.zero_grad()
        # SR = self.model(self.var_LR)
        SR = self.model_g(self.LR)
        loss = self.criterion_pix_weight * self.criterion_pix(SR, self.HR)

        real_tex = self.model_d(self.HR)
        fake_tex = self.model_d(SR)

        rand = torch.rand(1).item()
        sample = rand * self.HR + (1 - rand) * SR
        gp_tex = self.model_d(sample)
        gradient = torch.autograd.grad(gp_tex.mean(), sample, create_graph=True)[0]
        grad_pen = 10 * (gradient.norm() - 1) ** 2
        # update discriminator
        # self.model_d.zero_grad()
        d_tex_loss = discriminator_loss(real_tex, fake_tex, wasserstein=True, grad_penalties=grad_pen)  #GA
        # d_tex_loss.backward(retain_graph=True)
        # self.optimizer_d.step()

        # self.model_g.zero_grad()
        g_loss = self.g_loss_module(self.HR,fake_tex, real_tex)
        # print(g_loss)
        #
        # assert not torch.isnan(g_loss), 'Generator loss returns NaN values'
        # g_loss.backward()
        # self.optimizer_g.step()

        # # save data to tensorboard
        # rgb_loss = self.g_loss_module.rgb_loss(SR, self.HR)
        # mean_loss = self.g_loss_module.mean_loss(SR, self.HR)
####################################对抗损失########################################
        l_d_target_real = self.cri_gan(real_tex - fake_tex.mean(0, keepdim=True), True)
        l_d_target_fake = self.cri_gan(fake_tex - real_tex.mean(0, keepdim=True), False)

        l_d_target_total = (l_d_target_real + l_d_target_fake) / 2
        #
        # self.optimizer_g.zero_grad()
        # l_d_target_total.backward()
        # self.optimizer_g.step()0
        loss_pix = loss + 0.01 * d_tex_loss + 0.001 * g_loss + 0.0005 * l_d_target_total

        # TODO: skip_threshold
        if loss_pix < self.skip_threshold * self.last_epoch_loss:
            loss_pix.backward()
            if self.train_opt['clip_grad']:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.train_opt['clip_grad'])
            self.optimizer_g.step()
            self.optimizer_d.step()

        loss_step = loss_pix
        return loss_step



    def test(self, use_chop):
        self.model_g.eval()
        with torch.no_grad():
            if use_chop:
                self.SR = self._overlap_crop_forward(self.LR, use_curriculum=self.use_curriculum)
            else:  # use entire image
                if self.use_curriculum:
                    self.SR = self.model_g(self.LR)[-1]
                else:
                    self.SR = self.model_g(self.LR)

        if self.is_train:
            # print(self.SR.shape)
            # print(self.HR.shape)
            loss_pix = self.criterion_pix(self.SR, self.HR)
            return loss_pix

        self.model_g.eval()

    def _overlap_crop_forward(self, x, shave=10, min_size=160000, use_curriculum=False):
        # print('++++++++++++++++++++++++++++++++++++++++++++++++')
        # print(x.shape)
        # print('++++++++++++++++++++++++++++++++++++++++++++++++')
        n_GPUs = 2  # TODO
        scale = self.scale
        b, c, h, w = x.size()
        h_half, w_half = h // 2, w // 2
        h_size, w_size = h_half + shave, w_half + shave

        # h_size = (h_size // 8) * 8
        # w_size = (w_size // 8) * 8

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
                    sr_batch = self.model_g(lr_batch)[-1]
                else:
                    sr_batch = self.model_g(lr_batch)
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
            'state_dict': self.model_g.state_dict(),
            'optimizer': self.optimizer_g.state_dict(),
            'best_prec': self.best_prec,
            'results': self.results
        }
        torch.save(state, filename)
        if is_best:
            torch.save(state, os.path.join(self.checkpoint_dir, 'best_checkpoint.pth'))
        print(['=> Done.'])

    def load(self):
        checkpoint_path = os.path.join(self.checkpoint_dir, 'checkpoint.pth')
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
        return self.optimizer_d.param_groups[0]['lr']

    def update_learning_rate(self, epoch):
        self.scheduler.step(epoch)

    # def save_network(self):
    # def load_network(self):




class SRModel3(BaseSolver):
    def __init__(self, opt):
        super(SRModel3, self).__init__(opt)
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

            elif loss_type == 'l1_cl':
                size_average = False if self.train_opt['type'].upper() == "SGD" else True
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                self.criterion_pix0 = nn.L1Loss(size_average=size_average)
                self.criterion_pix1 = NT_Xent(16, 0.5, world_size=1).to(device)
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
        print('================================================================')



        # total_num = sum(p.numel() for p in self.model.parameters())
        # trainable_num = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        # print('Number of params: %.2fK' % (total_num/1e3))
        # print('trainable_num:', trainable_num)
        # print('=====================================================================')
        #
        # summaryX(self.model, torch.zeros((1,3,42,24)))
        # print('=====================================================================')
        # input = torch.randn(1,3,32,32)
        # macs, params = profile(self.model, inputs=(input,))
        # print(macs)
        # print(params)
        # print('=====================================================================')

        #
        model_name = 'MSAN'
        flops, params = get_model_complexity_info(self.model, (3,42,24), as_strings=True, print_per_layer_stat=True)
        print("%s |%s |%s" % (model_name, flops, params))
        # print('=====================================================================')
        # print(flopth(self.model, in_size=[3, 10, 10]))
        # print('=====================================================================')
        # stat(self.model, (3,42,24))
        # exit()



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

        elif loss_type == 'l1_ms':
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

        elif loss_type == 'l1_cl':
            SR, z_i, z_j = self.model(self.LR)

            loss_pix = self.criterion_pix0(SR, self.HR) + self.criterion_pix0(z_i, z_j)
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
                self.SR,z_i, z_j = self._overlap_crop_forward(self.LR, use_curriculum=self.use_curriculum)
            else:  # use entire image
                if self.use_curriculum:
                    self.SR,_,_ = self.model(self.LR)[-1]
                else:
                    self.SR,_,_ = self.model(self.LR)

        if self.is_train:
            # print(self.SR.shape)
            # print(self.HR.shape)
            loss_pix = self.criterion_pix0(self.SR, self.HR)
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
        h_size = (h_size //2) * 2
        w_size = (w_size // 2) * 2

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
                    sr_batch,z_i, z_j = self.model(lr_batch)[-1]
                else:
                    sr_batch,z_i, z_j = self.model(lr_batch)
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

        return output,z_i, z_j

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
        torch.save(state, filename)
        if is_best:
            torch.save(state, os.path.join(self.checkpoint_dir, 'best_checkpoint.pth'))
        print(['=> Done.'])

    def load(self):
        checkpoint_path = os.path.join(self.checkpoint_dir, 'checkpoint.pth')
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


class SRModel4(BaseSolver):
    def __init__(self, opt):
        super(SRModel4, self).__init__(opt)
        self.train_opt = opt['train']
        self.use_curriculum = False
        self.LR = self.Tensor()
        self.HR = self.Tensor()
        self.HRE = self.Tensor()
        self.HRS = self.Tensor()
        self.SR = None
        self.SRE = None
        self.SRS = None

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
                self.criterionCE = nn.CrossEntropyLoss()
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
        tc_summary(self.model, input_size)
        print('================================================================')



        # total_num = sum(p.numel() for p in self.model.parameters())
        # trainable_num = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        # print('Number of params: %.2fK' % (total_num/1e3))
        # print('trainable_num:', trainable_num)
        # print('=====================================================================')
        #
        # summaryX(self.model, torch.zeros((1,3,42,24)))
        # print('=====================================================================')
        # input = torch.randn(1,3,32,32)
        # macs, params = profile(self.model, inputs=(input,))
        # print(macs)
        # print(params)
        # print('=====================================================================')

        #
        model_name = 'MSAN'
        flops, params = get_model_complexity_info(self.model, (3,42,24), as_strings=True, print_per_layer_stat=True)
        print("%s |%s |%s" % (model_name, flops, params))
        # print('=====================================================================')
        # print(flopth(self.model, in_size=[3, 10, 10]))
        # print('=====================================================================')
        # stat(self.model, (3,42,24))
        # exit()



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

        elif loss_type == 'l1_ms':
            SR,SRE,SRS = self.model(self.LR)
            loss_pix = self.criterion_pix(SR, self.HR) + self.criterionCE()
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
        h_size = (h_size //2) * 2
        w_size = (w_size // 2) * 2

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
        torch.save(state, filename)
        if is_best:
            torch.save(state, os.path.join(self.checkpoint_dir, 'best_checkpoint.pth'))
        print(['=> Done.'])

    def load(self):
        checkpoint_path = os.path.join(self.checkpoint_dir, 'checkpoint.pth')
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