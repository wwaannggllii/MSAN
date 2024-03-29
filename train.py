# -*- coding: utf-8 -*-
import argparse, time, os
import random

import torch
import torchvision.utils as thutil
import pandas as pd
from tqdm import tqdm

import options.options as option
from utils import util
from models.SRModel import SRModel1,SRModelGAN,SRModel3
from data import create_dataloader
from data import create_dataset
from data.common import rgb2ycbcr
import cv2
import math


import matplotlib.pyplot as plt
import numpy as np
import scipy.misc as misc
os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_IS"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # os.environ['CUDA_VISIBLE_DEVICES']="1" # You can specify your GPU device here. I failed to perform it by `torch.cuda.set_device()`.
    parser = argparse.ArgumentParser(description='Train Super Resolution Models')
    parser.add_argument('-opt', type=str, required=True, help='Path to options JSON file.')
    opt = option.parse(parser.parse_args().opt)

    if opt['train']['resume'] is False:
        util.mkdir_and_rename(opt['path']['exp_root'])  # rename old experiments if exists
        util.mkdirs((path for key, path in opt['path'].items() if not key == 'exp_root' and \
                     not key == 'pretrain_G' and not key == 'pretrain_D'))
        option.save(opt)
        opt = option.dict_to_nonedict(opt)  # Convert to NoneDict, which return None for missing key.
    else:
        opt = option.dict_to_nonedict(opt)
        if opt['train']['resume_path'] is None:
            raise ValueError("The 'resume_path' does not declarate")

    if opt['exec_debug']:
        NUM_EPOCH = 100
        opt['datasets']['train']['dataroot_HR'] = opt['datasets']['train']['dataroot_HR_debug'] #"./dataset/TrainData/DIV2K_train_HR_sub",
        opt['datasets']['train']['dataroot_LR'] = opt['datasets']['train']['dataroot_LR_debug']#./dataset/TrainData/DIV2K_train_HR_sub_LRx3"

    else:
        NUM_EPOCH = int(opt['train']['num_epochs'])

    # random seed
    seed = opt['train']['manual_seed'] #0
    if seed is None:
        seed = random.randint(1, 10000)
    print("Random Seed: ", seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # create train and val dataloader
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = create_dataset(dataset_opt)
            train_loader = create_dataloader(train_set, dataset_opt)
            print('Number of train images in [%s]: %d' % (dataset_opt['name'], len(train_set)))
        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt)
            print('Number of val images in [%s]: %d' % (dataset_opt['name'], len(val_set)))
        elif phase == 'test':
            pass
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    if train_loader is None:
        raise ValueError("The training data does not exist")

    # TODO: design an exp that can obtain the location of the biggest error
    if opt['mode'] == 'sr':
        solver = SRModel1(opt)
    elif opt['mode'] == 'fi':
        solver = SRModel1(opt)
    elif opt['mode'] == 'srgan':
        solver = SRModelGAN(opt)
    elif opt['mode'] == 'msan':
        solver = SRModel1(opt)
    elif opt['mode'] == 'sr_curriculum':
        solver = SRModelCurriculum(opt)

    solver.summary(train_set[0]['LR'].size())
    solver.net_init()
    print('[Start Training]')

    start_time = time.time()

    start_epoch = 1
    if opt['train']['resume']:
        start_epoch = solver.load()

    for epoch in range(start_epoch, NUM_EPOCH + 1):
        # Initialization
        solver.training_loss = 0.0
        epoch_loss_log = 0.0


        if opt['mode'] == 'sr' or opt['mode'] == 'srgan' or opt['mode'] == 'sr_curriculum' or opt['mode'] == 'fi'or opt['mode'] == 'msan':
            training_results = {'batch_size': 0, 'training_loss': 0.0}
        else:
            pass    # TODO
        train_bar = tqdm(train_loader)


        # Train model
        for iter, batch in enumerate(train_bar):
            solver.feed_data(batch)
            iter_loss = solver.train_step()
            epoch_loss_log += iter_loss.item()
            batch_size = batch['LR'].size(0)
            training_results['batch_size'] += batch_size

            if opt['mode'] == 'sr':
                training_results['training_loss'] += iter_loss * batch_size
                train_bar.set_description(desc='[%d/%d] Loss: %.4f ' % (
                    epoch, NUM_EPOCH, iter_loss))
            elif opt['mode'] == 'srgan':
                training_results['training_loss'] += iter_loss * batch_size
                train_bar.set_description(desc='[%d/%d] Loss: %.4f ' % (
                    epoch, NUM_EPOCH, iter_loss))
            elif opt['mode'] == 'fi':
                training_results['training_loss'] += iter_loss * batch_size
                train_bar.set_description(desc='[%d/%d] Loss: %.4f ' % (
                    epoch, NUM_EPOCH, iter_loss))
            elif opt['mode'] == 'msan':
                training_results['training_loss'] += iter_loss * batch_size
                train_bar.set_description(desc='[%d/%d] Loss: %.4f ' % (
                    epoch, NUM_EPOCH, iter_loss))
            elif opt['mode'] == 'sr_curriculum':
                training_results['training_loss'] += iter_loss.data * batch_size
                train_bar.set_description(desc='[%d/%d] Loss: %.4f ' % (
                    epoch, NUM_EPOCH, iter_loss))
            else:
                pass    # TODO

        solver.last_epoch_loss = epoch_loss_log / (len(train_bar))

        train_bar.close()
        time_elapse = time.time() - start_time
        start_time = time.time()
        print('Train Loss: %.4f' % (training_results['training_loss'] / training_results['batch_size']))

        # validate
        val_results = {'batch_size': 0, 'val_loss': 0.0, 'psnr': 0.0, 'ssim': 0.0}

        if epoch % solver.val_step == 0 and epoch != 0:
            print('[Validating...]')
            start_time = time.time()
            solver.val_loss = 0.0

            vis_index = 1

            for iter, batch in enumerate(val_loader):
                visuals_list = []

                solver.feed_data(batch)
                iter_loss = solver.test(opt['chop'])
                batch_size = batch['LR'].size(0)
                val_results['batch_size'] += batch_size

                visuals = solver.get_current_visual()   # float cpu tensor

                sr_img = np.transpose(util.quantize(visuals['SR'], opt['rgb_range']).numpy(), (1,2,0)).astype(np.uint8)
                gt_img = np.transpose(util.quantize(visuals['HR'], opt['rgb_range']).numpy(), (1,2,0)).astype(np.uint8)

                # calculate PSNR
                crop_size = opt['scale']
                cropped_sr_img = sr_img[crop_size:-crop_size, crop_size:-crop_size, :]
                cropped_gt_img = gt_img[crop_size:-crop_size, crop_size:-crop_size, :]

                cropped_sr_img = cropped_sr_img / 255.
                cropped_gt_img = cropped_gt_img / 255.
                cropped_sr_img = rgb2ycbcr(cropped_sr_img).astype(np.float32)
                cropped_gt_img = rgb2ycbcr(cropped_gt_img).astype(np.float32)

                ##################################################################################
                # b, r, g = cv2.split(cropped_sr_img)
                #
                # RG = r - g
                # YB = (r + g) / 2 - b
                # m, n, o = np.shape(cropped_sr_img)  # img为三维 rbg为二维 o并未用到
                # K = m * n
                # alpha_L = 0.1
                # alpha_R = 0.1  # 参数α 可调
                # T_alpha_L = math.ceil(alpha_L * K)  # 向上取整 #表示去除区间
                # T_alpha_R = math.floor(alpha_R * K)  # 向下取整
                #
                # RG_list = RG.flatten()  # 二维数组转一维（方便计算）
                # RG_list = sorted(RG_list)  # 排序
                # sum_RG = 0  # 计算平均值
                # for i in range(T_alpha_L + 1, K - T_alpha_R):
                #     sum_RG = sum_RG + RG_list[i]
                # U_RG = sum_RG / (K - T_alpha_R - T_alpha_L)
                # squ_RG = 0  # 计算方差
                # for i in range(K):
                #     squ_RG = squ_RG + np.square(RG_list[i] - U_RG)
                # sigma2_RG = squ_RG / K
                #
                # # YB和RG计算一样
                # YB_list = YB.flatten()
                # YB_list = sorted(YB_list)
                # sum_YB = 0
                # for i in range(T_alpha_L + 1, K - T_alpha_R):
                #     sum_YB = sum_YB + YB_list[i]
                # U_YB = sum_YB / (K - T_alpha_R - T_alpha_L)
                # squ_YB = 0
                # for i in range(K):
                #     squ_YB = squ_YB + np.square(YB_list[i] - U_YB)
                # sigma2_YB = squ_YB / K
                #
                # uicm = -0.0268 * np.sqrt(np.square(U_RG) + np.square(U_YB)) + 0.1586 * np.sqrt(sigma2_RG + sigma2_RG)
                ##################################################################################

                val_results['val_loss'] += iter_loss * batch_size

                val_results['psnr'] += util.calc_psnr(cropped_sr_img * 255, cropped_gt_img * 255)
                val_results['ssim'] += util.compute_ssim1(cropped_sr_img * 255, cropped_gt_img * 255)

                if opt['mode'] == 'srgan':
                    pass    # TODO

                # if opt['save_image']:
                #     visuals_list.extend([util.quantize(visuals['HR'].squeeze(0), opt['rgb_range']),
                #                          util.quantize(visuals['SR'].squeeze(0), opt['rgb_range'])])
                #
                #     images = torch.stack(visuals_list)
                #     img = thutil.make_grid(images, nrow=2, padding=5)
                #     ndarr = img.byte().permute(1, 2, 0).numpy()
                #     misc.imsave(os.path.join(solver.vis_dir, 'epoch_%d_%d.png' % (epoch, vis_index)), ndarr)
                #     vis_index += 1

            avg_psnr = val_results['psnr']/val_results['batch_size']
            avg_ssim = val_results['ssim']/val_results['batch_size']
            print('Valid Loss: %.4f | Avg. PSNR: %.4f | Avg. SSIM: %.4f | Learning Rate: %f'%(val_results['val_loss']/val_results['batch_size'], avg_psnr, avg_ssim, solver.current_learning_rate()))

            time_elapse = start_time - time.time()

            #if epoch%solver.log_step == 0 and epoch != 0:
            # tensorboard visualization
            solver.training_loss = training_results['training_loss'] / training_results['batch_size']
            solver.val_loss = val_results['val_loss'] / val_results['batch_size']

            solver.tf_log(epoch)

            # statistics
            if opt['mode'] == 'sr' or opt['mode'] == 'srgan' or opt['mode'] == 'sr_curriculum' or opt['mode'] == 'fi'or opt['mode'] == 'msan':
                solver.results['training_loss'].append(solver.training_loss.cpu().data.item())
                solver.results['val_loss'].append(solver.val_loss.cpu().data.item())
                solver.results['psnr'].append(avg_psnr)
                solver.results['ssim'].append(avg_ssim)
            else:
                pass    # TODO

            is_best = False
            if solver.best_prec < solver.results['psnr'][-1]:
                solver.best_prec = solver.results['psnr'][-1]
                is_best = True

            print('#############################################################')
            print(solver.best_prec)
            print(solver.results['psnr'][-1])
            print('***************************************************************')
            # print(is_best)
            # print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            # print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            solver.save(epoch, is_best)

        # update lr
        solver.update_learning_rate(epoch)

    data_frame = pd.DataFrame(
        data={'training_loss': solver.results['training_loss']
            , 'val_loss': solver.results['val_loss']
            , 'psnr': solver.results['psnr']
            , 'ssim': solver.results['ssim']
              },
        index=range(1, NUM_EPOCH+1)
    )
    data_frame.to_csv(os.path.join(solver.results_dir, 'train_results.csv'),
                      index_label='Epoch')


if __name__ == '__main__':
    main()
#python train1.py -opt options/train/train_drudn.json
#python train1.py -opt options/train/EN.json
#python train1.py -opt options/train/msan.json

#python test.py -opt options/test/test_drudn.json
#python heatmaptest.py -opt options/test/test_drudn.json
#python heatmap_dct55.py -opt options/test/test_drudn.json