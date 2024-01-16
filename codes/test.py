import os.path as osp
import logging
import time
import argparse
from collections import OrderedDict
import torch
import torch.nn.functional as F
import numpy as np
import options.options as option
import utils.util as util
from data.util import bgr2ycbcr
from data import create_dataset, create_dataloader
from models import create_model
import cv2
import math



#### options
parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, default='options/test/test_smoothing.yml')
opt = option.parse(parser.parse_args().opt, is_train=False)
opt = option.dict_to_nonedict(opt)

util.mkdirs(
    (path for key, path in opt['path'].items()
     if not key == 'experiments_root' and 'pretrain_model' not in key and 'resume' not in key))
util.setup_logger('base', opt['path']['log'], 'test_' + opt['name'], level=logging.INFO,
                  screen=True, tofile=True)
logger = logging.getLogger('base')
logger.info(option.dict2str(opt))

#### Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt['datasets'].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
    test_loaders.append(test_loader)

model = create_model(opt)
for test_loader in test_loaders:
    test_set_name = test_loader.dataset.opt['name']
    logger.info('\nTesting [{:s}]...'.format(test_set_name))
    test_start_time = time.time()
    dataset_dir = osp.join(opt['path']['results_root'], test_set_name)
    print(dataset_dir)
    util.mkdir(dataset_dir)

    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    test_results['psnr_y'] = []
    test_results['ssim_y'] = []

    test_results['psnr_lr'] = []
    test_results['ssim_lr'] = []
    test_results['psnr_y_lr'] = []
    test_results['ssim_y_lr'] = []

    for data in test_loader:
        s = opt['img_multiple_of'] 
        data['Noisy'], pad_h, pad_w = util.processImg(data['Noisy'], s)
        model.feed_data(data)
        img_path = data['GT_path'][0]
        img_name = osp.splitext(osp.basename(img_path))[0]

        model.test(pad_h=pad_h, pad_w=pad_w) 
       
        visuals = model.get_current_visuals()

        sr_img = util.tensor2img(visuals['Denoised'])  # uint8
        srgt_img = util.tensor2img(visuals['GT'])  # uint8

        # save images, must use 'png' format to losslessly store images
        save_img_path = osp.join(dataset_dir, img_name + '.png')
        util.save_img(sr_img, save_img_path)

        # calculate PSNR and SSIM
        gt_img = util.tensor2img(visuals['GT'])

        gt_img = gt_img / 255.
        sr_img = sr_img / 255.


        psnr = util.calculate_psnr(sr_img * 255, gt_img * 255)
        ssim = util.calculate_ssim(sr_img * 255, gt_img * 255)
        test_results['psnr'].append(psnr)
        test_results['ssim'].append(ssim)
        torch.cuda.empty_cache()

        logger.info(
                '{:20s} - PSNR: {:.6f} dB; SSIM: {:.6f}.'.
            format(img_name, psnr, ssim))


    # Average PSNR/SSIM results
    ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
    ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])

    logger.info(
            '----Average PSNR/SSIM results for {}----\n\tpsnr: {:.6f} db; ssim: {:.6f}. \n'.format(
            test_set_name, ave_psnr, ave_ssim))

