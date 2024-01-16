import logging
from collections import OrderedDict
from re import L

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from torchvision import utils as vutils
import torch.nn.functional as F


logger = logging.getLogger('base')


class DL_Model(BaseModel):
    def __init__(self, opt):
        super(DL_Model, self).__init__(opt)

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']
        test_opt = opt['test']
        logger_opt = opt['logger']
        self.train_opt = train_opt
        self.test_opt = test_opt
        self.logger_opt = logger_opt
        self.netG = networks.define_G(opt).to(self.device)

        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)
        # print network
        self.print_network()
        self.load()

        if self.is_train:
            self.netG.train()

            # loss
            self.L1loss = nn.L1Loss()

            # optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            for k, v in self.netG.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            # self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
            #                                     weight_decay=wd_G,
            #                                     betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'])
            self.optimizers.append(self.optimizer_G)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()

    def feed_data(self, data):
        # self.ref_L = data['LQ'].to(self.device)  # LQ
        self.real_H = data['GT'].to(self.device)  # GT
        self.noisy_H = data['Noisy'].to(self.device)  # Noisy


    def feed_data_noreference(self, data):
        self.noisy_H = data.to(self.device)  # Noisy





    
    def optimize_parameters(self, step):
        self.optimizer_G.zero_grad()
        out = self.netG(x=self.noisy_H)
        self.output = out[0]
        loss = self.L1loss(self.output,self.real_H)
        loss += self.L1loss(out[1], F.interpolate(self.real_H, scale_factor=0.5))
        loss += self.L1loss(out[2], F.interpolate(self.real_H, scale_factor=0.25))
        loss.backward()

        # gradient clipping
        if self.train_opt['gradient_clipping']:
            nn.utils.clip_grad_norm_(self.netG.parameters(), self.train_opt['gradient_clipping'])

        self.optimizer_G.step()

        self.log_dict['loss'] = loss.item()

        if step % self.logger_opt['save_checkpoint_freq'] == 0:
            self.save(step)



    def test(self, pad_h=None, pad_w=None):
        self.input = self.noisy_H
        self.netG.eval()
        with torch.no_grad():
            out = self.netG(self.input)
            h, w = out[0].shape[2], out[0].shape[3]
            self.fake_H = out[0]
            if pad_h is not None:
                self.fake_H = self.fake_H[:, :, :h - pad_h, :]
            if pad_w is not None:
                self.fake_H = self.fake_H[:, :, :, :w - pad_w]
        self.netG.train()


    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['Denoised'] = self.fake_H.detach()[0].float().cpu()
        out_dict['GT'] = self.real_H.detach()[0].float().cpu()
        out_dict['Noisy'] = self.noisy_H.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)


