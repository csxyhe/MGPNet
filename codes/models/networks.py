import torch
import logging
from models.MGPNet import MGPNet
import math
import sys
sys.path.append('../')
logger = logging.getLogger('base')


####################
# define network
####################
def define_G(opt):
    opt_net = opt['network_G']

    netG = MGPNet(in_channels=opt_net['in_nc'], out_channels=opt_net['out_nc'], size_lr=(opt_net['fine_ws'], opt_net['fine_ws']), size_hr=(opt_net['coarse_ws'], opt_net['coarse_ws']), use_bias=opt_net['use_bias'], factor=opt_net['factor'])


    return netG

