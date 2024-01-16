import random
import numpy as np
import cv2
import lmdb
import torch
import torch.utils.data as data
import data.util as util


class LQGTRNDataset(data.Dataset):
    '''
    Read LQ (Low Quality, here is LR), GT and noisy image pairs.
    If only GT and noisy images are provided, generate LQ image on-the-fly.
    The pair is ensured by 'sorted' function, so please check the name convention.
    '''

    def __init__(self, opt):
        super(LQGTRNDataset, self).__init__()
        self.opt = opt
        self.data_type = self.opt['data_type']
        self.paths_GT, self.paths_Noisy = None, None
        self.sizes_GT, self.sizes_Noisy = None, None
        self.GT_env, self.Noisy_env = None, None  # environment for lmdb

        self.paths_GT, self.sizes_GT = util.get_image_paths(self.data_type, opt['dataroot_GT'])
        self.paths_Noisy, self.sizes_Noisy = util.get_image_paths(self.data_type, opt['dataroot_Noisy'])
        assert self.paths_GT, 'Error: GT path is empty.'
        assert self.paths_Noisy, 'Error: Noisy path is empty.'

    def _init_lmdb(self):
        # https://github.com/chainer/chainermn/issues/129
        self.GT_env = lmdb.open(self.opt['dataroot_GT'], readonly=True, lock=False, readahead=False,
                                meminit=False)
        self.Noisy_env = lmdb.open(self.opt['dataroot_Noisy'], readonly=True, lock=False, readahead=False,
                                meminit=False)

    def __getitem__(self, index):
        # print('fffffff')
        if self.data_type == 'lmdb':
            if (self.GT_env is None) or (self.LQ_env is None) or (self.Noisy_env is None):
                self._init_lmdb()
        GT_path, Noisy_path = None, None
        GT_size = self.opt['GT_size']

        # get GT image
        GT_path = self.paths_GT[index]
        if self.data_type == 'lmdb':
            resolution = [int(s) for s in self.sizes_GT[index].split('_')]
        else:
            resolution = None
        img_GT = util.read_img(self.GT_env, GT_path, resolution)

        # modcrop in the validation / test phase
        if self.opt['phase'] != 'train':
            img_GT = img_GT
        # change color space if necessary
        if self.opt['color']:
            img_GT = util.channel_convert(img_GT.shape[2], self.opt['color'], [img_GT])[0]

        # get Noisy image
        Noisy_path = self.paths_Noisy[index]
        if self.data_type == 'lmdb':
            resolution = [int(s) for s in self.sizes_Noisy[index].split('_')]
        else:
            resolution = None
        img_Noisy = util.read_img(self.Noisy_env, Noisy_path, resolution)

        # modcrop in the validation / test phase
        if self.opt['phase'] != 'train':
            img_Noisy =img_Noisy
        # change color space if necessary
        if self.opt['color']:
            img_Noisy = util.channel_convert(img_Noisy.shape[2], self.opt['color'], [img_Noisy])[0]

        if self.opt['phase'] == 'train':
            # force to 3 channels
            if img_Noisy.ndim == 2:
                img_GT = cv2.cvtColor(img_GT, cv2.COLOR_GRAY2BGR)
                img_Noisy = cv2.cvtColor(img_Noisy, cv2.COLOR_GRAY2BGR)
            # if the image size is too small
            H, W, _ = img_GT.shape
            if H < GT_size or W < GT_size:
                img_GT = cv2.resize(np.copy(img_GT), (GT_size, GT_size),
                                    interpolation=cv2.INTER_LINEAR)
                img_Noisy = cv2.resize(np.copy(img_Noisy), (GT_size, GT_size),
                                    interpolation=cv2.INTER_LINEAR)


            # randomly crop
            rnd_h = random.randint(0, max(0, H - GT_size))
            rnd_w = random.randint(0, max(0, W - GT_size))

            img_GT = img_GT[rnd_h:rnd_h + GT_size, rnd_w:rnd_w + GT_size, :]
            img_Noisy = img_Noisy[rnd_h:rnd_h + GT_size, rnd_w:rnd_w + GT_size, :]
            img_GT, img_Noisy = util.augment([img_GT, img_Noisy], self.opt['use_flip'],
                                          self.opt['use_rot'])

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_GT.shape[2] == 3:
            img_GT = img_GT[:, :, [2, 1, 0]]
            img_Noisy = img_Noisy[:, :, [2, 1, 0]]
        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
        img_Noisy = torch.from_numpy(np.ascontiguousarray(np.transpose(img_Noisy, (2, 0, 1)))).float()

        return {'Noisy':img_Noisy, 'GT': img_GT, 'GT_path': GT_path}

    def __len__(self):
        return len(self.paths_GT)
