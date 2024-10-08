import os
import torch
from basicsr.data.data_util import paired_paths_from_folder3, paired_paths_from_lmdb3
from basicsr.data.transforms import augment, paired_random_crop3
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
from torch.utils import data as data
from torchvision.transforms.functional import normalize


@DATASET_REGISTRY.register()
class GANAddlhPairedDataset(data.Dataset):
    def __init__(self, opt):
        super(GANAddlhPairedDataset).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        # mean and std for normalizing the input images
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder, self.lh_folder = opt['dataroot_gt'], opt['dataroot_lq'], opt['dataroot_lh']
        self.filename_tmpl = opt['filename_tmpl'] if 'filename_tmpl' in opt else '{}'

        # file client (lmdb io backend)
        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder, self.lh_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt', 'lh']
            self.paths = paired_paths_from_lmdb3([self.lq_folder, self.gt_folder, self.lh_folder], ['lq', 'gt', 'lh'])
        elif 'meta_info' in self.opt and self.opt['meta_info'] is not None:
            # disk backend with meta_info
            # Each line in the meta_info describes the relative path to an image
            with open(self.opt['meta_info']) as fin:
                paths = [line.strip() for line in fin]
            self.paths = []
            for path in paths:
                gt_path, lq_path, lh_path = path.split(', ')
                gt_path = os.path.join(self.gt_folder, gt_path)
                lq_path = os.path.join(self.lq_folder, lq_path)
                lh_path = os.path.join(self.lh_folder, lh_path)
                self.paths.append(dict([('gt_path', gt_path), ('lq_path', lq_path), ('lh_path', lh_path)]))
        else:
            # disk backend
            # it will scan the whole folder to get meta info
            # it will be time-consuming for folders with too many files. It is recommended using an extra meta txt file
            self.paths = paired_paths_from_folder3([self.lq_folder, self.gt_folder, self.lh_folder], ['lq', 'gt', 'lh'], self.filename_tmpl)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)
        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)
        lh_path = self.paths[index]['lh_path']
        img_bytes = self.file_client.get(lh_path, 'lh')
        img_lh = imfrombytes(img_bytes, float32=True)

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # random crop
            img_gt, img_lq, img_lh = paired_random_crop3(img_gt, img_lq, img_lh, gt_size, scale, gt_path)
            # flip, rotation
            img_gt, img_lq, img_lh = augment([img_gt, img_lq, img_lh], self.opt['use_hflip'], self.opt['use_rot'])

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq, img_lh = img2tensor([img_gt, img_lq, img_lh], bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
            normalize(img_lh, self.mean, self.std, inplace=True)

        if self.opt['use_grayatten']:
            r,g,b = img_lq[0]+1, img_lq[1]+1, img_lq[2]+1
            A_gray = 1. - (0.299*r+0.587*g+0.114*b)/2.
            A_gray = torch.unsqueeze(A_gray, 0)

            return {'lq': img_lq, 'gt': img_gt, 'lh': img_lh, 'gray': A_gray, 'lq_path': lq_path, 'gt_path': gt_path, 'lh_path': lh_path}
        else:
            return {'lq': img_lq, 'gt': img_gt, 'lh': img_lh, 'lq_path': lq_path, 'gt_path': gt_path, 'lh_path': lh_path}

    def __len__(self):
        return len(self.paths)