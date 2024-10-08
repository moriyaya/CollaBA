import numpy as np
import random
import torch
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.data.transforms import paired_random_crop
from basicsr.models.srgan_model import SRGANModel
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.utils.registry import MODEL_REGISTRY
from collections import OrderedDict
from torch.nn import functional as F
from ipdb import set_trace as st


@MODEL_REGISTRY.register()
class PATCHGANOWGNModel(SRGANModel):
    def __init__(self, opt):
        super(PATCHGANOWGNModel, self).__init__(opt)
        self.jpeger = DiffJPEG(differentiable=False).cuda()  # simulate JPEG compression artifacts
        self.usm_sharpener = USMSharp().cuda()  # do usm sharpening
        self.queue_size = opt.get('queue_size', 180)
        self.GN = opt.get('GN')
        self.use_grayatten = opt['datasets']['train'].get('use_grayatten')

    @torch.no_grad()
    def _dequeue_and_enqueue(self):
        """It is the training pair pool for increasing the diversity in a batch.

        Batch processing limits the diversity of synthetic degradations in a batch. For example, samples in a
        batch could not have different resize scaling factors. Therefore, we employ this training pair pool
        to increase the degradation diversity in a batch.
        """
        # initialize
        b, c, h, w = self.lq.size()
        if not hasattr(self, 'queue_lr'):
            assert self.queue_size % b == 0, f'queue size {self.queue_size} should be divisible by batch size {b}'
            self.queue_lr = torch.zeros(self.queue_size, c, h, w).cuda()
            _, c, h, w = self.gt.size()
            self.queue_gt = torch.zeros(self.queue_size, c, h, w).cuda()
            self.queue_ptr = 0
        if self.queue_ptr == self.queue_size:  # the pool is full
            # do dequeue and enqueue
            # shuffle
            idx = torch.randperm(self.queue_size)
            self.queue_lr = self.queue_lr[idx]
            self.queue_gt = self.queue_gt[idx]
            # get first b samples
            lq_dequeue = self.queue_lr[0:b, :, :, :].clone()
            gt_dequeue = self.queue_gt[0:b, :, :, :].clone()
            # update the queue
            self.queue_lr[0:b, :, :, :] = self.lq.clone()
            self.queue_gt[0:b, :, :, :] = self.gt.clone()

            self.lq = lq_dequeue
            self.gt = gt_dequeue
        else:
            # only do enqueue
            self.queue_lr[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.lq.clone()
            self.queue_gt[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.gt.clone()
            self.queue_ptr = self.queue_ptr + b

    @torch.no_grad()
    def feed_data(self, data):
        """Accept data from dataloader, and then add two-order degradations to obtain LQ images.
        """
        if self.is_train and self.opt.get('high_order_degradation', True):
            # training data synthesis
            self.gt = data['gt'].to(self.device)
            self.gt_usm = self.usm_sharpener(self.gt)

            self.kernel1 = data['kernel1'].to(self.device)
            self.kernel2 = data['kernel2'].to(self.device)
            self.sinc_kernel = data['sinc_kernel'].to(self.device)

            ori_h, ori_w = self.gt.size()[2:4]

            # ----------------------- The first degradation process ----------------------- #
            # blur
            out = filter2D(self.gt_usm, self.kernel1)
            # random resize
            updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob'])[0]
            if updown_type == 'up':
                scale = np.random.uniform(1, self.opt['resize_range'][1])
            elif updown_type == 'down':
                scale = np.random.uniform(self.opt['resize_range'][0], 1)
            else:
                scale = 1
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, scale_factor=scale, mode=mode)
            # add noise
            gray_noise_prob = self.opt['gray_noise_prob']
            if np.random.uniform() < self.opt['gaussian_noise_prob']:
                out = random_add_gaussian_noise_pt(
                    out, sigma_range=self.opt['noise_range'], clip=True, rounds=False, gray_prob=gray_noise_prob)
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=self.opt['poisson_scale_range'],
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False)
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range'])
            out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
            out = self.jpeger(out, quality=jpeg_p)

            # ----------------------- The second degradation process ----------------------- #
            # blur
            if np.random.uniform() < self.opt['second_blur_prob']:
                out = filter2D(out, self.kernel2)
            # random resize
            updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob2'])[0]
            if updown_type == 'up':
                scale = np.random.uniform(1, self.opt['resize_range2'][1])
            elif updown_type == 'down':
                scale = np.random.uniform(self.opt['resize_range2'][0], 1)
            else:
                scale = 1
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(
                out, size=(int(ori_h / self.opt['scale'] * scale), int(ori_w / self.opt['scale'] * scale)), mode=mode)
            # add noise
            gray_noise_prob = self.opt['gray_noise_prob2']
            if np.random.uniform() < self.opt['gaussian_noise_prob2']:
                out = random_add_gaussian_noise_pt(
                    out, sigma_range=self.opt['noise_range2'], clip=True, rounds=False, gray_prob=gray_noise_prob)
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=self.opt['poisson_scale_range2'],
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False)


            if np.random.uniform() < 0.5:
                # resize back + the final sinc filter
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
                out = filter2D(out, self.sinc_kernel)
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
            else:
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
                # resize back + the final sinc filter
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
                out = filter2D(out, self.sinc_kernel)

            # clamp and round
            self.lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.

            # random crop
            gt_size = self.opt['gt_size']
            (self.gt, self.gt_usm), self.lq = paired_random_crop([self.gt, self.gt_usm], self.lq, gt_size,
                                                                 self.opt['scale'])

            # training pair pool
            self._dequeue_and_enqueue()
            # sharpen self.gt again, as we have changed the self.gt with self._dequeue_and_enqueue
            self.gt_usm = self.usm_sharpener(self.gt)
            self.lq = self.lq.contiguous()  # for the warning: grad and param do not obey the gradient layout contract
        else:
            # for paired training or validation
            self.lq = data['lq'].to(self.device)
            if 'gt' in data:
                self.gt = data['gt'].to(self.device)
                self.gt_usm = self.usm_sharpener(self.gt)
            if 'gray' in data:
                self.atten = data['gray'].to(self.device)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        # do not use the synthetic process during validation
        self.is_train = False
        super(PATCHGANOWGNModel, self).nondist_validation(dataloader, current_iter, tb_logger, save_img)
        self.is_train = True


    def backward_G(self, current_iter, l1_gt, percep_gt, gan_gt, loss=False):
        self.l_g_total = 0
        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
            # pixel loss
            if self.cri_pix:
                self.l_g_pix = self.cri_pix(self.output, l1_gt)
                self.l_g_total += self.l_g_pix
                self.loss_dict['l_g_pix'] = self.l_g_pix
            # perceptual loss
            if self.cri_perceptual:
                self.l_g_percep, self.l_g_style = self.cri_perceptual(self.output, percep_gt)
                if self.l_g_percep is not None:
                    self.l_g_total += self.l_g_percep
                    self.loss_dict['l_g_percep'] = self.l_g_percep
                if self.l_g_style is not None:
                    self.l_g_total += self.l_g_style
                    self.loss_dict['l_g_style'] = self.l_g_style
            # gan loss
            fake_g_pred = self.net_d(self.output)
            if self.opt['train']['gan_opt']['use_ragan']:
                real_pred = self.net_d(gan_gt)
                self.l_g_gan = (self.cri_gan((real_pred - torch.mean(fake_g_pred)), False) + self.cri_gan((fake_g_pred - torch.mean(real_pred)), True)) / 2
            else:
                self.l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
            self.l_g_total += self.l_g_gan
            self.loss_dict['l_g_gan'] = self.l_g_gan
            # patch gan loss
            if self.opt['train']['gan_opt']['use_patch']:
                fake_patch_g_pred = self.net_d_patch(self.output_patch)
                self.l_g_gan_patch = self.cri_gan(fake_patch_g_pred, True)
            if self.opt['train']['gan_opt']['patchD_3'] > 0:
                for i in range(self.opt['train']['gan_opt']['patchD_3']):
                    fake_patch_g_pred_1 = self.net_d_patch(self.output_patch_1[i])
                    self.l_g_gan_patch += self.cri_gan(fake_patch_g_pred_1, True)
                self.l_g_gan_patch = self.l_g_gan_patch / float(self.opt['train']['gan_opt']['patchD_3'] + 1)
                self.l_g_total += self.l_g_gan_patch
            else:
                self.l_g_total += self.l_g_gan_patch
            self.loss_dict['l_g_gan_patch'] = self.l_g_gan_patch

            if loss:
                return self.l_g_total
            self.l_g_total.backward()


    def backward_D_basic(self, netD, real, fake, use_ragan):
        pred_real = netD(real)
        pred_fake = netD(fake.detach())
        if self.opt['train']['gan_opt']['use_ragan'] and use_ragan:
            loss_D = (self.cri_gan(pred_real - torch.mean(pred_fake), True) +
                                      self.cri_gan(pred_fake - torch.mean(pred_real), False)) / 2
        else:
            loss_D_real = self.cri_gan(pred_real, True)
            loss_D_fake = self.cri_gan(pred_fake, False)
            loss_D = (loss_D_real + loss_D_fake) * 0.5
        return loss_D


    def backward_D(self, gan_gt, loss=False):
        self.l_d = self.backward_D_basic(self.net_d, gan_gt, self.output, True)
        self.loss_dict['l_d'] = self.l_d
        if loss:
            return self.l_d
        self.l_d.backward()


    def backward_D_P(self, loss=False):
        self.l_d_p = self.backward_D_basic(self.net_d_patch, self.gan_gt_patch, self.output_patch, False)
        if self.opt['train']['gan_opt']['patchD_3'] > 0:
            # st()
            for i in range(self.opt['train']['gan_opt']['patchD_3']):
                self.l_d_p += self.backward_D_basic(self.net_d_patch, self.gan_gt_patch_1[i], self.output_patch_1[i], False)
            self.l_d_p = self.l_d_p / float(self.opt['train']['gan_opt']['patchD_3'] + 1)
        if loss:
            return self.l_d_p
        self.l_d_p.backward()


    def optimize_parameters(self, current_iter):
        # usm sharpening
        l1_gt = self.gt_usm
        percep_gt = self.gt_usm
        gan_gt = self.gt_usm
        if self.opt['l1_gt_usm'] is False:
            l1_gt = self.gt
        if self.opt['percep_gt_usm'] is False:
            percep_gt = self.gt
        if self.opt['gan_gt_usm'] is False:
            gan_gt = self.gt

        # for p in self.net_d.parameters():
        #     p.requires_grad = False

        self.loss_dict = OrderedDict()
        if self.use_grayatten:
            self.output = self.net_g(self.lq, self.atten)
        else:
            self.output = self.net_g(self.lq)

        # patch gan need
        if self.opt['train']['gan_opt']['use_patch']:
            w = self.lq.size(3)
            h = self.lq.size(2)
            w_offset = random.randint(0, max(0, w - self.opt['train']['gan_opt']['patchsize'] - 1))
            h_offset = random.randint(0, max(0, h - self.opt['train']['gan_opt']['patchsize'] - 1))

            self.output_patch = self.output[:,:, h_offset*self.opt['scale']:h_offset*self.opt['scale'] + self.opt['train']['gan_opt']['patchsize']*self.opt['scale'],
                                          w_offset*self.opt['scale']:w_offset*self.opt['scale'] + self.opt['train']['gan_opt']['patchsize']*self.opt['scale']]
            # self.l1_gt_patch = l1_gt[:,:, h_offset*self.opt['scale']:h_offset*self.opt['scale'] + self.opt['train']['gan_opt']['patchsize']*self.opt['scale'],
            #                               w_offset*self.opt['scale']:w_offset*self.opt['scale'] + self.opt['train']['gan_opt']['patchsize']*self.opt['scale']]
            # self.percep_gt_patch = percep_gt[:,:, h_offset*self.opt['scale']:h_offset*self.opt['scale'] + self.opt['train']['gan_opt']['patchsize']*self.opt['scale'],
            #                               w_offset*self.opt['scale']:w_offset*self.opt['scale'] + self.opt['train']['gan_opt']['patchsize']*self.opt['scale']]
            self.gan_gt_patch = gan_gt[:,:, h_offset*self.opt['scale']:h_offset*self.opt['scale'] + self.opt['train']['gan_opt']['patchsize']*self.opt['scale'],
                                          w_offset*self.opt['scale']:w_offset*self.opt['scale'] + self.opt['train']['gan_opt']['patchsize']*self.opt['scale']]
            # self.lq_patch = self.lq[:,:, h_offset:h_offset + self.opt['train']['gan_opt']['patchsize'],
            #                         w_offset:w_offset + self.opt['train']['gan_opt']['patchsize']]

            if self.opt['train']['gan_opt']['patchD_3'] > 0:
                self.output_patch_1 = []
                # self.l1_gt_patch_1 = []
                # self.percep_gt_patch_1 = []
                self.gan_gt_patch_1 = []
                # self.lq_patch_1 = []
                w = self.lq.size(3)
                h = self.lq.size(2)
                for i in range(self.opt['train']['gan_opt']['patchD_3']):
                    w_offset_1 = random.randint(0, max(0, w - self.opt['train']['gan_opt']['patchsize'] - 1))
                    h_offset_1 = random.randint(0, max(0, h - self.opt['train']['gan_opt']['patchsize'] - 1))

                    self.output_patch_1.append(self.output[:,:, h_offset_1*self.opt['scale']:h_offset_1*self.opt['scale'] + self.opt['train']['gan_opt']['patchsize']*self.opt['scale'],
                                          w_offset_1*self.opt['scale']:w_offset_1*self.opt['scale'] + self.opt['train']['gan_opt']['patchsize']*self.opt['scale']])
                    # self.l1_gt_patch_1.append(l1_gt[:,:, h_offset_1*self.opt['scale']:h_offset_1*self.opt['scale'] + self.opt['train']['gan_opt']['patchsize']*self.opt['scale'],
                    #                             w_offset_1*self.opt['scale']:w_offset_1*self.opt['scale'] + self.opt['train']['gan_opt']['patchsize']*self.opt['scale']])
                    # self.percep_gt_patch_1.append(percep_gt[:,:, h_offset_1*self.opt['scale']:h_offset_1*self.opt['scale'] + self.opt['train']['gan_opt']['patchsize']*self.opt['scale'],
                    #                             w_offset_1*self.opt['scale']:w_offset_1*self.opt['scale'] + self.opt['train']['gan_opt']['patchsize']*self.opt['scale']])
                    self.gan_gt_patch_1.append(gan_gt[:,:, h_offset_1*self.opt['scale']:h_offset_1*self.opt['scale'] + self.opt['train']['gan_opt']['patchsize']*self.opt['scale'],
                                                w_offset_1*self.opt['scale']:w_offset_1*self.opt['scale'] + self.opt['train']['gan_opt']['patchsize']*self.opt['scale']])
                    # self.lq_patch_1.append(self.lq[:,:, h_offset_1:h_offset_1 + self.opt['train']['gan_opt']['patchsize'],
                    #                         w_offset_1:w_offset_1 + self.opt['train']['gan_opt']['patchsize']])

        self.optimizer_g.zero_grad()
        self.backward_G(current_iter, l1_gt, percep_gt, gan_gt)

        if self.GN:
            gFyfy = 0
            gfyfy = 0
            F = self.backward_G(current_iter, l1_gt, percep_gt, gan_gt, True)
            f = self.backward_D(gan_gt, True)
            if not self.opt['train']['gan_opt']['use_patch']:
                f = f
            else:
                f = f + self.backward_D_P(True)
                dfyP = torch.autograd.grad(f, self.net_d_patch.parameters(), retain_graph=True)
                dFyP = torch.autograd.grad(F, self.net_d_patch.parameters(), retain_graph=True)
                for Fy, fy in zip(dFyP, dfyP):
                    gFyfy = gFyfy + torch.sum(Fy * fy)
                    gfyfy = gfyfy + torch.sum(fy * fy) + 1e-10

            dFyA = torch.autograd.grad(F, self.net_d.parameters(), retain_graph=True)
            dfyA = torch.autograd.grad(f, self.net_d.parameters(), retain_graph=True)

            for Fy, fy in zip(dFyA, dfyA):
                gFyfy = gFyfy + torch.sum(Fy * fy)
                gfyfy = gfyfy + torch.sum(fy * fy) + 1e-10

            GN_loss = -gFyfy.detach() / gfyfy.detach() * f
            GN_loss.backward()

        self.optimizer_g.step()

        self.optimizer_d.zero_grad()
        self.backward_D(gan_gt)
        if not self.opt['train']['gan_opt']['use_patch']:
            self.optimizer_d.step()
        else:
            self.optimizer_d_patch.zero_grad()
            self.backward_D_P()
            self.optimizer_d.step()
            self.optimizer_d_patch.step()

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

        self.log_dict = self.reduce_loss_dict(self.loss_dict)
