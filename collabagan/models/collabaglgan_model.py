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
class GLGANOWGNModel(SRGANModel):
    def __init__(self, opt):
        super(GLGANOWGNModel, self).__init__(opt)
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

            # JPEG compression + the final sinc filter
            # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
            # as one operation.
            # We consider two orders:
            #   1. [resize back + sinc filter] + JPEG compression
            #   2. JPEG compression + [resize back + sinc filter]
            # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
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
        super(RealMIRSRGLGANOWGNModel, self).nondist_validation(dataloader, current_iter, tb_logger, save_img)
        self.is_train = True

    def backward_G(self, current_iter, l1_gt, percep_gt, loss=False):
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
            # low light loss
            # color loss
            if self.cri_color:
                self.l_g_color = self.cri_color(self.output)
                self.l_g_total += self.l_g_color
                self.loss_dict['l_g_color'] = self.l_g_color
            # exposure control loss
            if self.cri_exp:
                self.l_g_exp = self.cri_exp(self.output)
                self.l_g_total += self.l_g_exp
                self.loss_dict['l_g_exp'] = self.l_g_exp
            # gan loss
            fake_g_pred = self.net_d(self.output)
            self.l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
            self.l_g_total += self.l_g_gan
            self.loss_dict['l_g_gan'] = self.l_g_gan
            # gan local loss
            if self.opt['train']['local_patch_num'] > 0:
                self.l_g_gan_local = 0
                for i in range(self.opt['train']['local_patch_num']):
                    fake_g_pred_local = self.net_d_local(self.output_patch[i])
                    self.l_g_gan_local += self.cri_gan_local(fake_g_pred_local, True, is_disc=False)
                self.l_g_gan_local = self.l_g_gan_local / float(self.opt['train']['local_patch_num'])
                self.l_g_total += self.l_g_gan_local
                self.loss_dict['l_g_gan_local'] = self.l_g_gan_local

            if loss:
                return self.l_g_total
            self.l_g_total.backward()

    def backward_D_real(self, gan_gt, loss=False):
        # real
        real_d_pred = self.net_d(gan_gt)
        self.l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
        self.loss_dict['l_d_real'] = self.l_d_real
        self.loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
        if loss:
            return self.l_d_real
        self.l_d_real.backward()

    def backward_D_fake(self, loss=False):
        # fake
        fake_d_pred = self.net_d(self.output.detach().clone())  # clone for pt1.9
        self.l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
        self.loss_dict['l_d_fake'] = self.l_d_fake
        self.loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
        if loss:
            return self.l_d_fake
        self.l_d_fake.backward()

    def backward_D_local_real(self, gan_gt_patch, loss=False):
        # real
        self.l_d_local_real = 0
        for i in range(self.opt['train']['local_patch_num']):
            real_d_local_pred = self.net_d_local(gan_gt_patch[i])
            self.l_d_local_real += self.cri_gan_local(real_d_local_pred, True, is_disc=True)
        self.l_d_local_real = self.l_d_local_real / self.opt['train']['local_patch_num']
        self.loss_dict['l_d_local_real'] = self.l_d_local_real
        self.loss_dict['out_d_local_real'] = torch.mean(real_d_local_pred.detach())
        if loss:
            return self.l_d_local_real
        self.l_d_local_real.backward()

    def backward_D_local_fake(self, loss=False):
        # fake
        self.l_d_local_fake = 0
        for i in range(self.opt['train']['local_patch_num']):
            fake_d_local_pred = self.net_d_local(self.output_patch[i].detach().clone())  # clone for pt1.9
            self.l_d_local_fake += self.cri_gan_local(fake_d_local_pred, False, is_disc=True)
        self.l_d_local_fake = self.l_d_local_fake / self.opt['train']['local_patch_num']
        self.loss_dict['l_d_local_fake'] = self.l_d_local_fake
        self.loss_dict['out_d_local_fake'] = torch.mean(fake_d_local_pred.detach())
        if loss:
            return self.l_d_local_fake
        self.l_d_local_fake.backward()

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

        if self.opt['train']['local_patch_num'] > 0:
            self.output_patch = []
            self.gan_gt_patch = []
            w = self.lq.size(3)
            h = self.lq.size(2)
            for i in range(self.opt['train']['local_patch_num']):
                w_offset = random.randint(0, max(0, w - self.opt['train']['local_patch_size'] - 1))
                h_offset = random.randint(0, max(0, h - self.opt['train']['local_patch_size'] - 1))
                self.output_patch.append(self.output[:,:, h_offset*self.opt['scale']:h_offset*self.opt['scale'] + self.opt['train']['local_patch_size']*self.opt['scale'],
                                            w_offset*self.opt['scale']:w_offset*self.opt['scale'] + self.opt['train']['local_patch_size']*self.opt['scale']])
                self.gan_gt_patch.append(gan_gt[:,:, h_offset*self.opt['scale']:h_offset*self.opt['scale'] + self.opt['train']['local_patch_size']*self.opt['scale'],
                                            w_offset*self.opt['scale']:w_offset*self.opt['scale'] + self.opt['train']['local_patch_size']*self.opt['scale']])

        self.optimizer_g.zero_grad()
        self.backward_G(current_iter, l1_gt, percep_gt)

        if self.GN:
            gFyfy = 0
            gfyfy = 0
            F = self.backward_G(current_iter, l1_gt, percep_gt, True)
            f = self.backward_D_real(gan_gt, True)
            f = f + self.backward_D_fake(True)
            if self.opt['train']['local_patch_num'] > 0:
                f = f + self.backward_D_local_real(self.gan_gt_patch, True)
                f = f + self.backward_D_local_fake(True)
                dfyP = torch.autograd.grad(f, self.net_d_local.parameters(), retain_graph=True)
                dFyP = torch.autograd.grad(F, self.net_d_local.parameters(), retain_graph=True)
                for Fy, fy in zip(dFyP, dfyP):
                    gFyfy = gFyfy + torch.sum(Fy * fy)
                    gfyfy = gfyfy + torch.sum(fy * fy) + 1e-10
            else:
                f = f
            dfy = torch.autograd.grad(f, self.net_d.parameters(), retain_graph=True)
            dFy = torch.autograd.grad(F, self.net_d.parameters(), retain_graph=True)
            for Fy, fy in zip(dFy, dfy):
                gFyfy = gFyfy + torch.sum(Fy * fy)
                gfyfy = gfyfy + torch.sum(fy * fy) + 1e-10
            self.GN_loss = -gFyfy.detach() / gfyfy.detach() * f
            self.GN_loss.backward()

        self.optimizer_g.step()



        self.optimizer_d.zero_grad()
        self.backward_D_real(gan_gt)
        self.backward_D_fake()
        if self.opt['train']['local_patch_num'] > 0:
            self.optimizer_d_local.zero_grad()
            self.backward_D_local_real(self.gan_gt_patch)
            self.backward_D_local_fake()
            self.optimizer_d.step()
            self.optimizer_d_local.step()
        else:
            self.optimizer_d.step()

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

        self.log_dict = self.reduce_loss_dict(self.loss_dict)
