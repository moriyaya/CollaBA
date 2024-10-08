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
class GANOWGNModel(SRGANModel):
    def __init__(self, opt):
        super(GANOWGNModel, self).__init__(opt)
        # self.jpeger = DiffJPEG(differentiable=False).cuda()  # simulate JPEG compression artifacts
        self.usm_sharpener = USMSharp().cuda()  # do usm sharpening
        self.queue_size = opt.get('queue_size', 180)
        self.GN = opt.get('GN')
        self.use_grayatten = opt['datasets']['train'].get('use_grayatten')
        self.use_aw = opt.get('use_autoloss')

    @torch.no_grad()
    def feed_data(self, data):
        # for paired training or validation
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
            self.gt_usm = self.usm_sharpener(self.gt)
        if 'lh' in data:
            self.lh = data['lh'].to(self.device)
            self.lh_usm = self.usm_sharpener(self.lh)
        if 'gray' in data:
            self.atten = data['gray'].to(self.device)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        # do not use the synthetic process during validation
        self.is_train = False
        super(GANOWGNModel, self).nondist_validation(dataloader, current_iter, tb_logger, save_img)
        self.is_train = True

    def backward_G(self, current_iter, l1_gt, percep_gt, loss=False):
        self.l_g_total = 0
        if self.use_aw:
            # pixel loss
            if self.cri_pix:
                self.l_g_pix = self.cri_pix(self.output, l1_gt)
                self.loss_dict['l_g_pix'] = self.l_g_pix
            # perceptual loss
            if self.cri_perceptual:
                self.l_g_percep, self.l_g_style = self.cri_perceptual(self.output, percep_gt)
                if self.l_g_percep is not None:
                    self.loss_dict['l_g_percep'] = self.l_g_percep
                if self.l_g_style is not None:
                    self.loss_dict['l_g_style'] = self.l_g_style
            # low light loss
            # color loss
            if self.cri_color:
                self.l_g_color = self.cri_color(self.output)
                self.loss_dict['l_g_color'] = self.l_g_color
            # exposure control loss
            if self.cri_exp:
                self.l_g_exp = self.cri_exp(self.output)
                self.loss_dict['l_g_exp'] = self.l_g_exp
            # TV loss
            if self.cri_tv:
                self.l_g_tv = self.cri_tv(self.ue)
                self.loss_dict['l_g_tv'] = self.l_g_tv
            # ue attention loss
            if self.cri_ue:
                self.l_g_ue = self.cri_ue(self.ue, torch.abs(torch.max(self.lh, 1)[0].unsqueeze(1)-torch.max(self.lq, 1)[0].unsqueeze(1))/(torch.max(self.lh, 1)[0].unsqueeze(1)+0.0001))
                self.loss_dict['l_g_ue'] = self.l_g_ue
            # gan loss
            fake_g_pred = self.net_d(self.output)
            self.l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
            self.loss_dict['l_g_gan'] = self.l_g_gan
            # AutomaticWeightedLoss
            self.l_g_total = self.cri_aw(self.l_g_pix, self.l_g_percep, self.l_g_ue, self.l_g_gan, self.l_g_tv)

        else:
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
            # TV loss
            if self.cri_tv:
                self.l_g_tv = self.cri_tv(self.ue)
                self.l_g_total += self.l_g_tv
                self.loss_dict['l_g_tv'] = self.l_g_tv
            # ue attention loss
            if self.cri_ue:
                self.l_g_ue = self.cri_ue(self.ue, torch.abs(torch.max(self.lh, 1)[0].unsqueeze(1)-torch.max(self.lq, 1)[0].unsqueeze(1))/(torch.max(self.lh, 1)[0].unsqueeze(1)+0.0001))
                self.l_g_total += self.l_g_ue
                self.loss_dict['l_g_ue'] = self.l_g_ue
            # gan loss
            fake_g_pred = self.net_d(self.output)
            self.l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
            self.l_g_total += self.l_g_gan
            self.loss_dict['l_g_gan'] = self.l_g_gan

        self.loss_dict['l_g_total'] = self.l_g_total
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

        self.loss_dict = OrderedDict()
        if self.use_grayatten:
            self.output, self.ue = self.net_g(self.lq, self.atten)
        else:
            self.output, self.ue = self.net_g(self.lq)

        self.optimizer_g.zero_grad()

        if self.net_d_iters >= 1:
            if self.net_d_iters == 1:
                self.backward_G(current_iter, l1_gt, percep_gt)

                if self.GN:
                    gFyfy = 0
                    gfyfy = 0
                    F = self.backward_G(current_iter, l1_gt, percep_gt, True)
                    f = self.backward_D_real(gan_gt, True)
                    f = f + self.backward_D_fake(True)
                    dfy = torch.autograd.grad(f, self.net_d.parameters(), retain_graph=True)
                    dFy = torch.autograd.grad(F, self.net_d.parameters(), retain_graph=True)
                    for Fy, fy in zip(dFy, dfy):
                        gFyfy = gFyfy + torch.sum(Fy * fy)
                        gfyfy = gfyfy + torch.sum(fy * fy) + 1e-10
                    self.GN_loss = -gFyfy.detach() / gfyfy.detach() * f
                    self.GN_loss.backward()

                self.optimizer_g.step()

            else:
                if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
                    self.backward_G(current_iter, l1_gt, percep_gt)
                    self.optimizer_g.step()

            self.optimizer_d.zero_grad()
            self.backward_D_real(gan_gt)
            self.backward_D_fake()
            self.optimizer_d.step()

        else:
            self.backward_G(current_iter, l1_gt, percep_gt)
            self.optimizer_g.step()
            self.optimizer_d.zero_grad()

            if (current_iter % self.net_g_iters == 0 and current_iter > self.net_d_init_iters):
                self.backward_D_real(gan_gt)
                self.backward_D_fake()
                self.optimizer_d.step()



        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

        self.log_dict = self.reduce_loss_dict(self.loss_dict)

