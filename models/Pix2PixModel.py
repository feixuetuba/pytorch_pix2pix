import logging
import os

import numpy as np
import torch

from models.NN import networks
from utils import tensor2img


class Pix2PixModel():
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.
    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).
    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """

    def __init__(self, cfg):
        """Initialize the pix2pix class.
        """
        m_cfg = cfg.copy()
        m_cfg.update(cfg['model'])
        self.cfg = m_cfg
        gpu_ids = [int(_) for _ in m_cfg.get('gpu_ids', [])]
        if len(gpu_ids) == 0:
            self.gpu_ids = None
        else:
            self.gpu_ids = [int(_) for _ in gpu_ids]
        self.isTrain = (cfg['stage'] == "train")
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids is not None else torch.device(
            'cpu')  # get device name: CPU or GPU

        G_cfg = cfg.copy()
        G_cfg.update(cfg['nn'])
        G_cfg.update(cfg['nn']['G'])
        self.netG = networks.define_G(
            G_cfg['input_nc'],
            G_cfg['output_nc'],
            G_cfg['nf'],
            G_cfg['net'],
            G_cfg['norm'],
            G_cfg['use_dropout'],
            G_cfg['init_type'],
            G_cfg['init_gain'],
            gpu_ids)

        if self.isTrain:
            D_cfg = cfg.copy()
            D_cfg.update(cfg['nn'])
            D_cfg.update(cfg['nn']['D'])
            self.netD = networks.define_D(
                D_cfg['input_nc'] + D_cfg['output_nc'],
                D_cfg['nf'],
                D_cfg['net'],
                D_cfg['n_layers'],
                D_cfg['norm'],
                D_cfg['init_type'],
                D_cfg['init_gain'],
                gpu_ids)

            self.criterionGAN = networks.GANLoss(m_cfg['gan_mode']).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            lr = m_cfg['learning_rate']
            beta1 = m_cfg['beta1']
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=lr, betas=(beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=lr, betas=(beta1, 0.999))
            self.optimizers = [self.optimizer_G, self.optimizer_D]
            self.schedulers = [networks.get_scheduler(optimizer, m_cfg) for optimizer in self.optimizers]
        else:
            self.netG.eval()
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.cfg['direction'] == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        if 'B' in input:
            self.real_B = input['B' if AtoB else 'A'].to(self.device)
            self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()
        self.current_loss.update({
            'D-loss': self.loss_D.item()
        })

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.cfg['lambda_L1']
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()
        self.current_loss.update({
            'G_L1': self.loss_G_L1.item(),
            'GAN-loss': self.loss_G_GAN.item()
        })

    def optimize_parameters(self):
        self.current_loss = {}
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights

    def save(self, save_dir, name, iter):
        for model, net_name in zip([self.netG, self.netD], ['G', 'D']):
            if hasattr(model, 'module'):
                model = model.module
            ckpt_path = os.path.join(save_dir, f"{iter}_{name}_{net_name}.pth")
            logging.info(f"Save ckpt:{ckpt_path}")
            torch.save(model.state_dict(), ckpt_path)

    def load(self, save_dir, name, iter):
        models = [self.netG]
        names = ['G']
        if self.isTrain:
            models.append(self.netD)
            names.append("D")
        for model, net_name in zip(models, names):
            logging.info(f"Load Net:{net_name}")
            ckpt_path = os.path.join(save_dir ,f"{iter}_{name}_{net_name}.pth")
            x = torch.load(ckpt_path)
            if hasattr(x, 'module'):
                x = x.module
                x = x.state_dict()
            model.load_state_dict(x)

    def get_current_visual(self, n):
        real_A = tensor2img(self.real_A, n)
        real_B = tensor2img(self.real_B, n)
        fake_B = tensor2img(self.fake_B, n)
        return np.hstack([real_A, real_B, fake_B])

    def get_current_error(self):
        return self.current_loss

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        old_lr = self.optimizers[0].param_groups[0]['lr']
        for scheduler in self.schedulers:
            if self.cfg['model']['lr_policy'] == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        return old_lr, lr

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad