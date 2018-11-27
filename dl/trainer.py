import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision.utils import save_image

from datetime import datetime

import os

from encoder_models import *

class Trainer(object):
    def __init__(self, train_loader, test_loader, config):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.num_epochs = config.num_epochs
        self.lr = config.lr

        self.image_size = config.image_size
        self.hidden_dim = config.hidden_dim
        self.output_dim = config.output_dim

        self.log_interval = config.log_interval
        self.sample_interval = config.sample_interval
        self.ckpt_interval = config.ckpt_interval
 
        self.sample_folder = config.sample_folder  
        self.ckpt_folder =  config.ckpt_folder

        self.build_net()

    def build_net(self):
        # define network
        if self.config.encoder == 'AE':
            self.net = AutoEncoder(self.image_size, self.hidden_dim, self.output_dim)
        elif self.config.encoder == 'VAE':
            self.net = VAE(self.image_size, self.hidden_dim, self.output_dim)
        else:
            print("Please Select Auto Encoder Mode")
            exit()

        if self.config.mode == 'test' and self.config.training_path == '':
            print("[*] Enter model path!")
            exit()

        # if training model exists
        if self.config.training_path != '':
            self.net.load_state_dict(torch.load(self.config.training_path, map_location=lambda storage, loc: storage))
            print("[*] Load weight from {}!".format(self.config.training_path))

        self.net.to(self.device)

    def loss_function(self, recon_x, x, mu, logvar):
        criterion = nn.BCELoss(reduction='sum').to(self.device)
        bce = criterion(recon_x, x.view(-1, self.image_size**2))
        kld = -0.5 * torch.sum(1 + logvar-mu**2 - logvar.exp())
        return bce + kld

    def train(self):
        # define loss function
        bce_criterion = nn.BCELoss().to(self.device)
        mse_criterion = nn.MSELoss().to(self.device)

        # define optimizer
        optimizer = Adam(self.net.parameters(), self.lr)

        step = 0
        print("[*] Learning started!")

        # get fixed sample
        temp_iter = iter(self.train_loader)
        fixed_imgs, _ = next(temp_iter)
        fixed_imgs = fixed_imgs.to(self.device)

        # save fixed sample image
        x_path = os.path.join(self.sample_folder, 'fixed_input.png')
        save_image(fixed_imgs, x_path)
        print("[*] Save fixed input image!")

        fixed_imgs = fixed_imgs.view(fixed_imgs.size(0), -1)

        for epoch in range(self.num_epochs):
            for i, (imgs, _) in enumerate(self.train_loader):
                self.net.train()

                imgs = imgs.view(imgs.size(0), -1)
                imgs = imgs.to(self.device)

                # forwarding
                if self.config.encoder == 'AE':
                    outputs = self.net(imgs)
                    bce_loss = bce_criterion(outputs, imgs)
                    mse_loss = mse_criterion(outputs, imgs)

                     # backwarding
                    optimizer.zero_grad()
                    bce_loss.backward()
                    optimizer.step()

                    # do logging
                    if (step+1) % self.log_interval == 0:
                        print("[{}/{}] [{}/{}] BCE loss:{:3f} MSE loss:{:3f}".format(
                            epoch+1, self.num_epochs, i+1, len(self.train_loader), bce_loss.item(), mse_loss.item())
                        )

                    # do sampling
                    if (step+1) % self.sample_interval == 0:
                        outputs = self.net(fixed_imgs)
                        x_hat = outputs.cpu().data.view(outputs.size(0), -1, self.image_size, self.image_size)
                        x_hat_path = os.path.join(self.sample_folder, 'output_epoch{}.png'.format(epoch+1))
                        save_image(x_hat, x_hat_path)

                        print("[*] Save sample images!")

                elif self.config.encoder == 'VAE':
                    recon, mu, logvar = self.net(imgs)
                    loss = self.loss_function(recon, imgs, mu, logvar)

                    # backwarding 
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if (step+1) % self.log_interval == 0:
                        print("[{}/{}] [{}/{}] Loss:{:3f}".format(epoch+1, self.num_epochs, i+1, len(self.train_loader), loss.item()/len(imgs)))
                    
                    if (step+1) % self.sample_interval == 0:
                        recon, mu, logvar = self.net(fixed_imgs)
                        recon = recon.view(-1,1, self.image_size, self.image_size)
                        x_hat_path = os.path.join(self.sample_folder, 'output_epoch{}.png'.format(epoch+1))
                        save_image(recon, x_hat_path)

                        print("[*] Save sample images!")

                step += 1

            if (epoch+1) % self.ckpt_interval == 0:
                ckpt_path = os.path.join(self.ckpt_folder, 'ckpt_epoch{}.pth'.format(epoch+1))
                torch.save(self.net.state_dict(), ckpt_path)
                print("[*] Checkpoint saved!")

        print("[*] Learning finished!")
        ckpt_path = os.path.join(self.ckpt_folder, 'final_model.pth')
        torch.save(self.net.state_dict(), ckpt_path)
        print("[*] Final weight saved!")
