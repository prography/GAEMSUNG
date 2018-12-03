import torch
import torch.nn as nn
from torch.optim import Adam
import os
from model import Classifier


class Trainer(object):
    # pass config directly to Trainer
    def __init__(self, train_loder, test_loader, config):
        self.train_loader = train_loder
        self.test_loader = test_loader
        self.config = config
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.num_channels = config.num_channels
        self.ndf = config.ndf
        self.num_classes = config.num_classes

        self.num_epochs = config.num_epochs
        self.lr = config.lr
        self.training_path = config.training_path
        self.ckpt_folder = config.ckpt_folder

        self.log_interval = config.log_interval
        self.val_interval = config.val_interval
        self.ckpt_interval = config.ckpt_interval

        self.build_net()

    def build_net(self):
        # define network
        self.net = Classifier(self.num_channels, self.ndf, self.num_classes)

        if self.config.mode == 'test' and self.training_path == '':
            print("[*] Enter model path!")
            exit()

        # if training model exists
        if self.training_path != '':
            self.net.load_state_dict(torch.load(self.training_path, map_location=lambda storage, loc: storage))
            print("[*] Load weight from {}!".format(self.training_path))

        self.net.to(self.device)

    def train(self):
        # define loss function
        criterion = nn.CrossEntropyLoss().to(self.device)

        # define optimizer
        optimizer = Adam(self.net.parameters(), lr=self.lr)

        step = 0

        print("[*] Learning started!")
        for epoch in range(self.num_epochs):
            for i, (imgs, labels) in enumerate(self.train_loader):
                # convert model to training mode
                self.net.train()

                imgs = imgs.to(self.device)
                labels = labels.to(self.device)

                # forward to network
                outputs = self.net(imgs)
                loss = criterion(outputs, labels)

                # backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # print log message
                if (step+1) % self.log_interval == 0:
                    print("[{}/{}] [{}/{}] Loss:{:3f}".format(
                        epoch+1, self.num_epochs, i+1, len(self.train_loader), loss.item()
                    ))

                # do validating
                if (step+1) % self.val_interval == 0:
                    self.eval()

                step += 1

            # do checkpointing
            if (epoch+1) % self.ckpt_interval == 0:
                torch.save(self.net.state_dict(), os.path.join(self.ckpt_folder, 'checkpoint_epoch{}.pth'.format(epoch+1)))
                print("[*] Checkpoint saved!")

        print("[*] Learning finished!")
        torch.save(self.net.state_dict(), os.path.join(self.ckpt_folder, 'final_network.pth'))

    def eval(self):
        # convert model to evaluating mode
        self.net.eval()
        with torch.no_grad():
            correct = 0
            total = 0

            for imgs, labels in self.test_loader:
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.net(imgs)
                preds = torch.argmax(outputs, dim=1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

            print("[*] Test Accuracy: {}%".format(100*correct / float(total)))