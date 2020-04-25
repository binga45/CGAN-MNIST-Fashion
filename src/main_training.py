# Pytorch libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
# classic libraries
import numpy as np
#import statsmodels.api as sm    # to estimate an average with 'loess'
import matplotlib.pyplot as plt
#from matplotlib import rc
#rc('font', **font)
#rc('text', usetex=True)
import pandas as pd
import random, string
import os, time, datetime, json
import utils
# perso libraries
font = {'family' : 'DejaVu Sans',
        'weight' : 'bold',
        'size'   : 14}
#rc('font', **font)
#rc('text', usetex=True)
import pandas as pd
import random, string
import os, time, datetime, json
#-------------------------------------------------#
#               A) Hyper-parameters               #
#-------------------------------------------------#
CUDA = True
data_dir = '../dataset/fashionmnist'
out_dir = '../output'
epochs=30
#LOG_FILE = os.path.join(OUT_PATH, 'log.txt')
batch_size = 128
lr = 0.0002
latent_dim=100
classes = 10
img_size = 64
log_interval=100
channels=1
train=True
seed = 1
#-------------------------------------------------#
#               B) Data/Model/Loss                #
#-------------------------------------------------#
# B.1) dataset/corpus
#--------------------
# Corpus
# print the number of parameters


        
clear_folder(out_dir)
#print("Logging to {}\n".format(LOG_FILE))
#sys.stdout = utils.StdOut(LOG_FILE)
CUDA = CUDA and torch.cuda.is_available()
print("PyTorch version: {}".format(torch.__version__))
if CUDA:
    print("CUDA version: {}\n".format(torch.version.cuda))
if seed is None:
    seed = np.random.randint(1, 10000)
print("Random Seed: ", seed)
np.random.seed(seed)
torch.manual_seed(seed)
if CUDA:
    torch.cuda.manual_seed(seed)
cudnn.benchmark = True
device = torch.device("cuda:0" if CUDA else "cpu")

def _to_onehot(var, dim):
        res = torch.zeros((var.shape[0], dim), device=device)
        res[range(var.shape[0]), var] = 1.
        return res

def save_to(name=None,
                verbose=True):
        if verbose:
            print('\nSaving models to {}_G.pt and {}_D.pt ...'.format(name, name))
        torch.save(netG.state_dict(), os.path.join(out_dir, '{}_G.pt'.format(name)))
        torch.save(netD.state_dict(), os.path.join(out_dir, '{}_D.pt'.format(name)))


dataset = dset.FashionMNIST(root=data_dir, download=True,
                     transform=transforms.Compose([
                     transforms.Resize(img_size),
                     transforms.ToTensor(),
                     transforms.Normalize((0.5,), (0.5,))
                     ]))

assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=16, pin_memory=True)

netG = Generator( classes, channels, img_size, latent_dim).to(device)
#netG.apply(weights_init)
print(netG)
nbr_param = sum(p.numel() for p in netG.parameters() if p.requires_grad)
print("generator params", nbr_param)

netD = Discriminator(classes, channels, img_size, latent_dim).to(device)
#netD.apply(weights_init)
print(netD)
nbr_param = sum(p.numel() for p in netD.parameters() if p.requires_grad)
print("discrimator params", nbr_param)

optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))




netG.train()
netD.train()
viz_z = torch.zeros((batch_size, latent_dim), device=device)
viz_noise = torch.randn(dataloader.batch_size, latent_dim, device=device)
nrows = dataloader.batch_size // 8
viz_label = torch.LongTensor(np.array([num for _ in range(nrows) for num in range(8)])).to(device)
viz_onehot = _to_onehot(viz_label, dim=classes)
total_time = time.time()
for epoch in range(epochs):
    batch_time = time.time()
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        batch_size = data.size(0)
        real_label = torch.full((batch_size, 1), 1., device=device)
        fake_label = torch.full((batch_size, 1), 0., device=device)
        # Train G
        netG.zero_grad()
        z_noise = torch.randn(batch_size, latent_dim, device=device)
        x_fake_labels = torch.randint(0, classes, (batch_size,), device=device)
        x_fake = netG(z_noise, x_fake_labels)
        y_fake_g = netD(x_fake, x_fake_labels)
        g_loss = netD.loss(y_fake_g, real_label)
        g_loss.backward()
        optimizerG.step()

        # Train D
        netD.zero_grad()
        y_real = netD(data, target)
        d_real_loss = netD.loss(y_real, real_label)

        y_fake_d = netD(x_fake.detach(), x_fake_labels)
        d_fake_loss = netD.loss(y_fake_d, fake_label)
        d_loss = (d_real_loss + d_fake_loss) / 2
        d_loss.backward()
        optimizerD.step()

        if batch_idx % log_interval == 0 and batch_idx > 0:  
            print('Epoch {} [{}/{}] loss_D: {:.4f} loss_G: {:.4f} time: {:.2f}'.format(
                  epoch, batch_idx, len(dataloader),
                  d_loss.mean().item(),
                  g_loss.mean().item(),
                  time.time() - batch_time))
            vutils.save_image(data, os.path.join(out_dir, 'real_samples.png'), normalize=True)
            with torch.no_grad():
                viz_sample = netG(viz_noise, viz_label)
                vutils.save_image(viz_sample, os.path.join(out_dir, 'fake_samples_{}.png'.format(epoch)), nrow=8, normalize=True)
            batch_time = time.time()

    save_to( name='CGAN_{}'.format(epoch))
print('Total train time: {:.2f}'.format(time.time() - total_time))
    
