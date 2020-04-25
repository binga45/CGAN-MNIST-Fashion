import os
import sys

from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
#import utils
from utils.toolbox import *
from utils.cgan import *

from scipy.interpolate import interp1d

VIZ_MODE=0
CUDA = True
data_dir = '../dataset/fashionmnist'
out_dir = '../output'
epochs=3
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
cudnn.benchmark = True      # May train faster but cost more memory

device = torch.device("cuda:0" if CUDA else "cpu")
netG = Generator(classes, channels, img_size, latent_dim).to(device)
#netG.load_state_dict(torch.load(os.path.join(out_dir, 'CGAN_29_G.pt')))
netG.load_state_dict( torch.load('CGAN_29_G.pt',map_location='cpu'))
netG.to(device)
netD = Discriminator(classes, channels, img_size, latent_dim).to(device)

netG.eval()
netD.eval()
if batch_size is None:
    batch_size = data_loader.batch_size
nrows = batch_size // 8
viz_labels = np.array([num for _ in range(nrows) for num in range(2,10)])
viz_labels = torch.LongTensor(viz_labels).to(device)
# viz_label = torch.LongTensor(np.array([num for _ in range(nrows) for num in range(8)])).to(device)
# viz_onehot = _to_onehot(viz_label, dim=classes)

with torch.no_grad():
    # z_noise = torch.randn(batch_size, latent_dim, device=device)
    # x_fake_labels = torch.randint(0, classes, (batch_size,), device=device)
    # x_fake = netG(z_noise, x_fake_labels)
    viz_tensor = torch.randn(batch_size, latent_dim, device=device)
    viz_sample = netG(viz_tensor, viz_labels)
    viz_vector = to_np(viz_tensor).reshape(batch_size, latent_dim)
    cur_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    np.savetxt('vec_{}.txt'.format(cur_time), viz_vector)
    vutils.save_image(viz_sample, os.path.join(out_dir,'img_{}.png'.format(cur_time)), nrow=8, normalize=True)

