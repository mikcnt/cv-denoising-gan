from discriminator import Discriminator
from generator import Generator
import dataset
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm

import torchvision.transforms as tf
import torch.nn as nn
import torch.nn.functional as F

discriminator = Discriminator()
disc_opt = optim.Adam(discriminator.parameters(), lr=0.01, momentum=0.9)
generator = Generator(0,0,0,0)
disc_opt = optim.Adam(generator.parameters(), lr=0.01, momentum=0.9)

train_dataset = dataset.ImageDataset("data/train")
test_dataset = dataset.ImageDataset("data/test")
train_loader = DataLoader(train_dataset, batch_size = 5, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = 5, shuffle = True)

disc_criterion = nn.BCELoss()
gen_criterion = nn.BCELoss()

for data, target in tqdm(train_loader):
    fake = generator.forward(data)
    p = discriminator.forward(data) # probability of classifying real data
    q = discriminator.forward(fake) # probability of misclassifying fake data
    
    disc_loss = disc_criterion(p, 1-q)
    gen_loss = gen_criterion(...)
    
    disc_loss.backward()
    gen_loss.backward()
    
    disc_opt.step()
    disc_opt.zero_grad()
    
    gen_opt.step()
    gen_opt.zero_grad()