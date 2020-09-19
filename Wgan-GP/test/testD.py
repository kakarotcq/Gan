import torch as t
from torch import nn
from torch.autograd import Variable
from torch.optim import RMSprop
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.datasets import CIFAR10
import torch.backends.cudnn as cudnn
import torch.utils.data as Data
import matplotlib.pyplot as plt
import torchvision
import save_images
import os
import pickle

FILE_PATH='cifar10/cifar-10-batches-py'
'''transform=transforms.Compose([transforms.Resize(image_size),transforms.ToTensor(),transforms.Normalize([0.5]*3,[0.5]*3)])
dataset=CIFAR10(root='cifar10/',transform=transform,download=False)

train_loader=Data.DataLoader(dataset=dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=1)
'''
DIM=128

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        preprocess = nn.Sequential(
            nn.Linear(128, 4*4*4 * DIM),
        )
        preprocess2=nn.Sequential(
            nn.BatchNorm2d( 4*4*4 * DIM),
            nn.ReLU(True),   
        )
        block1 = nn.Sequential(
            nn.ConvTranspose2d(4 * DIM, 2 * DIM, 2, stride=2),
            nn.BatchNorm2d(2 * DIM),
            nn.ReLU(True),
        )
        block2 = nn.Sequential(
            nn.ConvTranspose2d(2 * DIM, DIM, 2, stride=2),
            nn.BatchNorm2d(DIM),
            nn.ReLU(True),
        )
        deconv_out = nn.ConvTranspose2d(DIM, 3, 2, stride=2)

        self.preprocess = preprocess
        self.preprocess2 = preprocess2
        self.block1 = block1
        self.block2 = block2
        self.deconv_out = deconv_out
        self.tanh = nn.Tanh()

    def forward(self, input):
        output = self.preprocess(input)
        output = output.view(-1, 4*4*4*DIM,1,1)
        output = self.preprocess2(output)
        output = output.view(-1, 4 * DIM, 4, 4)
        output = self.block1(output)
        output = self.block2(output)
        output = self.deconv_out(output)
        output = self.tanh(output)
        return output.view(-1, 3, 32, 32)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        main = nn.Sequential(
            nn.Conv2d(3, DIM, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(DIM, 2 * DIM, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(2 * DIM, 4 * DIM, 3, 2, padding=1),
            nn.LeakyReLU(),
        )

        self.main = main
        self.linear = nn.Linear(4*4*4*DIM, 1)

    def forward(self, input):
        output = self.main(input)
        output = output.view(-1, 4*4*4*DIM)
        output = self.linear(output)
        return output

Gnet=Generator()
Gnet.load_state_dict(t.load('Gnet_it20.pth'))

Dnet=Discriminator()
Dnet.load_state_dict(t.load('Dnet_it20.pth'))

fixed_noise_128 = t.randn(10000, 128)
#fixed_noise_128 = fixed_noise_128.cuda(0)
noisev = Variable(fixed_noise_128, volatile=True)
samples = Gnet(noisev)
'''samples = samples.view(-1, 3, 32, 32)
samples = samples.mul(0.5).add(0.5)
samples = samples.cpu().data.numpy()

test=0'''
#save_images.save_images(samples, 'samples_{}.jpg'.format(test))

result=Dnet(samples)
result=t.sigmoid(result)
a=0
for i in result:
  if(i>0.5):
    a+=1

print('accuracy of generated samples:{}%'.format((float)(a/10000*100)))
#print(result.__class__())

def load_file(filename):
  with open(filename,'rb') as fo:
    data=pickle.load(fo,encoding='latin1')
  return data
  
testdata=load_file(FILE_PATH+'/'+'test_batch')
#print(testdata.keys())
testdata=testdata['data']
preprocess = torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])
data=testdata.reshape(-1,3,32,32).transpose(0,2,3,1)
real_data = t.stack([preprocess(item) for item in data])
a=0
result=Dnet(real_data)
result=t.sigmoid(result)
for i in result:
  if(i>0.5):
    a+=1
print('accuracy of real images:{}%'.format((float)(a/10000*100)))
#print(testdata.shape)
