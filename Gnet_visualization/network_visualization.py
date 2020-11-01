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
import numpy as np
from PIL import Image
from scipy import misc


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
        #print(output.shape)
        output = output.view(-1, 4*4*4*DIM,1,1)
        #print(output.shape)
        output = self.preprocess2(output)
        #print(output.shape)
        output = output.view(-1, 4 * DIM, 4, 4)
        #print(output.shape)
        output1=output
        output = self.block1(output)
        #print(output.shape)
        output2=output
        output = self.block2(output)
        #print(output.shape)
        output3=output
        output = self.deconv_out(output)
        #print(output.shape)
        output = self.tanh(output)
        #print(output.shape)
        return output1,output2,output3, output.view(-1, 3, 32, 32)

'''activation={}

def get_activation(name):
    def hook(model,input,output):
      activation[name]=output
    return hook'''
def make_dirs(path):
    if os.path.exists(path) is False:
        os.makedirs(path)

Gnet=Generator()

fixed_noise_128 = t.randn(10, 128)
#fixed_noise_128 = fixed_noise_128.cuda(0)
noisev = Variable(fixed_noise_128, volatile=True)    
#print(Gnet)

Gnet.load_state_dict(t.load('Gnet_it20.pth'))

'''Gnet.preprocess.register_forward_hook(get_activation('block1'))
Gnet.preprocess.register_forward_hook(get_activation('block2'))
Gnet.preprocess.register_forward_hook(get_activation('deconv_out'))'''
output1,output2,output3,samples=Gnet(noisev)
#print(output1[0].shape,output2[0].shape,output3[0].shape)

items={}
items['layer-1']=output1
items['layer-2']=output2
items['layer-3']=output3

dst='./g_features'

for i in items:
  features=items[i][0]
  #print(features.shape)
  ans=np.zeros(shape=(features.shape[0],features.shape[1],features.shape[2]))
  for j in range(features.shape[0]):
    #features = features.mul(0.5).add(0.5)
    #samples = samples.cpu().data.numpy()
    feature=features.cpu().data.numpy()
    ans[j]=feature[j,:,:]
    feature_img=feature[j,:,:]
    feature_img=feature_img.reshape(1,feature_img.shape[0],feature_img.shape[1])
    #new_im=Image.fromarray(feature)
    
    #print(feature_img.shape)
    feature_img=np.asarray(feature_img*255,dtype=np.uint8)
    
    dst_path=os.path.join(dst,i)
    make_dirs(dst_path)
    print(dst_path)
    #misc.imsave(dst_path+'/'+'layer1{}_{}.jpg'.format(i,j),feature_img)
    save_images.save_images(feature_img, dst_path+'/'+'{}_{}.jpg'.format(i,j))
  save_images.save_images(ans,'{}.jpg'.format(i))


samples = samples.view(-1, 3, 32, 32)
samples = samples.mul(0.5).add(0.5)
samples = samples.cpu().data.numpy()

save_images.save_images(samples, 'samples_{}.jpg'.format(10))
#print(samples.shape)
'''print(activation['block1'].shape)
print(activation['block2'].shape)
print(activation['deconv_out'].view(-1,3,32,32))'''
