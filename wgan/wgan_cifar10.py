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
import os
Epoch=500
BATCH_SIZE=32
LR=0.00005
nd=100 #noise dimension
image_size=64 #image size
channel=3 #image channel of MNIST
gc=64 #generate channel
dc=64 #discriminator channel

#print("is it run?")

gnet=nn.Sequential(
    nn.ConvTranspose2d(nd,gc*8,4,1,0,bias=False),
    nn.BatchNorm2d(gc*8),
    nn.ReLU(True),

    nn.ConvTranspose2d(gc*8,gc*4,4,2,1,bias=False),
    nn.BatchNorm2d(gc*4),
    nn.ReLU(True),

    nn.ConvTranspose2d(gc*4,gc*2,4,2,1,bias=False),
    nn.BatchNorm2d(gc*2),
    nn.ReLU(True),

    nn.ConvTranspose2d(gc*2,gc,4,2,1,bias=False),
    nn.BatchNorm2d(gc),
    nn.ReLU(True),

    nn.ConvTranspose2d(gc,channel,4,2,1,bias=False),
    nn.Tanh()
)
dnet=nn.Sequential(
    nn.Conv2d(channel,dc,4,2,1,bias=False),
    nn.LeakyReLU(0.2,inplace=True),

    nn.Conv2d(dc,dc*2,4,2,1,bias=False),
    nn.BatchNorm2d(dc*2),
    nn.LeakyReLU(0.2,inplace=True),

    nn.Conv2d(dc*2,dc*4,4,2,1,bias=False),
    nn.BatchNorm2d(dc*4),
    nn.LeakyReLU(0.2,inplace=True),

    nn.Conv2d(dc*4,dc*8,4,2,1,bias=False),
    nn.BatchNorm2d(dc*8),
    nn.LeakyReLU(0.2,inplace=True),

    nn.Conv2d(dc*8,1,4,1,0,bias=False),
    #nn.Sigmoid(),
)

transform=transforms.Compose([transforms.Resize(image_size),transforms.ToTensor(),transforms.Normalize([0.5]*3,[0.5]*3)])

DOMNLOAD=False

if not(os.path.exists('./cifar10/')) or not os.listdir('./cifar10/'):
    DOWNLOAD=True
dataset=CIFAR10(root='cifar10/',transform=transform,download=DOWNLOAD)

train_loader=Data.DataLoader(dataset=dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=1)

cudnn.benchmark=True

def weights_init(m):
  classname=m.__class__.__name__
  if classname.find('Conv')!=-1:
    m.weight.data.normal_(0.0,0.02)
  elif classname.find('BatchNorm')!=-1:
    m.weight.data.normal_(1.0,0.02)
    m.bias.data.fill_(0)
    
gnet.apply(weights_init)
dnet.apply(weights_init)

'''DOWNLOAD_MNIST=False
if not(os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
    DOWNLOAD_MNIST=True
train_data=torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST,
)
train_loader=Data.DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True)'''

opt_G=t.optim.RMSprop(gnet.parameters(),lr=LR)
opt_D=t.optim.RMSprop(dnet.parameters(),lr=LR)

fix_noise=Variable(t.FloatTensor(BATCH_SIZE,nd,1,1).normal_(0,1))
fix_noise=fix_noise.cuda()
gnet.cuda()
dnet.cuda()
'''for epoch in range(Epoch):
    for step,b_x in enumerate(train_loader,0):
        real,_=b_x
       # print(b_x.shape)
        real=Variable(real)
        real=real.cuda()
       # print(real.size,real.size(0))
        noise=t.randn(real.size(0),nd,1,1)
        noise=Variable(noise)
        noise=noise.cuda()
        
        for parm in dnet.parameters():
                parm.data.clamp_(-0.01,0.01)

        G_paintings=gnet(noise)

       # print(G_paintings.shape)
        prob_Gpaintings=dnet(G_paintings)
        prob_Real = dnet(real)
        D_loss = t.mean(prob_Real) - t.mean( prob_Gpaintings)
        opt_D.zero_grad()
        D_loss.backward(retain_graph=True)
        opt_D.step()

        if step%5==0:
            G_loss=-t.mean(prob_Gpaintings)
            opt_G.zero_grad()
            G_loss.backward()
            opt_G.step()
    print(epoch)'''
input=t.FloatTensor(BATCH_SIZE,3,image_size,image_size)
gen_iterations=0
one=t.FloatTensor([1])
mone=one*-1
one=one.cuda()
mone=mone.cuda()
input=t.FloatTensor(BATCH_SIZE,3,image_size,image_size)
noise=t.FloatTensor(BATCH_SIZE,nd,1,1)
input=input.cuda()
noise=noise.cuda()
#dnet.load_state_dict(t.load('epoch_wnetd.pth'))
#gnet.load_state_dict(t.load('epoch_wnetg.pth'))
for epoch in range(Epoch):
  data_iter=iter(train_loader)
  i=0
  while i<len(train_loader):
    #update D network
    for p in dnet.parameters():
      p.requires_grad=True
    if gen_iterations<25 or gen_iterations%500==0:
      Diters=100
    else:
      Diters=5
    j=0
    while j<Diters and i<len(train_loader):
      j+=1
      for p in dnet.parameters():
        p.data.clamp_(-0.01,0.01)
      data=data_iter.next()
      i+=1
      #train with real
      real,_=data
      dnet.zero_grad()
      real=real.cuda()
      input.resize_as_(real).copy_(real)
      inputv=Variable(input)
      errD_real=dnet(inputv).mean(0).view(1)
      #print(errD_real.shape)
      #print(errD_real.mean(0))
      #print(errD_real.shape)
      errD_real.backward(one)
      #train with fake
      noise.resize_(BATCH_SIZE,nd,1,1).normal_(0,1)
      noisev=Variable(noise)
      fake=gnet(noisev).detach()
      inputv=Variable(fake.data)
      errD_fake=dnet(inputv).mean(0).view(1)
      errD_fake.backward(mone)
      errD=errD_real-errD_fake
      opt_D.step()
      
    #update G network
    for p in dnet.parameters():
      p.requires_grad=False
    gnet.zero_grad()
    noise.resize_(BATCH_SIZE,nd,1,1).normal_(0,1)
    noisev=Variable(noise)
    fake=gnet(noisev)
    errG=dnet(fake)
    errG=errG.mean(0).view(1)
    errG.backward(one)
    opt_G.step()
    gen_iterations+=1
  print(epoch)
    
    #print('[%d/%d][%d/%d][%d] Loss_D: %f Loss_D_real: %f Loss_D_fake %f' % (epoch, Epoch,i,len(train_loader),gen_iterations,errD.data[0],errG.data[0],errD_real.data[0],errD_fake.data[0]))
      
   # fake_p=gnet(fix_noise)
   # imgs=make_grid(fake_p.data*0.5+0.5).cpu()
   # plt.imshow(imgs.permute(1,2,0).numpy())
   # plt.show()
t.save(dnet.state_dict(),'epoch_wnetd.pth')
t.save(gnet.state_dict(),'epoch_wnetg.pth')

dnet.load_state_dict(t.load('epoch_wnetd.pth'))
gnet.load_state_dict(t.load('epoch_wnetg.pth'))

noise=t.randn(64,nd,1,1).cuda()
noise=Variable(noise)
fake_p=gnet(noise)
imgs=make_grid(fake_p*0.5+0.5).cpu()
plt.figure(figsize=(5,5))
plt.imshow(imgs.permute(1,2,0).detach().numpy())
plt.savefig("generated_wgan_cifar10_500.png")
#print("does it run?")
#plt.show()
