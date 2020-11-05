import torch as t
from torch import nn
from torch.autograd import Variable
from torch.optim import RMSprop
from torchvision import transforms
from torchvision.utils import make_grid
import torch.utils.data as Data
import matplotlib.pyplot as plt
import torchvision
import os
Epoch=500
BATCH_SIZE=50
LR=0.001
nd=100 #noise dimension
image_size=28 #mnist image size
channel=1 #image channel of MNIST
gc=28 #generate channel
dc=28 #discriminator channel


gnet=nn.Sequential(
    nn.ConvTranspose2d(nd,gc*8,4,1,0,bias=False),
    nn.BatchNorm2d(gc*8),
    nn.ReLU(True),

    nn.ConvTranspose2d(gc*8,gc*4,4,2,1,bias=False),
    nn.BatchNorm2d(gc*4),
    nn.ReLU(True),

    nn.ConvTranspose2d(gc*4,gc*2,2,2,1,bias=False),
    nn.BatchNorm2d(gc*2),
    nn.ReLU(True),

   # nn.ConvTranspose2d(gc*2,gc,4,1,0,bias=False),
   # nn.BatchNorm2d(gc),
   # nn.ReLU(True),

    nn.ConvTranspose2d(gc*2,channel,4,2,1,bias=False),
    nn.Tanh()
)
dnet=nn.Sequential(
    nn.Conv2d(channel,dc,4,2,1,bias=False),
    nn.LeakyReLU(0.2,inplace=True),

    nn.Conv2d(dc,dc*2,2,2,1,bias=False),
    nn.BatchNorm2d(dc*2),
    nn.LeakyReLU(0.2,inplace=True),

    nn.Conv2d(dc*2,dc*4,2,2,0,bias=False),
    nn.BatchNorm2d(dc*4),
    nn.LeakyReLU(0.2,inplace=True),

    #nn.Conv2d(dc*4,dc*8,2,2,1,bias=False),
   # nn.BatchNorm2d(dc*8),
   # nn.LeakyReLU(0.2,inplace=True),

    nn.Conv2d(dc*4,1,4,1,0,bias=False),
    nn.Sigmoid(),
)
DOWNLOAD_MNIST=False
if not(os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
    DOWNLOAD_MNIST=True
train_data=torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST,
)
train_loader=Data.DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True)
opt_G=t.optim.Adam(gnet.parameters(),lr=LR)
opt_D=t.optim.Adam(dnet.parameters(),lr=LR)

fix_noise=Variable(t.FloatTensor(BATCH_SIZE,nd,1,1).normal_(0,1))
fix_noise=fix_noise.cuda()
gnet.cuda()
dnet.cuda()
for epoch in range(Epoch):
    for step,(b_x,b_y) in enumerate(train_loader,0):
        real=b_x
       # print(b_x.shape)
        real=Variable(real)
        real=real.cuda()
       # print(real.size,real.size(0))
        noise=t.randn(real.size(0),nd,1,1)
        noise=Variable(noise)
        noise=noise.cuda()

        G_paintings=gnet(noise)
       # print(G_paintings.shape)
        prob_Gpaintings=dnet(G_paintings)
        G_loss=t.mean(t.log(-prob_Gpaintings))
        opt_G.zero_grad()
        G_loss.backward()
        opt_G.step()

        prob_Real=dnet(real)
        prob_Gpaintings=dnet(G_paintings.detach())
       # print(prob_Real.shape)
       # print(prob_Gpaintings.shape)
        D_loss=-t.mean(t.log(prob_Real)+t.log(1.-prob_Gpaintings))
        opt_D.zero_grad()
        D_loss.backward(retain_graph=True)
        opt_D.step()
    print(epoch)
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
plt.savefig("generated.png")
#plt.show()
