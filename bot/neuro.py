import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
import uuid

the_model = 0
DEVICE = 0

# Модель будет восстановлена как экземпляр своего класса
#
# based on:
# https://github.com/podgorskiy/VAE/blob/master/net.py
class VAE(nn.Module):
    def __init__(self, zsize, layer_count=3, channels=3):
        super(VAE, self).__init__()

        d = 128
        self.d = d
        self.zsize = zsize

        self.layer_count = layer_count

        mul = 1
        inputs = channels
        for i in range(self.layer_count):
            setattr(self, "conv%d" % (i + 1), nn.Conv2d(inputs, d * mul, 4, 2, 1))
            setattr(self, "conv%d_bn" % (i + 1), nn.BatchNorm2d(d * mul))
            inputs = d * mul
            mul *= 2

        self.d_max = inputs

        self.fc1 = nn.Linear(2048, zsize)
        self.fc2 = nn.Linear(2048, zsize)

        self.d1 = nn.Linear(zsize, 2048)

        mul = inputs // d // 2

        for i in range(1, self.layer_count):
            setattr(self, "deconv%d" % (i + 1), nn.ConvTranspose2d(inputs, d * mul, 4, 2, 1))
            setattr(self, "deconv%d_bn" % (i + 1), nn.BatchNorm2d(d * mul))
            inputs = d * mul
            mul //= 2

        setattr(self, "deconv%d" % (self.layer_count + 1), nn.ConvTranspose2d(inputs, channels, 4, 2, 1))

        self.dout = nn.Linear(3072, 6075)

    def encode(self, x):

        for i in range(self.layer_count):
            x = F.relu(getattr(self, "conv%d_bn" % (i + 1))(getattr(self, "conv%d" % (i + 1))(x)))
        x = x.view(x.shape[0], 2048)
        h1 = self.fc1(x)
        h2 = self.fc2(x)
        return h1, h2

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, x):
        x = x.view(x.shape[0], self.zsize)
        x = self.d1(x)
        x = x.view(x.shape[0], self.d_max, 1, 1)
        x = F.leaky_relu(x, 0.2)

        for i in range(1, self.layer_count):
            x = F.leaky_relu(getattr(self, "deconv%d_bn" % (i + 1))(getattr(self, "deconv%d" % (i + 1))(x)), 0.2)

        x = F.tanh(getattr(self, "deconv%d" % (self.layer_count + 1))(x))
        x = x.view(x.shape[0], 3072)
        x = F.tanh(self.dout(x))
        return x.view(x.shape[0], 3, 45, 45)

    def forward(self, x):
        mu, logvar = self.encode(x)
        mu = mu.squeeze()
        logvar = logvar.squeeze()
        z = self.reparameterize(mu, logvar)
        return self.decode(z.view(-1, self.zsize, 1, 1)), mu, logvar

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

def init():
    print("init neuro start")

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    train_on_gpu = torch.cuda.is_available()

    if not train_on_gpu:
        print('CUDA is not available.  Training on CPU ...')
    else:
        print('CUDA is available!  Training on GPU ...')

    global DEVICE
    global the_model

    DEVICE = torch.device("cuda")

    the_model = torch.load("model9")
    the_model.to(DEVICE)
    print("init neuro done")

def getRecon(filename):

    img = np.array(Image.open(filename).resize([45, 45]))
    inputs = torch.Tensor([img, img]) # Дублирование изображения т.к. модель требует батча ( > 1 изображение)
    inputs = inputs.view(-1, 45, 45, 3)
    inputs = inputs.permute(0, 3, 1, 2).to(DEVICE)

    reconstruction, mu, logvar = the_model(inputs)
    reconstruction = reconstruction.view(-1, 3, 45, 45)
    x_reconstruction = torch.Tensor.cpu(reconstruction[0]).detach()

    x_reconstruction = x_reconstruction.permute(1, 2, 0)

    answ_filename = 'answ_' + str(uuid.uuid4()) + '.jpg'

    arrtosave = np.array(x_reconstruction.view(45, 45, 3)) * 255
    arrtosave = arrtosave.astype(np.uint8)
    data = Image.fromarray(arrtosave)
    data = data.resize([250, 250])
    data.save(answ_filename)

    return answ_filename



