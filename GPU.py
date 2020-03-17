#pip install advertorch

import matplotlib.pyplot as plt
%matplotlib inline

import os
import argparse
import torch
import torch.nn as nn
from torchvision.utils import save_image
import torchvision
from advertorch.utils import predict_from_logits
from advertorch_examples.utils import get_mnist_test_loader
from advertorch_examples.utils import _imshow

torch.manual_seed(0)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

from advertorch.test_utils import LeNet5
from advertorch_examples.utils import TRAINED_MODEL_PATH

filename = "mnist_lenet5_clntrained.pt"
# filename = "mnist_lenet5_advtrained.pt"
#torch.load with map_location=torch.device('cpu')
model = LeNet5()
model.load_state_dict(
    torch.load(os.path.join( filename),map_location=torch.device('cpu')))
model.to(device)
model.eval()

batch_size = 100
loader = get_mnist_test_loader(batch_size=batch_size)
for cln_data, true_label in loader:
    break
cln_data, true_label = cln_data.to(device), true_label.to(device)

from advertorch.defenses import MedianSmoothing2D
from advertorch.defenses import BitSqueezing
from advertorch.defenses import JPEGFilter

bits_squeezing = BitSqueezing(bit_depth=5)
median_filter = MedianSmoothing2D(kernel_size=3)
jpeg_filter = JPEGFilter(10)

defense = nn.Sequential(
    jpeg_filter,
    bits_squeezing,
    median_filter,
)
from advertorch.attacks import LocalSearchAttack
from advertorch.bpda import BPDAWrapper
#defense_withbpda = BPDAWrapper(defense, forwardsub=lambda x: x)
#defended_model = nn.Sequential(defense_withbpda, model)
from advertorch.attacks import SinglePixelAttack
bpda_adversary = LocalSearchAttack(
    model, loss_fn=nn.CrossEntropyLoss(reduction="sum")#, num_classes=10
    )


bpda_adv = bpda_adversary.perturb(cln_data, true_label)
#torch.save(bp[ii].squeeze())
#orch.save(cp[ii].squeeze())
import matplotlib.pyplot as plt
import numpy as np
plt.figure(figsize=(10, 8))
for ii in range(batch_size):
    save_image(bpda_adv[ii].squeeze(),'./drive/My Drive/ls/5'+str(ii)+'.png')
    save_image(cln_data[ii].squeeze(),'./drive/My Drive/ls/500'+str(ii)+'.png')
    #plt.imsave('./drive/My Drive/ls/5'+str(ii)+'.png',torch.save(bpda_adv[ii].squeeze()))
    #plt.imsave('./drive/My Drive/ls/50'+str(ii)+'.png',torch.save(cln_data[ii].squeeze()))
