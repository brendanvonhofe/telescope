from pathlib import Path
import json
import time
import copy
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter

from dataset import getTrainValSplit, getTransforms, MatteDataset
from linknet import LinkNet34

PATH = Path('/wdblue/deep_image_matting/Combined_Dataset/Training_set')
BG = PATH/'bg'
FG = PATH/'fg'
MASKS = PATH/'mask'
MODELS = Path('/home/bread/telescope/deep_image_matting/models')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    with open("config.json", "r") as read_file:
        config = json.load(read_file)

    batch_size = config['batch_size']
    pretrained = config['pretrained_model']
    savename = config['savename']
    iterations = config['iterations']

    train_fns, val_fns = getTrainValSplit(BG)
    data_transform = getTransforms()

    image_datasets = {'train': MatteDataset(train_fns, root_dir=PATH, fg_path=FG, transform=data_transform),
                  'val': MatteDataset(val_fns, root_dir=PATH, fg_path=FG, transform=data_transform)}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                shuffle=True, num_workers=8)
                for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    telescope = LinkNet34(1)
    if(len(pretrained)):
        print("Loading weights from", pretrained)
        telescope.load_state_dict(torch.load(MODELS/pretrained))
    telescope = telescope.to(device)

    criterion = dim_loss_weighted()
    optimizer = optim.Adam(telescope.parameters(), lr=1e-5)
    model = telescope

    print(datetime.now())
    since = time.time()

    it = 0

    writer = SummaryWriter('runs/' + savename)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 50000
    
    num_epochs = int(iterations / 250)
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = []

            # Iterate over data.
            for i, sample in enumerate(dataloaders[phase]):
                if(phase=='train'):
                    it += 1
                if(i != 0 and i % 250 == 0 and phase == 'train'):
                    break
                inputs, labels, fg, bg = sample['im_map'], sample['mask'], sample['fg'], sample['bg']
                inputs = inputs.to(device)
                labels = labels.to(device)
                fg = fg.to(device)
                bg = bg.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels, fg, bg, inputs[:,3,:,:])
                    # loss = criterion(outputs, labels, fg, bg)
                    running_loss.append(loss.item()/batch_size)
                    print("Loss at step {}: {}".format(i, loss/batch_size))

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        writer.add_scalar('train_loss', loss/batch_size, it)
                        loss.backward()
                        optimizer.step()

            epoch_loss = np.array(running_loss).mean()
            writer.add_scalar(phase + "epoch_loss", epoch_loss, it)
            
            print('{} Loss: {:.4f}'.format(
                phase, epoch_loss))

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                
            print("Saving model at", savename)
            torch.save(model.state_dict(), MODELS/savename)

    print(datetime.now())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:4f}'.format(best_loss))

    writer.close()
    # load best model weights
    model.load_state_dict(best_model_wts)

    torch.save(telescope.state_dict(), MODELS/savename)

def composite(fg, bg, alpha):
    foreground = torch.mul(alpha, fg)
    background = torch.mul(1.0 - alpha, bg)
    return torch.add(foreground, background)

class _Loss(nn.Module):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(_Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction

def alpha_pred_loss(p_mask, gt_mask, eps=1e-6):
    return torch.sqrt(gt_mask.sub(p_mask).pow(2).sum() + eps)

def alpha_pred_loss_weighted(p_mask, gt_mask, trimap, eps=1e-6):
    sqr_diff = gt_mask.sub(p_mask).pow(2)
    unknown = torch.eq(trimap, torch.FloatTensor(np.ones(gt_mask.shape)*(128./255)).to(device)).float()
    return torch.sqrt(torch.mul(sqr_diff, unknown).sum() + eps)

class temp_loss(_Loss):
    def __init__(self, eps=1e-6):
        super(temp_loss, self).__init__()
        self.eps = eps

    def forward(self, p_mask, gt_mask):
        return alpha_pred_loss(p_mask, gt_mask, self.eps)

def compositional_loss(p_mask, gt_mask, fg, bg, eps=1e-6):
    gt_comp = composite(fg, bg, gt_mask)
    p_comp = composite(fg, bg, p_mask)
    return torch.sqrt(gt_comp.sub(p_comp).pow(2).sum() + eps)

def compositional_loss_weighted(p_mask, gt_mask, fg, bg, trimap, eps=1e-6):
    gt_comp = composite(fg, bg, gt_mask)
    p_comp = composite(fg, bg, p_mask)
    bs, h, w = trimap.shape
    ones = torch.FloatTensor(np.ones(trimap.shape)*(128./255)).to(device)
    unknown = torch.eq(trimap, ones).float().expand(3, bs, h, w).contiguous().view(bs,3,h,w)
    s_diff = gt_comp.sub(p_comp).pow(2)
    return torch.sqrt(torch.mul(s_diff, unknown).sum() + eps)
    
class dim_loss(_Loss):
    def __init__(self, eps=1e-6, w=0.5):
        super(dim_loss, self).__init__()
        self.eps = eps
        self.w = w
        
    def forward(self, p_mask, gt_mask, fg, bg):
        return self.w * alpha_pred_loss(p_mask, gt_mask, self.eps) + \
               (1-self.w) * compositional_loss(p_mask, gt_mask, fg, bg, self.eps)
    
class dim_loss_weighted(_Loss):
    def __init__(self, eps=1e-6, w=0.5):
        super(dim_loss_weighted, self).__init__()
        self.eps = eps
        self.w = w
        
    def forward(self, p_mask, gt_mask, fg, bg, trimap):
        return self.w * alpha_pred_loss_weighted(p_mask, gt_mask, trimap, self.eps) + \
               (1-self.w) * compositional_loss_weighted(p_mask, gt_mask, fg, bg, trimap, self.eps)

if __name__ == "__main__":
    main()