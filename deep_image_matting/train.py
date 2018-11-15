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

from dataset import getTrainValSplit, getTransforms, MatteDataset
from linknet import LinkNet34

PATH = Path('/wdblue/deep_image_matting/Combined_Dataset/Training_set')
BG = PATH/'bg'
FG = PATH/'fg'
MASKS = PATH/'mask'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    with open("config.json", "r") as read_file:
        config = json.load(read_file)

    batch_size = config['batch_size']
    pretrained = config['pretrained_model']
    savename = config['savename']

    train_fns, val_fns = getTrainValSplit(BG)
    data_transform = getTransforms()

    image_datasets = {'train': MatteDataset(train_fns, root_dir=PATH, fg_path=FG, transform=data_transform),
                  'val': MatteDataset(val_fns, root_dir=PATH, fg_path=FG, transform=data_transform)}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                shuffle=True, num_workers=8)
                for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    telescope = LinkNet34(1)
    telescope.load_state_dict(torch.load(pretrained))
    telescope = telescope.to(device)

    criterion = dim_loss_weighted()
    optimizer_ft = optim.Adam(telescope.parameters(), lr=1e-5)

    print(datetime.now())
    telescope = train_model(telescope, dataloaders, criterion, optimizer_ft, batch_size, num_epochs=1)
    print(datetime.now())

    torch.save(telescope.state_dict(), savename)

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
    return torch.sqrt(torch.mul(gt_mask.sub(p_mask).pow(2), torch.eq(trimap, torch.FloatTensor(np.ones(gt_mask.shape)*(127./255)).to(device)).float()).sum() + eps)

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
    unknown = torch.eq(trimap, torch.FloatTensor(np.ones(trimap.shape)*(127./255)).to(device)).float().expand(3, bs, h, w).contiguous().view(bs,3,h,w)
    s_diff = gt_comp.sub(p_comp).pow(2)
    return torch.sqrt(torch.mul(s_diff, unknown).sum() + eps)
    
class dim_loss(_Loss):
    def __init__(self, eps=1e-6, w=0.5):
        super(dim_loss, self).__init__()
        self.eps = eps
        self.w = w
        
    def forward(self, p_mask, gt_mask, fg, bg):
        return self.w * alpha_pred_loss(p_mask, gt_mask, self.eps) + (1-self.w) * compositional_loss(p_mask, gt_mask, fg, bg, self.eps)
    
class dim_loss_weighted(_Loss):
    def __init__(self, eps=1e-6, w=0.5):
        super(dim_loss_weighted, self).__init__()
        self.eps = eps
        self.w = w
        
    def forward(self, p_mask, gt_mask, fg, bg, trimap):
        return self.w * alpha_pred_loss_weighted(p_mask, gt_mask, trimap, self.eps) + (1-self.w) * compositional_loss_weighted(p_mask, gt_mask, fg, bg, trimap, self.eps)


def train_model(model, dataloaders, criterion, optimizer, bs, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 50000
    
    loss_records = {'train': [], 'val': [], 'per_batch': []}

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
#             running_corrects = 0

            # Iterate over data.
            for i, sample in enumerate(dataloaders[phase]):
                if(i % 100 == 0):
                    torch.save(model.state_dict(), "modelv0_5x.pt")
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
#                     loss = criterion(outputs, labels)
                    loss_records['per_batch'].append(loss/bs)
                    print("Loss at step {}: {}".format(i, loss/bs))

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
#                 running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
#             epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            loss_records[phase].append(epoch_loss)
            print('{} Loss: {:.4f}'.format(
                phase, epoch_loss))

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                
            torch.save(model.state_dict(), "modelv0_5x.pt")

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, loss_records

if __name__ == "__main__":
    main()