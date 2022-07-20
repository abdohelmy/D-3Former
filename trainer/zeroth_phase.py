""" Training code for the 0-th phase """
import torch
import tqdm
import numpy as np
import torch.nn as nn
import torchvision
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from utils.misc import *

import torch.nn.functional as F
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
import wandb
from torch.nn import DataParallel as DDP

def incremental_train_and_eval_zeroth_phase(the_args, epochs, cur_model, ref_model, \
    tg_optimizer, tg_lr_scheduler, trainloader, testloader, iteration, start_iteration, \
    lamda, fix_bn=False, weight_per_class=None, device=None):

    # Setting up the CUDA device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
    mixup_args = dict(mixup_alpha=0.8, cutmix_alpha=0.8,prob=1.0, switch_prob=0.5, mode='batch',label_smoothing=0.1, num_classes=the_args.nb_cl_fg)
    mixup_fn = Mixup(**mixup_args)
    min_lr=0
    lr=the_args.base_lr1
    warmup_epochs=the_args.warmup
    for epoch in range(epochs):
        # Set model to the training mode
        cur_model.train()
        cur_model = DDP(cur_model.module)

        # Set all the losses to zeros
        train_loss = 0
        train_loss1 = 0
        train_loss2 = 0
        # Set the counters to zeros
        correct = 0
        total = 0
        # Learning rate decay
        if epoch<warmup_epochs:
            pass
        else:          
            tg_lr_scheduler.step()

        # Print the information
        print('\nEpoch: %d, learning rate: ' % epoch, end='')
        print(tg_lr_scheduler.get_lr()[0])

        for batch_idx, (inputs, targets) in enumerate(trainloader):

            if epoch<warmup_epochs:
                epoch_step=(batch_idx+1)*(epoch+1)/len(trainloader) #steps per epoch
                warmup=np.minimum(1.,epoch_step/warmup_epochs)
                wlr=np.where(warmup<1,lr*warmup,np.maximum(lr*warmup,min_lr))
                tg_optimizer.param_groups[0]['lr'] = wlr
                tg_optimizer.param_groups[1]['lr'] = wlr
                if epoch==warmup_epochs-1:
                    tg_optimizer.param_groups[0]['lr'] = the_args.base_lr1
                    tg_optimizer.param_groups[1]['lr'] = the_args.classifier_lr
            
            if the_args.dataset == 'imagenet_sub' or the_args.dataset == 'imagenet':
                inputs = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(inputs)
            
            inputs, targets = inputs.to(device), targets.to(device)
            inputs, targets = mixup_fn(inputs, targets) 

            # Get a batch of training samples, transfer them to the device
            inputs, targets = inputs.to(device), targets.to(device)
            # Clear the gradient of the paramaters for the tg_optimizer
            tg_optimizer.zero_grad()
            # Forward the samples in the deep networks
            outputs,_ = cur_model(inputs)
            # Compute classification loss
            loss = SoftTargetCrossEntropy()(outputs, targets)

            # Backward and update the parameters
            loss.backward()
            tg_optimizer.step()
            # Record the losses and the number of samples to compute the accuracy
            train_loss += loss.item()


        # Running the test for this epoch
        cur_model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs,_ = cur_model(inputs)
                loss = nn.CrossEntropyLoss(weight_per_class)(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        print('Test set: {} test loss: {:.4f} accuracy: {:.4f}'.format(len(testloader), test_loss/(batch_idx+1), 100.*correct/total))
        wandb.log({"Test accuracy (FC)  ":  100.*correct/total})

    return cur_model
