""" Training code for D^3Former """
import torch
import tqdm
import numpy as np
import torch.nn as nn
from sklearn.metrics import confusion_matrix
import torchvision
from torch_grad_cam.utils.image import show_cam_on_image
from torch_grad_cam import GradCAM
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from utils.misc import *
import wandb
import copy
from timm.loss import *
from timm.data import Mixup
from torch.nn import DataParallel as DDP


def map_labels(order_list, Y_set):
    map_Y = []
    for idx in Y_set:
        map_Y.append(order_list.index(idx))
    map_Y = np.array(map_Y)
    return map_Y

def compute_adjustment(train_loader, device,tro=1):
    """compute the base probabilities"""

    label_freq = {}
    for i, (inputs, target) in enumerate(train_loader):
        target = target.to(device)
        for j in target:
            key = int(j.item())
            label_freq[key] = label_freq.get(key, 0) + 1
    label_freq = dict(sorted(label_freq.items()))
    label_freq_array = np.array(list(label_freq.values()))
    label_freq_array = label_freq_array / label_freq_array.sum()
    adjustments = np.log(label_freq_array ** tro + 1e-12)
    adjustments = torch.from_numpy(adjustments)
    adjustments = adjustments.to(device)
    return adjustments
def gradcam_loss(old, cur_model, ref_model, target_layers, ref_target_layers, device):
    loss3 = torch.zeros(1).to(device)
    temp = None
    ref_temp = None
    if old.shape[0]>0:

        cam = GradCAM(model=cur_model.module, target_layers=target_layers, use_cuda=True)
        cam.batch_size=old.shape
        target = None
        grayscale_cam = cam(input_tensor=old, targets=target)


        ref_cam = GradCAM(model=ref_model.module, target_layers=ref_target_layers, use_cuda=True)
        ref_cam.batch_size=old.shape
        ref_target = None
        ref_grayscale_cam = ref_cam(input_tensor=old, targets=ref_target)

        for i in range(old.shape[0]):
            # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.

            cam = grayscale_cam[i, :]
            cam_image = show_cam_on_image(old[i,:].permute(1,2,0).numpy(), cam, use_rgb=True)
            out = torch.from_numpy(cam_image).float().to(device)

            if(temp is None):
                temp = out.unsqueeze(0)
            else:
                temp = torch.vstack((temp,out.unsqueeze(0)))

            ref_cam = ref_grayscale_cam[i, :]
            ref_cam_image = show_cam_on_image(old[i,:].permute(1,2,0).numpy(), ref_cam, use_rgb=True)
            ref_out = torch.from_numpy(ref_cam_image).float().to(device)

            if(ref_temp is None):
                ref_temp = ref_out.unsqueeze(0)
            else:
                ref_temp = torch.vstack((ref_temp,ref_out.unsqueeze(0)))

        loss3+=nn.L1Loss()(temp,ref_temp)#L1Loss
    return loss3
def incremental_train_and_eval(the_args, epochs,cur_model, ref_model, \
    tg_optimizer, tg_lr_scheduler, trainloader, testloader, iteration, \
    start_iteration, X_protoset_cumuls, Y_protoset_cumuls, order_list, the_lambda,  \
     balancedloader, num_classes, fix_bn=False, weight_per_class=None, device=None):

    # Setting up the CUDA device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set reference model to the evaluation mode
    ref_model.eval()
    mixup_args = dict(mixup_alpha=0.8, cutmix_alpha=0.8,prob=1.0, switch_prob=0.5, mode='batch',label_smoothing=0.1, num_classes=num_classes)
    mixup_fn = Mixup(**mixup_args)

    # Get the number of old classes
    num_old_classes = ref_model.module.fc.out_features
    min_lr=0
    lr=0.0025
    warmup_epochs=5
    adjustments=compute_adjustment(trainloader,device, tro=the_args.tau)

    for epoch in range(epochs):
        # Start training for the current phase
        cur_model.train()
        cur_model = DDP(cur_model.module)

        # Set all the losses to zeros
        train_loss = 0
        train_loss1 = 0
        train_loss2 = 0
        train_loss3 = 0

        # Set the counters to zeros
        correct = 0
        total = 0
    

        tg_lr_scheduler.step()

        # Print the information
        print('\nEpoch: %d, learning rate: ' % epoch, end='')
        print(tg_lr_scheduler.get_lr()[0])
        
        cur_list = list(cur_model.module.model.levels)
        target_layers = [cur_list[2]]

        ref_cur_list = list(ref_model.module.model.levels)
        ref_target_layers = [ref_cur_list[2]]


        for batch_idx, (inputs, targets) in enumerate(trainloader):
     
            # Get a batch of training samples, transfer them to the device

            old = inputs[targets<num_old_classes].clone()




            
            loss3 = gradcam_loss(old, cur_model, ref_model, target_layers, ref_target_layers, device)*the_args.gamma 
            
            
            
            
            #Normalize for imagenet
            if the_args.dataset == 'imagenet_sub' or the_args.dataset == 'imagenet':
                inputs = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(inputs)

            inputs, targets = inputs.cuda(), targets.cuda()

            exemplers=targets<num_old_classes
            
            if the_args.dataset == 'imagenet_sub' or the_args.dataset == 'imagenet':
                inputs, targets = mixup_fn(inputs, targets)
    
            inputs, targets = inputs.to(device), targets.to(device)

            # Clear the gradient of the paramaters for the tg_optimizer
            tg_optimizer.zero_grad()
            

            # Forward the samples in the deep networks
            outputs, cur_features=cur_model(inputs)
            with torch.no_grad():
                _, ref_features=ref_model(inputs)
                
            # Loss 1: feature-level distillation loss
            if the_args.dataset == 'imagenet_sub' or the_args.dataset == 'imagenet':
                loss1 = nn.CosineEmbeddingLoss()(cur_features, ref_features.detach(), torch.ones(inputs.shape[0]).to(device)) * (the_lambda)
            else:
                loss1 = nn.CosineEmbeddingLoss()(cur_features[exemplers], ref_features[exemplers].detach(), torch.ones(inputs[exemplers].shape[0]).to(device)) * (the_lambda)
            # Loss 2: classification loss
            if the_args.dataset == 'imagenet_sub' or the_args.dataset == 'imagenet':
                loss2 = SoftTargetCrossEntropy()(outputs+adjustments, targets)
            else:
                loss2 = nn.CrossEntropyLoss()(outputs+adjustments, targets)

            # Sum up all looses
            loss = loss1 + loss2 + loss3

            # Backward and update the parameters
            loss.backward()
            tg_optimizer.step()

            # Record the losses and the number of samples to compute the accuracy
            train_loss += loss.item()
            train_loss1 += loss1.item()
            train_loss2 += loss2.item()
            train_loss3 += loss3.item()

        # Running the test for this epoch
        cur_model.eval()
        test_loss = 0
        correct = 0
        total = 0

        #Testing
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs, _ = cur_model(inputs)
                loss = nn.CrossEntropyLoss(weight_per_class)(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        print('Test set: {} test loss: {:.4f} accuracy: {:.4f}'.format(len(testloader), test_loss/(batch_idx+1), 100.*correct/total))
        wandb.log({"Test accuracy (FC)  ":  100.*correct/total})
        wandb.log({"Distil ":  train_loss1/(batch_idx+1)})
        wandb.log({"CE_adj ":  train_loss2/(batch_idx+1)})
        wandb.log({"CAM ":  train_loss3/(batch_idx+1)})

    return cur_model
