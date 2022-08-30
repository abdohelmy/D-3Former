"""2 Main function for this project. """
import os
import argparse
import numpy as np
from trainer.trainer import Trainer
from utils.gpu_tools import occupy_memory
import torch
import random
import torch.multiprocessing as mp
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds")%2**32 - 1)
np.random.seed(hash("improves reprod")%2**32 - 1)
torch.manual_seed(hash("remove stochasticity")%2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable")%2**32-1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ### Basic parameters
    parser.add_argument('--gpu', default='0', help='the index of GPU')
    parser.add_argument('--dataset', default='cifar100', type=str, choices=['cifar100', 'imagenet_sub', 'imagenet'])
    parser.add_argument('--data_dir', default='data/seed_1993_subset_100_imagenet/data', type=str)
    parser.add_argument('--baseline', default='lucir', type=str, choices=['lucir'], help='baseline method')
    parser.add_argument('--ckpt_label', type=str, default='exp01', help='the label for the checkpoints')
    parser.add_argument('--ckpt_dir_fg', type=str, default='-', help='the checkpoint file for the 0-th phase')
    parser.add_argument('--resume_fg', action='store_true', help='resume 0-th phase model from the checkpoint')
    parser.add_argument('--resume', action='store_true', help='resume from the checkpoints')
    parser.add_argument('--num_workers', default=16, type=int, help='the number of workers for loading data')
    parser.add_argument('--random_seed', default=1993, type=int, help='random seed for class order')
    parser.add_argument('--train_batch_size', default=128, type=int, help='the batch size for train loader')
    parser.add_argument('--test_batch_size', default=100, type=int, help='the batch size for test loader')
    parser.add_argument('--eval_batch_size', default=100, type=int, help='the batch size for validation loader')
    parser.add_argument('--disable_gpu_occupancy', action='store_false', help='disable GPU occupancy')
    ### Incremental learning parameters
    parser.add_argument('--num_classes', default=100, type=int, help='the total number of classes')
    parser.add_argument('--nb_cl_fg', default=50, type=int, help='the number of classes in the 0-th phase')
    parser.add_argument('--nb_cl', default=10, type=int, help='the number of classes for each phase')
    parser.add_argument('--nb_protos', default=20, type=int, help='the number of exemplars for each class')
    parser.add_argument('--base_epochs', default=250, type=int, help='the number of epochs for zeroth phase')
    parser.add_argument('--epochs', default=250, type=int, help='the number of epochs for incremental phases')
    parser.add_argument('--warmup', default=20, type=int, help='the number of epochs')
    parser.add_argument('--dynamic_budget', action='store_false', help='using dynamic budget setting')
    ### General learning parameters
    parser.add_argument('--lr_factor', default=0.1, type=float, help='learning rate decay factor')
    parser.add_argument('--base_lr1', default=0.00025, type=float, help='learning rate for the 0-th phase')
    parser.add_argument('--base_lr2', default=0.00025, type=float, help='learning rate for the following phases')
    parser.add_argument('--classifier_lr', default=0.0025, type=float, help='learning rate for the following phases')
    ### D^3Former parameters
    parser.add_argument('--the_lambda', default=10, type=float, help='lamda for Knowledge distillation')
    parser.add_argument('--tau', default=1, type=float, help='tau for logits adjustments')
    parser.add_argument('--gamma', default=0.1, type=float, help='gamma for Grad-cam loss')

    the_args = parser.parse_args()

    # Checke the number of classes, ensure they are reasonable
    assert(the_args.nb_cl_fg % the_args.nb_cl == 0)
    assert(the_args.nb_cl_fg >= the_args.nb_cl)

    # Print the parameters
    print(the_args)

    # Set GPU index
    os.environ['CUDA_VISIBLE_DEVICES'] = the_args.gpu
    print('Using gpu:', the_args.gpu)

    trainer = Trainer(the_args)
    # Set the trainer and start training
    trainer.train()



