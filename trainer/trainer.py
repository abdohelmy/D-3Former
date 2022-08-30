""" Class-incremental learning trainer. """
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import numpy as np
import time
import os
import os.path as osp
import sys
import copy
import argparse
from PIL import Image
try:
    import cPickle as pickle
except:
    import pickle
import math
import utils.misc
from utils.imagenet.utils_dataset import split_images_labels
from utils.imagenet.utils_dataset import merge_images_labels
from utils.incremental.compute_accuracy import compute_accuracy
from trainer.incremental_lucir import incremental_train_and_eval as incremental_train_and_eval_lucir
from trainer.zeroth_phase import incremental_train_and_eval_zeroth_phase as incremental_train_and_eval_zeroth_phase
from utils.misc import process_mnemonics
from trainer.base_trainer import BaseTrainer
import warnings
warnings.filterwarnings('ignore')
import wandb

class Trainer(BaseTrainer):
    def train(self):
        """The class that contains the code for the class-incremental system.
        This trianer is based on the base_trainer.py in the same folder.
        If you hope to find the source code of the functions used in this trainer, you may find them in base_trainer.py.
        """
        
        # # Set tensorboard recorder
        self.train_writer = SummaryWriter(comment=self.save_path)

        # Initial the array to store the accuracies for each phase
        top1_acc_list_cumul = np.zeros((int(self.args.num_classes/self.args.nb_cl), 3, 1))
        top1_acc_list_ori = np.zeros((int(self.args.num_classes/self.args.nb_cl), 3, 1))

        # Load the training and test samples from the dataset
        X_train_total, Y_train_total, X_valid_total, Y_valid_total = self.set_dataset()
    

        # Initialize the class order
        order, order_list = self.init_class_order()
        np.random.seed(None)

        # Set empty lists for the data    
        X_valid_cumuls    = []
        X_protoset_cumuls = []
        X_train_cumuls    = []
        Y_valid_cumuls    = []
        Y_protoset_cumuls = []
        Y_train_cumuls    = []

        # Initialize the prototypes
        alpha_dr_herding, prototypes = self.init_prototypes(self.dictionary_size, order, X_train_total, Y_train_total)

        # Set the starting iteration
        # We start training the class-incremental learning system from e.g., 50 classes to provide a good initial encoder
        start_iter = int(self.args.nb_cl_fg/self.args.nb_cl)-1

        # Set the models and some parameter to None
        # These models and parameters will be assigned in the following phases
        cur_model = None
        ref_model = None
        the_lambda_mult = None
        config = dict(
        epochs=self.args.epochs,
        memory=self.args.nb_protos,
        batch_size=self.args.train_batch_size,
        dataset=self.args.dataset,
        learning_rate = self.args.base_lr1,
        phases=self.args.nb_cl,
        tau=self.args.tau,
        the_lambda=self.args.the_lambda,
        gamma=self.args.gamma
        
    )
        
        flag=0
        with wandb.init(mode="offline",project="nest",config=config):

        

            cur_model = None
            ref_model = None
            the_lambda_mult = None

            self.train_writer = SummaryWriter(comment=self.save_path)

            # Initial the array to store the accuracies for each phase
            top1_acc_list_cumul = np.zeros((int(self.args.num_classes/self.args.nb_cl), 3, 1))
            top1_acc_list_ori = np.zeros((int(self.args.num_classes/self.args.nb_cl), 3, 1))

            # Load the training and test samples from the dataset
            X_train_total, Y_train_total, X_valid_total, Y_valid_total = self.set_dataset()

    
            # Initialize the class order
            order, order_list = self.init_class_order()
            np.random.seed(None)

            # Set empty lists for the data    
            X_valid_cumuls    = []
            X_protoset_cumuls = []
            X_train_cumuls    = []
            Y_valid_cumuls    = []
            Y_protoset_cumuls = []
            Y_train_cumuls    = []

            # Set the starting iteration
            # We start training the class-incremental learning system from e.g., 50 classes to provide a good initial encoder
            start_iter = int(self.args.nb_cl_fg/self.args.nb_cl)-1
            
            for iteration in range(start_iter, int(self.args.num_classes/self.args.nb_cl)):
                ### Initialize models for the current phase
               
                ############################################################
                #self.setup(self.args.world_size-1,self.args.world_size) 
                ###############################################################
                cur_model, ref_model, lambda_mult, cur_lambda, last_iter = self.init_current_phase_model(iteration, start_iter, cur_model)
                num_classes = (iteration+1)*self.args.nb_cl
                if flag==0:
                    wandb.watch(cur_model,log="all", log_freq=1)
                    flag=1
                ### Initialize datasets for the current phase
                if iteration == start_iter:
                    indices_train_10, X_valid_cumul, X_train_cumul, Y_valid_cumul, Y_train_cumul, \
                        X_train_cumuls, Y_valid_cumuls, X_protoset_cumuls, Y_protoset_cumuls, X_valid_cumuls, Y_valid_cumuls, \
                        X_train, map_Y_train, map_Y_valid_cumul, X_valid_ori, Y_valid_ori = \
                        self.init_current_phase_dataset(iteration, \
                        start_iter, last_iter, order, order_list, X_train_total, Y_train_total, X_valid_total, Y_valid_total, \
                        X_train_cumuls, Y_train_cumuls, X_valid_cumuls, Y_valid_cumuls, X_protoset_cumuls, Y_protoset_cumuls)
                else:
                    indices_train_10, X_valid_cumul, X_train_cumul, Y_valid_cumul, Y_train_cumul, \
                        X_train_cumuls, Y_valid_cumuls, X_protoset_cumuls, Y_protoset_cumuls, X_valid_cumuls, Y_valid_cumuls, \
                        X_train, map_Y_train, map_Y_valid_cumul, X_protoset, Y_protoset = \
                        self.init_current_phase_dataset(iteration, \
                        start_iter, last_iter, order, order_list, X_train_total, Y_train_total, X_valid_total, Y_valid_total, \
                        X_train_cumuls, Y_train_cumuls, X_valid_cumuls, Y_valid_cumuls, X_protoset_cumuls, Y_protoset_cumuls)                
    
                is_start_iteration = (iteration == start_iter)
    
                # Imprint weights
                if iteration > start_iter:
                    cur_model = self.imprint_weights(cur_model, iteration, is_start_iteration, X_train, map_Y_train, self.dictionary_size)
    
                # Update training and test dataloader
                trainloader, testloader = self.update_train_and_valid_loader(X_train, map_Y_train, X_valid_cumul, map_Y_valid_cumul, \
                    iteration, start_iter)
    
                # Set the names for the checkpoints
                ckp_name = osp.join(self.save_path, 'iter_{}_b1.pth'.format(iteration))
                ckp_name_b2 = osp.join(self.save_path, 'iter_{}_b2.pth'.format(iteration))            
                print('Check point name: ', ckp_name)
    
                if iteration==start_iter and self.args.resume_fg:
                    # Resume the 0-th phase model according to the config
                    cur_model = torch.load(self.args.ckpt_dir_fg)
                elif self.args.resume and os.path.exists(ckp_name):
                    # Resume other models according to the config
                    cur_model = torch.load(ckp_name)
                else:
                    # Start training (if we don't resume the models from the checkppoints)
        
                    # Set the optimizer
                    tg_optimizer, tg_lr_scheduler = self.set_optimizer(iteration, \
                        start_iter, cur_model, ref_model)     
    
                    if iteration > start_iter:
                        # Training the class-incremental learning system from the 1st phase
    
                        # Set the balanced dataloader
                        balancedloader = self.gen_balanced_loader(X_train_total, Y_train_total, indices_train_10, X_protoset, Y_protoset, order_list)
    
          

                        if self.args.baseline == 'lucir':
                            cur_model = incremental_train_and_eval_lucir(self.args, self.args.epochs, cur_model, ref_model, \
                                tg_optimizer, tg_lr_scheduler, \
                              trainloader, testloader, iteration, start_iter, \
                                X_protoset_cumuls, Y_protoset_cumuls, order_list, cur_lambda , balancedloader,num_classes)
                        else:
                            raise ValueError('Please set the correct baseline.')       
                    else:         
                        # Training the class-incremental learning system from the 0th phase           
                        cur_model = incremental_train_and_eval_zeroth_phase(self.args, self.args.base_epochs, cur_model, \
                            ref_model, tg_optimizer, tg_lr_scheduler, trainloader, testloader, iteration, start_iter, \
                            cur_lambda ) 
    
                # Select the exemplars according to the current model
                X_protoset_cumuls, Y_protoset_cumuls, class_means, alpha_dr_herding = self.set_exemplar_set(cur_model, \
                    is_start_iteration, iteration, last_iter, order, alpha_dr_herding, prototypes)
                
                # Compute the accuracies for current phase
                top1_acc_list_ori, top1_acc_list_cumul = self.compute_acc(class_means, order, order_list, cur_model, X_protoset_cumuls, Y_protoset_cumuls, \
                    X_valid_ori, Y_valid_ori, X_valid_cumul, Y_valid_cumul, iteration, is_start_iteration, top1_acc_list_ori, top1_acc_list_cumul)
    
                # Compute the average accuracy
                num_of_testing = iteration - start_iter + 1
                avg_cumul_acc_fc = np.sum(top1_acc_list_cumul[start_iter:,0])/num_of_testing
                avg_cumul_acc_icarl = np.sum(top1_acc_list_cumul[start_iter:,1])/num_of_testing
                print('Computing average accuracy...')
                print("  Average accuracy (FC)         :\t\t{:.2f} %".format(avg_cumul_acc_fc))
                print("  Average accuracy (Proto)      :\t\t{:.2f} %".format(avg_cumul_acc_icarl))
                wandb.log({"Average accuracy (FC)  ": avg_cumul_acc_fc})
                wandb.log({"Average accuracy (Proto)  ": avg_cumul_acc_icarl})
                # Write the results to the tensorboard
                self.train_writer.add_scalar('avg_acc/fc', float(avg_cumul_acc_fc), iteration)
                self.train_writer.add_scalar('avg_acc/proto', float(avg_cumul_acc_icarl), iteration)
                
        # Save the results and close the tensorboard writer
        torch.save(top1_acc_list_ori, osp.join(self.save_path, 'acc_list_ori.pth'))
        torch.save(top1_acc_list_cumul, osp.join(self.save_path, 'acc_list_cumul.pth'))
        self.train_writer.close()
