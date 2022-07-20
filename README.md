## D^3Former: Debiased Dual Distilled Transformer for Incremental Learning

Class incremental learning (CIL) involves learning a classification model where groups of new classes are encountered in every learning phase. The goal is to learn a unified model performant on all the classes observed so far. Given the recent popularity of Vision Transformers (ViTs) in conventional classification settings, an interesting question is to study their continual learning behaviour. In this work, we develop a Debiased Dual Distilled Transformer for CIL dubbed D^3Former. The proposed model leverages a hybrid nested ViT design to ensure data efficiency and scalability to small as well as large datasets. In contrast to a recent ViT based CIL approach, our D^3Former does not dynamically expand its architecture when new tasks are learned and remains suitable for a large number of incremental tasks. The improved CIL behaviour of D^3Former owes to two fundamental changes to the ViT design. First, we treat the incremental learning as a long-tail classification problem where the majority samples from new classes vastly outnumber the limited exemplars available for old classes. To avoid biasness against the minority old classes, we propose to dynamically adjust logits to emphasize on retaining the representations relevant to old tasks. Second, we propose to preserve the configuration of spatial attention maps as the learning progresses across tasks. This helps in reducing catastrophic forgetting via constraining the model to retain the attention on the most discriminative regions. D^3Former obtains favorable results on incremental versions of CIFAR-100, MNIST, SVHN,  and ImageNet datasets.


## Setup

A yml file has been provided for the conda environment

## CIFAR-100

```
python3 main.py --gpu 0 --dataset cifar100 --nb_cl_fg 50 --nb_cl 10 --the_lambda 10 --tau 1 --gamma 0.1 --warmup 10
```


## ImageNet subset-100

```
python3 main.py --gpu 0 --dataset imagenet_sub --nb_cl_fg 50 --nb_cl 10 --the_lambda 4 --tau 0.3 --gamma 0.05 --warmup 20
```


## ImageNet-1K

```
python3 main.py --gpu 0 --dataset imagenet --nb_cl_fg 500 --nb_cl 100 --the_lambda 4 --tau 0.3 --gamma 0.05 --warmup 20
```

## Acknowledgements

We are thankful to the authors of [AANet](https://github.com/yaoyao-liu/class-incremental-learning/tree/main/adaptive-aggregation-networks) for their implementation.
