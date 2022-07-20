## D^3Former: Debiased Dual Distilled Transformer for Incremental Learning

![main figure](images/architecture.png)
> *<div style="text-align: justify"> **Abstract:** Class incremental learning (CIL) involves learning a classification model where groups of new classes are encountered in every learning phase. The goal is to learn a unified model performant on all the classes observed so far. Given the recent popularity of Vision Transformers (ViTs) in conventional classification settings, an interesting question is to study their continual learning behaviour. In this work, we develop a Debiased Dual Distilled Transformer for CIL dubbed D^3Former. The proposed model leverages a hybrid nested ViT design to ensure data efficiency and scalability to small as well as large datasets. In contrast to a recent ViT based CIL approach, our D^3Former does not dynamically expand its architecture when new tasks are learned and remains suitable for a large number of incremental tasks. The improved CIL behaviour of D^3Former owes to two fundamental changes to the ViT design. First, we treat the incremental learning as a long-tail classification problem where the majority samples from new classes vastly outnumber the limited exemplars available for old classes. To avoid biasness against the minority old classes, we propose to dynamically adjust logits to emphasize on retaining the representations relevant to old tasks. Second, we propose to preserve the configuration of spatial attention maps as the learning progresses across tasks. This helps in reducing catastrophic forgetting via constraining the model to retain the attention on the most discriminative regions. D^3Former obtains favorable results on incremental versions of CIFAR-100, MNIST, SVHN,  and ImageNet datasets. </div>*

<hr />

# :rocket: News
* **(July 20, 2022)** 
  * Code released
<hr />

## Installation

Our code has been tested on CUDA 11.3 and pytorch version 1.10.1

For easier sake, we also provide a [yml file](incremental.yml). to recreate the conda environment

## Results

![Results](images/acc_plot.png)

<strong> D^3Former performance on small scale datasets: </strong> Plots showing task wise accuracy for
different number of incremental tasks for CIFAR-100. D^3Former achieves relatively high accuracy
compared to other state-of-the-art methods when adding 10, 5 and 2 classes per task.

![Results](images/cifar100.png)
Results of <strong>CIFAR-100</strong> with Average accuracy (%), last phase accuracy (%) and forgetting
rate F(%) of different methods in 5,10 and 25 tasks settings.

![Results](images/imagenet100.png)
Results of <strong>ImageNet100</strong> with Average accuracy (%), last phase accuracy (%) and forgetting
rate F(%) of different methods in 5,10 and 25 tasks settings.

## Training

<strong> For CIFAR-100</strong>

```
python3 main.py --gpu 0 --dataset cifar100 --nb_cl_fg 50 --nb_cl 10 --the_lambda 10 --tau 1 --gamma 0.1 --warmup 10
```


<strong>ImageNet subset-100</strong>

```
python3 main.py --gpu 0 --dataset imagenet_sub --nb_cl_fg 50 --nb_cl 10 --the_lambda 4 --tau 0.3 --gamma 0.05 --warmup 20
```


<strong>ImageNet-1K</strong>

```
python3 main.py --gpu 0 --dataset imagenet --nb_cl_fg 500 --nb_cl 100 --the_lambda 4 --tau 0.3 --gamma 0.05 --warmup 20
```

## Acknowledgement

Our code is built upon [AANet](https://github.com/yaoyao-liu/class-incremental-learning/tree/main/adaptive-aggregation-networks). We would like to thank the authors for their implementation.

## Contact
Please feel free to create an issue here or contact us at abdelrahman.mohamed@mbzuai.ac.ae or rushali.grandhe@mbzuai.ac.ae
