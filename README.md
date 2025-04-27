# Who Dominates Social Influence? Impact Differences of Humans and Bots in Online Social Networks

The software and data in this repository are the materials used in the research paper titled *Who Dominates Social Influence? Impact Differences of Humans and Bots in Online Social Networks*.

## Computing Requirements

We highly recommend using `Conda` for installing requirements and implementation (an open-source package and environment management system for `Python`). For installing `Conda`, please refer to the website https://www.anaconda.com/products/distribution. To ensure that the code runs successfully, the following dependencies need to be installed:

> python 3.8.17 \
> numpy 1.24.4 \
> pytorch 1.13.1 \
> networkx 3.1 \
> scipy 1.10.1 \
> scikit-learn 1.3.0 \
> torch-geometric 2.3.1


Note: **The torch_geometric package requires matching CUDA and pytorch versions** to run. Our code is run under the GPU of GeForce RTX 4080, with CUDA Version 12.2. A choice of matching packages for torch-geometric 2.3.1 is: torch-cluster 1.6.1, torch-scatter 2.1.1, torch-sparse 0.6.17, torch-spline-conv 1.2.2.

If you want to run the code locally, please check the suitable packages matching version at: https://pytorch-geometric.com/whl/, and install the matched version (with your Pytorch and CUDA version). After installing the torch-cluster, torch-scatter, torch-sparse, and torch-spline-conv, you can run the comment to install torch_geometric:

```
pip install torch-geometric
```

For those who want to directly export the environment, we also provide a YAML file. For torch_geometric implementation details, see the documentation: https://pytorch-geometric.readthedocs.io/en/latest/index.html. 

## File Structures

To implement the AdvG framework, practitioners can refer to the following files:

* **AdvG.py**: The proposed AdvG model Framework.
* **train_simulation.py**: Training the AdvG model for simulation data.
* **train_empirical.py**: Training the AdvG model for empirical data.
* **utils.py**: Reusable modules for model training and evaluation.
* **evaluate.py**: Evaluation module.
* **link_pred_check.py**: Performing the link prediction check between bots and human accounts.
* **simulation.py**: Generate simulation data for model validation.
* **load_data.py**: Module for loading necessary data.
* Dataset/twi22: This contains the demo data with different discussion domains.

We provide the demo data following the appointment. To use the data for your own research, please refer to the repository https://twibot22.github.io/ and cite the following paper:

*Feng, S., Tan, Z., Wan, H., Wang, N., Chen, Z., Zhang, B., ...    & Luo, M. (2022). **Twibot-22: Towards graph-based twitter bot detection**. Advances in Neural Information Processing Systems, 35, 35254-35269.*

For any use of the entire Twibot22 dataset, please contact the authors in the above paper. The MGTAB dataset is provided in the following paper, and please ensure to make the citation:

*Shi, S., Qiao, K., Chen, J., Yang, S., Yang, J., Song, B., ... & Yan, B. (2023). **Mgtab: A multi-relational graph-based twitter account detection benchmark**. arXiv preprint arXiv:2301.01123.*

## Implementations

**We provide a simple implementation of our framework. For simulation experiments, practitioners should first generate their synthetic data.** We provide some (scenario) examples, including 'random', 'randomu', 'highdu', 'highbc', or 'highcc'. E.g., a quick run with an argument specifying the simulation scenario:

```
python simulation.py --type 'random'
```

**The simulation data will be stored at `Dataset/synthetic`**. 

Note: Simulating 100 networks for repeated experiments takes about 5-8 minutes on our device (for more complex calculations, such as 'highbc', this may take longer).

Then, **the model can be trained on the simulation networks by specifying the data type 'random', 'randomu', 'highdu', 'highbc', or 'highcc'**. E.g.,

```
python train_simulation.py --type 'random' --effect_true -1
```

Note: For the simulation experiments, researchers can set the ground-truth causal effect by the argument '--effect_true' based on their simulation settings. The code will expected to output the estimated eATE and ePEHE score. It takes about 1-3 minutes to train on a given network on our device. For researchers who want to run a quick demo on their laptop, they can reduce the number of repeats or network sizes.

Using Twibot22 as an example, **to estimate the causal impact differences, please use the following commands**:

```
python train_empirical.py --type 't1_pos'
```

Note: argument --type is for specifying the data used. With 't1', 't2', and 't3' representing the discussion domains 'Pandemic', 'War', and 'Climate'. '_pos', '_neg' represent the scenario that influencers express positive/negative attitudes. The code will output the estimated potential outcomes for the treatment and control groups. The results may show some stochasticity on different devices due to the random parameter initialization process of neural networks. Running on the entire large-scale network takes 5-8 minutes on our device.


**For the tie predictability check, please run the following command**:

```
python link_pred_check.py --type 't1_pos'
```

Note: Our predictability tests can be compared under the same model settings. Other settings that are different from the paper can also be tried.

## Extensions

**We highly encourage researchers to build on our foundation and extend more broadly (both empirical and methodological) if you find the work interesting. Our current code provides various settings and configurations**, some brief examples are as follows.

For social network and causal effect simulations, provide the following arguments to `simulation.py`. Researchers are also encouraged to apply our framework to other network scenarios (estimating the differences in effects among different participants) and data.

```
usage: simulation.py [-h] [--type TYPE] [--sample_user SAMPLE_USER] [--sample_bot SAMPLE_BOT]
                    [--betaZ BETAZ] [--betaT BETAT] [--betaB BETAB] [--EPSILON EPSILON]

optional arguments:
  -h, --help               show this help message and exit
  --type TYPE              simulation network type (default: random)
  --sample_user SAMPLE_USER
                           number of human users in a network
  --sample_bot SAMPLE_BOT  number of bots in a network
  --betaZ BETAZ            causal influence of latent trait
  --betaT BETAT            causal influence of a human influencer
  --betaB BETAB            causal influence of a bot influencer
  --EPSILON EPSILON        random disturbance
```

This helps you test results in our simulations and develop further configurations or settings.

For training the model, provide the following arguments to `train_xxx.py`.

```
usage: train_xxx.py [-h] [--type TYPE] [--effect_true EFFECT_TRUE] [--mask_homo MASK_HOMO] [--hm HM]
                [--dzm DZM] [--hd HD] [--he HE] [--ho HO] [--gpu GPU] [--ljt LJT]
                [--ljg LJG] [--ljd LJD] [--train_prec TRAIN_PREC] [--lr_mf LR_MF] [--lr_mg LR_MG]
                [--lr_md LR_MD] [--max_epoch MAX_EPOCH]

optional arguments:
  -h, --help                  show this help message and exit
  --type TYPE                 data used (enter the name of domain or datasets)
  --effect_true EFFECT_TRUE   ground-truth casual effect (can be set to 0 for empirical analysis without ground-truth labels)
  --mask_homo MASK_HOMO       edge permutation percentage @ Homphily Edge Detector
  --hm HM                     hidden dimension of the Homphily Edge Detector 
  --dzm DZM                   output dimension of the Homphily Edge Detector 
  --hd HD                     dimension of the discriminator
  --he HE                     dimension of the graph feature encoder
  --ho HO                     hidden dimension of the outcome generator
  --gpu GPU                   id of your gpu device
  --ljt LJT                   regularization coefficient for treatment prediction
  --ljg LJG                   regularization coefficient for counterfactual sample generation
  --ljd LJD                   regularization coefficient for counterfactual discrimination
  --train_prec TRAIN_PREC     precentage of training samples
  --lr_mf LR_MF               learning rate of for theta_f
  --lr_mg LR_MG               learning rate of for theta_g
  --lr_md LR_MD               learning rate of for theta_d
  --max_epoch MAX_EPOCH       max training epochs
```

## Copyrights

This is a temporal version for a manuscript under review. The repository will be officially released and set permanently public after a formal acceptance by an academic journal for researchers in the relevant fields to implement, practically use, and conduct their extended research.

## Citations

The recommended citations will be provided after publication.

## Acknowledgement

We thank the authors of the paper **Twibot-22: Towards graph-based twitter bot detection** for providing us the valuable, entire networks to conduct our research, and the authors in **Mgtab: A multi-relational graph-based twitter account detection benchmark.** for the source of their complete dataset.

We also like to thank the editors and anonymous reviewers for their detailed and constructive feedback in helping improve the paper.

## Development and Support

This code may be further developed and refined to make it user-friendly and extended to more interesting configurations.
