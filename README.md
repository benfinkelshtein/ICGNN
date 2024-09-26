# Learning on Large Graphs using Intersecting Communities

This repository contains the official code base of the paper **[Learning on Large Graphs sing Intersecting Communities](https://arxiv.org/abs/2405.20724)**, accepted to NeurIPS 2024.

[PyG]: https://pytorch-geometric.readthedocs.io/en/latest/

## Installation ##
To reproduce the results please use Python 3.9, PyTorch version 2.3.0, Cuda 11.8, PyG version 2.5.3, and torchmetrics.

```bash
pip3 install torch==2.3.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cu118.html
pip3 install torch-geometric==2.5.3
pip3 install torchmetrics
pip3 install pandas
pip3 install neptune
pip3 install matplotlib
pip3 install einops
pip3 install git+https://github.com/TorchSpatiotemporal/tsl.git
```

## Running

### ICG approximation - node classification and spatio-temporal experiments 

The script we use to run the ICG approximation is ``./icg_approximation_main.py``.
Note that the script should be run with ``.`` as the main directory or source root.

The parameters of the script are:

- ``--dataset_type``: name of the dataset.
The available options are: communities (a synthetic dataset to test the code with), tolokers, squirrel, twitch_gamers, bay, la.

- ``--icg_approx_epochs``: the number of epochs for the approximation.
- ``--icg_approx_lr``: the lr for the approximation.
- ``--loss_scale``: the scaler $\lambda$ between the graph and signal loss.
- ``--cut_norm``: whether or not to calculate the cut norm using the cutnorm package https://github.com/pingkoc/cutnorm/tree/master (which is already inserted to this repo.
- ``--add_eigen``: whether or not to use the eigen initialization.
- ``--node_drop_ratio``: the ratio of nodes dropped from the graph.
- ``--num_communities``: the number of communities used in the approximation.
- ``--encode_dim``: the output dimension of the linear encoder that is applied to the features. If set to zero, no encoding is applied.
- ``--seed``: a seed to set random processes.

To perform experiments over the tolokers dataset for 50 communities, 10 epochs, a lr of 0.001 and $\lambda=0.5$, while removing 0% of the graph: 
```bash
python -u icg_approximation_main.py --dataset_type tolokers --num_communities 50 --icg_approx_epochs 10 --icg_approx_lr 0.001 --loss_scale 0.5 --node_drop_ratio 0.0
```

### ICG-NN - node classification experiments

After an ICG has been trained. The script we use to run ICG-NN is ``./icgnn_main.py``.
Note that the script should be run with ``.`` as the main directory or source root.

Make sure that the following parameters match those used for the ICG approximation:

- ``--dataset_type``: name of the dataset.
The available options are: communities (a synthetic dataset to test the code with), tolokers, squirrel, twitch_gamers, bay, la.
- ``--icg_approx_epochs``: the number of epochs used for the approximation.
- ``--icg_approx_lr``: the lr for the approximation.
- ``--loss_scale``: the scaler $\lambda$ between the graph and signal loss.
- ``--add_eigen``: whether or not to use the eigen initialization.
- ``--node_drop_ratio``: the ratio of nodes dropped from the graph.
- ``--num_communities``: the number of communities used in the approximation.
- ``--encode_dim``: the output dimension of the linear encoder that is applied to the features. If set to zero, no encoding is applied.
- ``--seed``: a seed to set random processes.

The new parameters for the scripts are:
- ``--num_layers``: the number of ICG-NN layers.
- ``--icgnn_type``: the ICG-NN variation.
The available options are: Matrix for ICG$_u$-NN or MHA for ICG-NN.
- ``--nn_num_layers``: the number of MultiHeadAttention layers used in a single ICG-NN block.
- ``--hidden_dim``: the hidden dimension.
- ``--dropout``: the dropout ratio.
- ``--skip``: whether or not to include a residual connection.
- ``--epochs``: the number of epochs used for the fitting.
- ``--lr``: the learning used for the fitting.

To perform experiments over a 3 layered ICG$_u$-NN model with a hidden dimension of 128 on the tolokers dataset for 300 epochs with a lr of 0.03 using the previous ICG approximation: 
```bash
python -u icgnn_main.py --dataset_type tolokers --num_communities 50 --icg_approx_epochs 10 --icg_approx_lr 0.001 --loss_scale 0.5 --node_drop_ratio 0.0 --num_layers 3 --icgnn_type Matrix --hidden_dim 128 --epochs 300 --lr 0.03
```

### ICG-NN - spatio-temporal experiments

After an ICG has been trained. We build over the codebase https://github.com/Graph-Machine-Learning-Group/taming-local-effects-stgnns for spatio-temporal experiments.
The script we use to run ICG-NN is ``./icgnn_spatio_temporal_main.py``.
Note that the script should be run with ``.`` as the main directory or source root.
Also note that subsampling is not implemented for these datasets (node_drop_ratio=0) and nodes shouldn't be removed from the graph ($\lambda=0$). 

Make sure that the following parameters match those used for the ICG approximation:

- ``dataset=``: name of the dataset.
The available options are: communities (a synthetic dataset to test the code with), tolokers, squirrel, twitch_gamers, bay, la.
- ``model.icg_approx_train_args.icg_approx_epochs``: the number of epochs used for the approximation.
- ``model.icg_approx_train_args.icg_approx_lr``: the lr for the approximation.
- ``model.icg_approx_train_args.loss_scale``: the scaler $\lambda$ between the graph and signal loss.
- ``model.icg_approx_args.add_eigen=``: whether or not to use the eigen initialization.
- ``model.icg_approx_args.num_communities=``: the number of communities used in the approximation.
- ``model.icg_approx_args.encode_dim=``: the output dimension of the linear encoder that is applied to the features. If set to zero, no encoding is applied.

The new parameters for the scripts are:
- ``model.icgnn_args.num_layers=``: the number of ICG-NN layers.
- ``model.icgnn_args.icgnn_type=``: the ICG-NN variation.
The available options are: Matrix for ICG$_u$-NN or MHA for ICG-NN.
- ``model.icgnn_args.nn_num_layers=``: the number of MultiHeadAttention layers used in a single ICG-NN block.
- ``model.hidden_dim=``: the hidden dimension.
- ``model.icgnn_args.dropout=``: the dropout ratio.
- ``model.icgnn_args.skip=``: whether or not to include a residual connection.
- ``epochs=``: the number of epochs used for the fitting.
- ``optimizer.hparams.lr=``: the learning used for the fitting.

To perform experiments over a 3 layered ICG$_u$-NN model with a hidden dimension of 128 on the METR-LA dataset for 300 epochs with a lr of 0.03 using an ICG approximation with 50 communities, 10 epochs and a lr of 0.001: 
```bash
python -u icg_approximation_main.py --dataset_type la --num_communities 50 --icg_approx_epochs 10 --icg_approx_lr 0.001 --loss_scale 0.0 --node_drop_ratio 0.0
python -u icgnn_spatio_temporal_main.py config=benchmarks model=icgnn dataset=la model.icg_approx_args.num_communities=50 model.icg_approx_train_args.epochs=10 model.icg_approx_train_args.lr=0.001 model.icg_approx_train_args.loss_scale=0.0 model.icgnn_args.num_layers=3 model.icgnn_args.icgnn_type=Matrix model.hidden_dim=128 epochs=300 optimizer.hparams.lr=0.03
```

## Cite

If you make use of this code, or its accompanying [paper](https://arxiv.org/abs/2405.20724), please cite this work as follows:
```bibtex
@inproceedings{finkelshtein2024learninglargegraphs,
  title={Learning on Large Graphs using Intersecting Communities}, 
  author={Ben Finkelshtein and İsmail İlkan Ceylan and Michael Bronstein and Ron Levie},
  year = "2024",
  booktitle = "Proceedings of 38th Conference on Neural Information Processing Systems (NeurIPS 2024)",
  url = "https://arxiv.org/abs/2310.01267",
}
```
