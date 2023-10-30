
# Data Preparation

Our datasets are generated as the following procedures, following the practice of [CIGA](https://github.com/LFhase/CIGA).
The main difference in GALA is the generation of two-piece graphs, which is modified from SPMotif dataset generation.

## Two-Piece Datasets

The `global_b` controls the spurious correlation strength while the `label_noise` controls the invariant correlation strength.
The four datasets used in our paper are generated with

`two-piece graph {0.8,0.6}`:

- global_b=0.6;
- label_noise=0.2;


`two-piece graph {0.8,0.7}`:

- global_b=0.7;
- label_noise=0.2;

`two-piece graph {0.8,0.9}`:

- global_b=0.9;
- label_noise=0.2;

`two-piece graph {0.7,0.9}`:

- global_b=0.9;
- label_noise=0.3;

and then running the following command:

```bash
cd dataset_gen
python gen_basis.py
```

The generated data will be stored as in `./data/tSPMotif-{global_b}` at the root directory of this repo.
To use the dataset in `main.py`, specify the `--dataset` option and `--bias` option as `mSPMotif` and a corresponding bias, respectively.
Feel free to play with other parameter settings!

## DrugOOD Datasets

To obtain the DrugOOD datasets tested in our paper, i.e., `drugood_lbap_core_ec50_assay`, `drugood_lbap_core_ec50_scaffold`
and `drugood_lbap_core_ec50_size`,
we use the DrugOOD curation codes based on the commit `eeb00b8da7646e1947ca7aec93041052a48bd45e` and `chembl_30` database.
After curating the datasets, put the corresponding json files under `./data/DrugOOD`,
and specify the `--dataset` option as the corresponding dataset name to use, e.g., `drugood_lbap_core_ec50_assay`.

Similar operations can be performed to generate `ki` datasets.

## CMNIST-sp

The CMNIST dataset is generated following the Invariant Risk Minimization
and then converted into graphs using the SLIC superpixels algorithm.
To generate the dataset, simply run the codes as the following:

```bash
cd dataset_gen
python prepare_mnist.py  --dataset 'cmnist'  -t 8 -s 'train'
python prepare_mnist.py  --dataset 'cmnist'  -t 8 -s 'test'
```

and the generated data will be put into `./data/CMNISTSP` at the root directory of this repo.
Note that two auxiliary datasets `./data/MNIST` and `./data/ColoredMNIST` will also be created as the base for the generation of `./data/CMNISTSP`.
To use the dataset, simply specify `--dataset` option as `CMNIST`.

## Graph-SST2

Both of `Graph-SST2` and `Twitter` are based on the datasets provided by [DIG](https://github.com/divelab/DIG).
To get the datasets, you may download via this [link](https://drive.google.com/drive/folders/1dt0aGMBvCEUYzaG00TYu1D03GPO7305z)
provided by DIG and the GNN explainability survey authors.
Then unzip the data into `./data/Graph-SST2/raw`.
By specifying `--dataset` as the dataset name in `main.py`, the data loading process will
add the degree biases automatically.

## NCI1, NCI109, PROTEINS and DD

We use the datasets provided by [size-invariant-GNNs](https://github.com/PurdueMINDS/size-invariant-GNNs) authors,
who already sampled the datasets with graph size distribution shifts injected.
The datasets can be downloaed via this [link](https://www.dropbox.com/s/38eg3twe4dd1hbt/data.zip).
After downloading, simply unzip the datasets into `./data/TU`.
To use the datasets, simply specify `--dataset` as the dataset name in `main.py`.
