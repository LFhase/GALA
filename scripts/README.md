
# Reproducing Results

To generate the data, please follow the instructions given in [dataset_gen/README.md](dataset_gen/README.md).

## Two-piece graphs


### Two-piece graphs {0.8 0.6}

```
# train the proxy model
python3 main_cl.py  -c_in 'raw' -c_rep 'rep'  --seed '[1,2,3,4,5]' --num_layers 3 --pretrain 100 --batch_size 128 --dataset 'tSPMotif' --bias 0.6 --r -1 --save_model --spu_coe 0. --model 'gin' --dropout 0. --commit 'gin_nid_best'  --erm

# use the proxy model as environment assistant
python3 main_cl.py  -c_in 'raw' -c_rep 'rep'  --seed '[1,2,3,4,5]' --num_layers 3 --pretrain 100 --batch_size 128 --dataset 'tSPMotif' --bias 0.6 --r -1 --contrast 0.5 --spu_coe 0. --model 'gin' --dropout 0. -c_sam 'cluster' --num_envs 3 --commit 'gin_nid_best' -pe 10 --ginv_opt 'ciga'
```

### Two-piece graphs {0.8 0.7}

```
# train the proxy model
python3 main_cl.py  -c_in 'raw' -c_rep 'rep'  --seed '[1,2,3,4,5]' --num_layers 3 --pretrain 100 --batch_size 128 --dataset 'tSPMotif' --bias 0.7 --r -1 --save_model --spu_coe 0. --model 'gin' --dropout 0. --commit 'gin_nid_best'  --erm

# use the proxy model as environment assistant
python3 main_cl.py  -c_in 'raw' -c_rep 'rep'  --seed '[1,2,3,4,5]' --num_layers 3 --pretrain 100 --batch_size 128 --dataset 'tSPMotif' --bias 0.7 --r -1 --contrast 1 --spu_coe 0. --model 'gin' --dropout 0. -c_sam 'cluster' --num_envs 3 --commit 'gin_nid_best' -pe 10 --ginv_opt 'ciga'
```

### Two-piece graphs {0.8 0.9}

```
# train the proxy model
python3 main_cl.py -c_in 'raw' -c_rep 'rep'  --seed '[1,2,3,4,5]' --num_layers 3 --pretrain 100 --batch_size 128 --dataset 'tSPMotif' --bias 0.90 --r -1 --save_model --spu_coe 0. --model 'gin' --dropout 0. --commit 'gin_nid_best'  --erm

# use the proxy model as environment assistant
python3 main_cl.py  -c_in 'raw' -c_rep 'rep'  --seed '[1,2,3,4,5]' --num_layers 3 --pretrain 100 --batch_size 128 --dataset 'tSPMotif' --bias 0.90 --r -1 --contrast 0.5 --spu_coe 0. --model 'gin' --dropout 0. -c_sam 'cluster' --num_envs 3 --commit 'gin_nid_best' -pe 10 --ginv_opt 'ciga' -mp 3
```

### Two-piece graphs {0.7 0.9}

```
# train the proxy model
python3 main_cl.py  -c_in 'raw' -c_rep 'rep'  --seed '[1,2,3,4,5]' --num_layers 3 --pretrain 100 --batch_size 128 --dataset 'tSPMotif' --bias 0.9 --r -1 --save_model --spu_coe 0. --model 'gin' --dropout 0. --commit 'gin_nid_best'  --erm

# use the proxy model as environment assistant

python3 main_cl.py  -c_in 'raw' -c_rep 'rep'  --seed '[1,2,3,4,5]' --num_layers 3 --pretrain 100 --batch_size 128 --dataset 'tSPMotif' --bias 0.9 --r -1 --contrast 128 --spu_coe 0. --model 'gin' --dropout 0. -c_sam 'cluster' --num_envs 3 --commit 'gin_nid_best' -pe 10 --ginv_opt 'ciga' -mp 4
```


## EC50

### EC50-Assay

```
# train the proxy model
python -u main_cl.py --save_model  --eval_metric 'auc' --num_layers 4  --batch_size 128 --emb_dim 128 --model 'gin' --pooling 'sum'  -c_dim 128 --dataset 'drugood_lbap_core_ec50_assay' --seed '[1,2,3,4,5]' --dropout 0.5   -c_in 'raw'  -c_rep 'rep'   --spu_coe 0 --ginv_opt 'ciga' --commit 'xgnn_best' -nm


# use the proxy model as environment assistant
python -u main_cl.py --eval_metric 'auc' --num_layers 4  --batch_size 128 --emb_dim 128 --model 'gin' --pooling 'sum' -c_dim 128 --dataset 'drugood_lbap_core_ec50_assay' --seed '[1,2,3,4,5]' --dropout 0.5  --r -1 -c_sam 'cluster' --num_envs 3  --contrast 1 -c_in 'raw'  -c_rep 'rep'   --spu_coe 0 --ginv_opt 'ciga' --commit 'xgnn_best' -mp 2
```

### EC50-Scaffold

```
# train the proxy model
python -u main_cl.py --save_model -ea --eval_metric 'auc' --num_layers 4  --batch_size 128 --emb_dim 128 --model 'gin' --pooling 'sum' -c_dim 128 --dataset 'drugood_lbap_core_ec50_scaffold' --seed '[1,2,3,4,5]' --dropout 0.5 -c_in 'raw'  -c_rep 'rep'   --spu_coe 0 --ginv_opt 'ciga' --commit 'xgnn' -nm

# use the proxy model as environment assistant
python -u main_cl.py --eval_metric 'auc' --num_layers 4  --batch_size 128 --emb_dim 128 --model 'gin' --pooling 'sum' -c_dim 128 --dataset 'drugood_lbap_core_ec50_scaffold' --seed '[1,2,3,4,5]' --dropout 0.5  --r -1 -c_sam 'cluster' --num_envs 2  --contrast 32 -c_in 'raw'  -c_rep 'rep'   --spu_coe 0 --ginv_opt 'ciga' --commit 'xgnn' -mp 1 
```

### EC50-Size

```
# train the proxy model
python -u main_cl.py --save_model -ea --eval_metric 'auc' --num_layers 4  --batch_size 128 --emb_dim 128 --model 'gin' --pooling 'sum' -c_dim 128 --dataset 'drugood_lbap_core_ec50_size' --seed '[1,2,3,4,5]' --dropout 0.5 -c_in 'raw'  -c_rep 'rep'   --spu_coe 0 --ginv_opt 'ciga' --commit 'best' -nm --erm

# use the proxy model as environment assistant
python -u main_cl.py --eval_metric 'auc' --num_layers 4  --batch_size 128 --emb_dim 128 --model 'gin' --pooling 'sum' -c_dim 128 --dataset 'drugood_lbap_core_ec50_size' --seed '[1,2,3,4,5]' --dropout 0.5  --r -1 -c_sam 'cluster' --num_envs 2  --contrast 8 -c_in 'raw'  -c_rep 'rep'   --spu_coe 0 --ginv_opt 'ciga' --commit 'best' -nm 
```

## Ki

### Ki-Assay
```
# train the proxy model
python -u main_cl.py --eval_metric 'auc' --num_layers 4  --batch_size 128 --emb_dim 128 --model 'gin' --pooling 'sum' -c_dim 128 --dataset 'drugood_lbap_core_ki_assay' --seed '[1,2,3,4,5]' --dropout 0.5  --r -1 --commit 'xgnn' -c_in 'raw'  -c_rep 'rep'     --save_model -ea --spu_coe 0 --ginv_opt 'ciga' -nm 

# use the proxy model as environment assistant
python -u main_cl.py --eval_metric 'auc' --num_layers 4  --batch_size 128 --emb_dim 128 --model 'gin' --pooling 'sum' -c_dim 128 --dataset 'drugood_lbap_core_ki_assay' --seed '[1,2,3,4,5]' --dropout 0.5  --r -1 -c_sam 'cluster' --num_envs 10  --contrast 64 -c_in 'raw'  -c_rep 'rep'   --spu_coe 0 --ginv_opt 'ciga' --commit 'xgnn' -nm
```
### Ki-Scaffold

```
# train the proxy model
python -u main_cl.py --save_model -ea --eval_metric 'auc' --num_layers 4  --batch_size 128 --emb_dim 128 --model 'gin' --pooling 'sum' -c_dim 128 --dataset 'drugood_lbap_core_ki_scaffold' --seed '[1,2,3,4,5]' --dropout 0.5 -c_in 'raw'  -c_rep 'rep'   --spu_coe 0 --ginv_opt 'ciga' --commit 'best' -nm --erm

# use the proxy model as environment assistant
python -u main_cl.py --eval_metric 'auc' --num_layers 4  --batch_size 128 --emb_dim 128 --model 'gin' --pooling 'sum' -c_dim 128 --dataset 'drugood_lbap_core_ki_scaffold' --seed '[1,2,3,4,5]' --dropout 0.5  --r -1 -c_sam 'cluster' --num_envs 5  --contrast 128 -c_in 'raw'  -c_rep 'rep'   --spu_coe 0 --ginv_opt 'ciga' --commit 'best' -mp 1
```

### Ki-Size

```
# train the proxy model
python -u main_cl.py --save_model -ea --eval_metric 'auc' --num_layers 4  --batch_size 128 --emb_dim 128 --model 'gin' --pooling 'sum' -c_dim 128 --dataset 'drugood_lbap_core_ki_size' --seed '[1,2,3,4,5]' --dropout 0.5 -c_in 'raw'  -c_rep 'rep'   --spu_coe 0 --ginv_opt 'ciga' --commit 'best' -nm --erm

# use the proxy model as environment assistant
python -u main_cl.py --eval_metric 'auc' --num_layers 4  --batch_size 128 --emb_dim 128 --model 'gin' --pooling 'sum' -c_dim 128 --dataset 'drugood_lbap_core_ki_size' --seed '[1,2,3,4,5]' --dropout 0.5  --r -1 -c_sam 'cluster' --num_envs 10  --contrast 64 -c_in 'raw'  -c_rep 'rep'   --spu_coe 0 --ginv_opt 'ciga' --commit 'best' -nm 
```

## Others


### CMNIST-sp

```
# train the proxy model
python main_cl.py --r 0.8 --num_layers 3  --batch_size 256 --emb_dim 32 --model 'gin' -c_dim 128 --dataset 'CMNIST' --seed '[1,2,3,4,5]' --contrast 0 --spu_coe 0 -c_in 'raw' -c_sam 'cnc' --commit 'xgnn'  -c_rep 'rep'  --save_model -ea

# use the proxy model as environment assistant
python main_cl.py --r 0.8 --num_layers 3  --batch_size 256 --emb_dim 32 --model 'gin' -c_dim 128 --dataset 'CMNIST' --seed '[1,2,3,4,5]' --contrast 8 --spu_coe 0 -c_in 'raw' -c_sam 'cluster' --commit 'xgnn'  -c_rep 'rep' --num_envs 4
```

### Graph-SST2
```
# train the proxy model
python main_cl.py --r 0.6 --num_layers 3  --batch_size 128 --emb_dim 128 --model 'gin' -c_dim 128 --dataset 'Graph-SST2' --seed '[1,2,3,4,5]' --contrast 0 --spu_coe 0 -c_in 'feat'  -c_rep 'feat' --commit 'xgnn' --save_model -ea

# use the proxy model as environment assistant
python main_cl.py --r 0.6 --num_layers 3  --batch_size 128 --emb_dim 128 --model 'gin' -c_dim 128 --dataset 'Graph-SST2' --seed '[1,2,3,4,5]' --contrast 2  --spu_coe 0 -c_in 'feat'  -c_rep 'feat' -c_sam 'cluster' --num_envs 2 --commit 'xgnn'
```
