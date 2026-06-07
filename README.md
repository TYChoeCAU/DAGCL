# DAGCL: Degree-Adaptive Graph Contrastive Learning for Recommendation


## Requirements

```bash
# 1. PyTorch (CUDA 11.7)
pip install torch==1.13.1+cu117 --index-url https://download.pytorch.org/whl/cu117

# 2. PyTorch Geometric
pip install torch-scatter==2.1.1 torch-sparse==0.6.17 torch-geometric==2.5.2 \
    -f https://data.pyg.org/whl/torch-1.13.1+cu117.html

# 3. RecBole + others
pip install recbole==1.1.1 numpy==1.25.2 pandas==2.2.3 scipy==1.10.1 \
    tqdm==4.67.3 colorlog==4.7.2 tensorboard==2.20.0
```

Or:
```bash
pip install -r requirements.txt
```

## Dataset

Place datasets under `dataset/`. Supported: `ali`, `gowalla`, `yelp2018`.

Each dataset directory requires an `.inter` file (RecBole interaction format, tab-separated with `user_id:token`, `item_id:token`, `rating:float`, `timestamp:float`).
The processed Gowalla dataset is included as an example benchmark. Ali and Yelp2018 follow the same data format and can be added under `dataset/` when needed.


```
dataset/
├── ali/
│   └── ali.inter
├── gowalla/
│   └── gowalla.inter
└── yelp2018/
    └── yelp2018.inter
```

## Run

### Gowalla

```bash
python run_recbole_gnn.py \
  --model=DAGCL --dataset=gowalla --gpu_id=0 --seed=2021 \
  --embedding_size=64 --n_layers=3 --layer_cl=2 \
  --lambda=0.1 --eps=0.2 --temperature=0.12 \
  --degree_adaptive_eps_alpha=1.5 \
  --degree_eps_min_scale=0.5 --degree_eps_max_scale=3.0 \
  --cl_head_weight=0.2 --head_ratio_u=0.3 \
  --item_cl_head_weight=1.0 --item_head_ratio=0.0 \
  --valid_metric=Recall@10 --topk=[10,20] \
  --show_progress=False --stopping_step=10
```


### Hyperparameter Settings

| Dataset  | lambda | eps | temperature | degree_adaptive_eps_alpha | degree_eps_min_scale | degree_eps_max_scale | cl_head_weight | head_ratio_u | item_cl_head_weight | item_head_ratio |
| -------- | -----: | --: | ----------: | ------------------------: | -------------------: | -------------------: | -------------: | -----------: | ------------------: | --------------: |
| Gowalla  |    0.1 | 0.2 |        0.12 |                       1.5 |                  0.5 |                  3.0 |            0.2 |          0.3 |                 1.0 |             0.0 |
| Ali      |   0.05 | 0.1 |        0.15 |                       1.0 |                  0.5 |                  3.0 |            0.1 |          0.1 |                 1.0 |             0.0 |
| Yelp2018 |   0.05 | 0.1 |        0.15 |                       1.0 |                  0.5 |                  3.0 |            0.3 |          0.3 |                 1.0 |             0.0 |

The following settings are shared across all datasets.

| Parameter      |     Value |
| -------------- | --------: |
| embedding_size |        64 |
| n_layers       |         3 |
| layer_cl       |         2 |
| valid_metric   | Recall@10 |
| topk           |   [10,20] |
| show_progress  |     False |
| stopping_step  |        10 |
| seed           |      2021 |

```


## Key Parameters

| Parameter | Description |
|-----------|-------------|
| `--lambda` | Overall CL loss weight |
| `--eps` | Base perturbation magnitude |
| `--temperature` | InfoNCE temperature |
| `--layer_cl` | Layer index from which CL embeddings are taken |
| `--degree_adaptive_eps_alpha` | Degree-scaling exponent for eps (0 = disabled) |
| `--degree_eps_min_scale` | Min eps scale factor for low-degree nodes |
| `--degree_eps_max_scale` | Max eps scale factor for high-degree nodes |
| `--cl_head_weight` | CL loss weight for head users (< 1 = downweight) |
| `--head_ratio_u` | Top-k ratio of users treated as head (e.g. 0.3 = top 30%) |
| `--item_cl_head_weight` | CL loss weight for head items |
| `--item_head_ratio` | Top-k ratio of items treated as head |


