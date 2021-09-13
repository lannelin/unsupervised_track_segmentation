# Scripts

## kitti_eval.py

(WIP)

Evaluate against a sample of the KITTI dataset (training data).

### Prereq

1. Need to download the data
   from [here](http://www.cvlibs.net/datasets/kitti/eval_instance_seg.php?benchmark=instanceSeg2015)
   and unpack to [data](/data). This should give the structure:

```
├── data
└── data_semantics
     ├── testing
     │   └── image_2
     └── training
         ├── image_2
         ├── instance
         ├── semantic
         └── semantic_rgb
```

2. The script uses [wandb](https://wandb.ai/), you should login on the cli before
   running. The project name is set in the config file.

### Config

The script uses a config, located [here](/config/config.yml)

### Running

The script uses [hydra](https://hydra.cc/).

We run the script from the config values with:

```bash
python kitti_eval.py
```

or take advantage of some hydra features, such as sweeping over config values. Here, we
sweep over two values for `model.max_steps` and two values for `model.n_channels`:

```bash
python kitti_eval.py --multirun ++model.max_steps=5,10 ++model.n_channels=24,48
```

### Results

Results can be viewed on your [wandb](https://wandb.ai/) dashboard for the project. The
project name is set in the config file.

Historic results from experiment runs will summarised in [results](/results). 
