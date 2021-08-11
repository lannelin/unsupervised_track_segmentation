# Scripts

## kitti_eval.py

(WIP)

Evaluate against a sample of the KITTI dataset (training data).

Need to download the data from [here](http://www.cvlibs.net/datasets/kitti/eval_instance_seg.php?benchmark=instanceSeg2015)
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