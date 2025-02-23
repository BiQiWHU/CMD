# [TIP 2025] Cross-level Multi-instance Distillation for Self-supervised Fine-grained Visual Categorization

This is the official implementation of our work entitled as ```Cross-level Multi-instance Distillation for Self-supervised Fine-grained Visual Categorization```, which is going to be published on ```IEEE Transactions on Image Processing (TIP'2025)``` soon.

## Implementation of Cross-level Multi-instance Distillation (CMD)

To set up the environment, please install the following packages:
```
opencv-python
pyyaml
json_tricks
tensorboardX==2.0
yacs
pycocotools
scikit-learn
tensorwatch
pandas
timm==0.3.2
numpy==1.19.3
einops
```

![avatar](/framework.png)

The training and inference command is:

```python main.py --c configs/MTARSI_SwinT.yaml```

Please remember to change the file folder to your own in the ```.yaml``` file.

## Citation

If you find this work useful for your research, please cite our work as follows:

```BibTeX
@article{bi2025universal,
  title={Universal Fine-grained Visual Categorization by Concept Guided Learning},
  author={Bi, Qi and Yi, Jingjun and Zhan, Haolan and Wei, Ji and Xia, Gui-Song},
  journal={IEEE Transactions on Image Processing},
  year={2025}
}
```

