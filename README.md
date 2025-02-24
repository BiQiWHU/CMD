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

The command for self-supervised pre-training is:

```
PROJ_PATH=your_esvit_project_path
DATA_PATH=$PROJ_PATH/project/data/imagenet

OUT_PATH=$PROJ_PATH/output/esvit_exp/ssl/swin_tiny_imagenet/
python -m torch.distributed.launch --nproc_per_node=16 main_esvit.py --arch swin_tiny --data_path $DATA_PATH/train --output_dir $OUT_PATH --batch_size_per_gpu 32 --epochs 300 --teacher_temp 0.07 --warmup_epochs 10 --warmup_teacher_temp_epochs 30 --norm_last_layer false --use_dense_prediction True --cfg experiments/imagenet/swin/swin_tiny_patch4_window7_224.yaml
```

Please remember to change the file folder to your own in the ```.yaml``` file.

The command for inference is:

```
PROJ_PATH=your_esvit_project_path
DATA_PATH=$PROJ_PATH/project/data/imagenet

OUT_PATH=$PROJ_PATH/exp_output/esvit_exp/swin/swin_tiny/bl_lr0.0005_gpu16_bs32_dense_multicrop_epoch300
CKPT_PATH=$PROJ_PATH/exp_output/esvit_exp/swin/swin_tiny/bl_lr0.0005_gpu16_bs32_dense_multicrop_epoch300/checkpoint.pth

python -m torch.distributed.launch --nproc_per_node=4 eval_linear.py --data_path $DATA_PATH --output_dir $OUT_PATH/lincls/epoch0300 --pretrained_weights $CKPT_PATH --checkpoint_key teacher --batch_size_per_gpu 256 --arch swin_tiny --cfg experiments/imagenet/swin/swin_tiny_patch4_window7_224.yaml --n_last_blocks 4 --num_labels 1000 MODEL.NUM_CLASSES 0

python -m torch.distributed.launch --nproc_per_node=4 eval_knn.py --data_path $DATA_PATH --dump_features $OUT_PATH/features/epoch0300 --pretrained_weights $CKPT_PATH --checkpoint_key teacher --batch_size_per_gpu 256 --arch swin_tiny --cfg experiments/imagenet/swin/swin_tiny_patch4_window7_224.yaml MODEL.NUM_CLASSES 0
```

Please remember to change the file folder to your own in the ```.yaml``` file.

## Acknowledgement

The development of ```CMD``` is based on the source code from ```EsViT```, with the code link [https://github.com/microsoft/esvit]. We sincerely appreciate the authors of ```Efficient Self-supervised Vision Transformers for Representation Learning``` to advance self-supervised representation learning. 

## Citation

If you find this work useful for your research, please cite our work as follows:

```BibTeX
@article{bi2025cross,
  title={Cross-level Multi-instance Distillation for Self-supervised Fine-grained Visual Categorization},
  author={Bi, Qi and Ji, Wei and Yi, Jingjun and Zhan, Haolan and Xia, Gui-Song},
  journal={IEEE Transactions on Image Processing},
  year={2025}
}
```

