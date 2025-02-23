@REM python -m torch.distributed.launch --nproc_per_node=2 eval_linear_5.py --data_path E:/FineGrained/datasets/102Flowers --output_dir output_102Flowers/eval_linear_more --pretrained_weights output_102Flowers/checkpoint0270.pth --checkpoint_key teacher --batch_size_per_gpu 8 --arch swin_tiny --cfg experiments/imagenet/swin/swin_tiny_patch4_window14_224.yaml --n_last_blocks 4 --num_labels 102 --num_workers 1 MODEL.NUM_CLASSES 0
@REM python -m torch.distributed.launch --nproc_per_node=2 eval_knn_5.py --data_path E:/FineGrained/datasets/102Flowers --dump_features output_102Flowers/eval_knn_more --pretrained_weights output_102Flowers/checkpoint0270.pth --checkpoint_key teacher --batch_size_per_gpu 8 --arch swin_tiny --cfg experiments/imagenet/swin/swin_tiny_patch4_window14_224.yaml --num_workers 1 MODEL.NUM_CLASSES 0
@REM python -m torch.distributed.launch --nproc_per_node=2 eval_linear_5.py --data_path E:/FineGrained/datasets/FGVC-Aircrafts --output_dir output_Aircrafts/eval_linear_more --pretrained_weights output_Aircrafts/checkpoint0270.pth --checkpoint_key teacher --batch_size_per_gpu 8 --arch swin_tiny --cfg experiments/imagenet/swin/swin_tiny_patch4_window14_224.yaml --n_last_blocks 4 --num_labels 100 --num_workers 1 MODEL.NUM_CLASSES 0
@REM python -m torch.distributed.launch --nproc_per_node=2 eval_knn_5.py --data_path E:/FineGrained/datasets/FGVC-Aircrafts --dump_features output_Aircrafts/eval_knn_more --pretrained_weights output_Aircrafts/checkpoint0270.pth --checkpoint_key teacher --batch_size_per_gpu 8 --arch swin_tiny --cfg experiments/imagenet/swin/swin_tiny_patch4_window14_224.yaml --num_workers 1 MODEL.NUM_CLASSES 0
@REM python -m torch.distributed.launch --nproc_per_node=2 eval_linear_5.py --data_path E:/FineGrained/datasets/CUB_200_2011 --output_dir output_CUB1.0/eval_linear_more --pretrained_weights output_CUB1.0/checkpoint0270.pth --checkpoint_key teacher --batch_size_per_gpu 8 --arch swin_tiny --cfg experiments/imagenet/swin/swin_tiny_patch4_window14_224.yaml --n_last_blocks 4 --num_labels 200 --num_workers 1 MODEL.NUM_CLASSES 0
@REM python -m torch.distributed.launch --nproc_per_node=2 eval_knn_5.py --data_path E:/FineGrained/datasets/CUB_200_2011 --dump_features output_CUB1.0/eval_knn_more --pretrained_weights output_CUB1.0/checkpoint0270.pth --checkpoint_key teacher --batch_size_per_gpu 8 --arch swin_tiny --cfg experiments/imagenet/swin/swin_tiny_patch4_window14_224.yaml --num_workers 1 MODEL.NUM_CLASSES 0
@REM python -m torch.distributed.launch --nproc_per_node=2 eval_linear_5.py --data_path E:/FineGrained/datasets/StanfordCars --output_dir output_Cars/eval_linear_more --pretrained_weights output_Cars/checkpoint0270.pth --checkpoint_key teacher --batch_size_per_gpu 8 --arch swin_tiny --cfg experiments/imagenet/swin/swin_tiny_patch4_window14_224.yaml --n_last_blocks 4 --num_labels 196 --num_workers 1 MODEL.NUM_CLASSES 0
@REM python -m torch.distributed.launch --nproc_per_node=2 eval_knn_5.py --data_path E:/FineGrained/datasets/StanfordCars --dump_features output_Cars/eval_knn_more --pretrained_weights output_Cars/checkpoint0270.pth --checkpoint_key teacher --batch_size_per_gpu 8 --arch swin_tiny --cfg experiments/imagenet/swin/swin_tiny_patch4_window14_224.yaml --num_workers 1 MODEL.NUM_CLASSES 0
python -m torch.distributed.launch --nproc_per_node=2 eval_linear_5.py --data_path E:/FineGrained/datasets/NABirds --output_dir output_NABirds/eval_linear_more --pretrained_weights output_NABirds/checkpoint0090.pth --checkpoint_key teacher --batch_size_per_gpu 8 --arch swin_tiny --cfg experiments/imagenet/swin/swin_tiny_patch4_window14_224.yaml --n_last_blocks 4 --num_labels 555 --num_workers 1 MODEL.NUM_CLASSES 0
python -m torch.distributed.launch --nproc_per_node=2 eval_knn_5.py --data_path E:/FineGrained/datasets/NABirds --dump_features output_NABirds/eval_knn_more --pretrained_weights output_NABirds/checkpoint0090.pth --checkpoint_key teacher --batch_size_per_gpu 8 --arch swin_tiny --cfg experiments/imagenet/swin/swin_tiny_patch4_window14_224.yaml --num_workers 1 MODEL.NUM_CLASSES 0


@REM python -m torch.distributed.launch --nproc_per_node=2 eval_linear_ratio0.5.py --data_path E:/FineGrained/datasets/FGVC-Aircrafts --output_dir output_Aircrafts0.1/eval_linear0.5 --pretrained_weights output_Aircrafts0.1/checkpoint0270.pth --checkpoint_key teacher --batch_size_per_gpu 8 --arch swin_tiny --cfg experiments/imagenet/swin/swin_tiny_patch4_window14_224.yaml --n_last_blocks 4 --num_labels 100 --num_workers 1 MODEL.NUM_CLASSES 0
@REM python -m torch.distributed.launch --nproc_per_node=2 eval_knn_ratio0.5.py --data_path E:/FineGrained/datasets/FGVC-Aircrafts --dump_features output_Aircrafts0.1/eval_knn0.5 --pretrained_weights output_Aircrafts0.1/checkpoint0270.pth --checkpoint_key teacher --batch_size_per_gpu 8 --arch swin_tiny --cfg experiments/imagenet/swin/swin_tiny_patch4_window14_224.yaml --num_workers 1 MODEL.NUM_CLASSES 0
@REM 1python -m torch.distributed.launch --nproc_per_node=2 eval_linear_ratio0.5.py --data_path E:/FineGrained/datasets/CUB_200_2011 --output_dir output_CUB0.1/eval_linear0.5 --pretrained_weights output_CUB0.1/checkpoint0270.pth --checkpoint_key teacher --batch_size_per_gpu 8 --arch swin_tiny --cfg experiments/imagenet/swin/swin_tiny_patch4_window14_224.yaml --n_last_blocks 4 --num_labels 200 --num_workers 1 MODEL.NUM_CLASSES 0
@REM 1python -m torch.distributed.launch --nproc_per_node=2 eval_knn_ratio0.5.py --data_path E:/FineGrained/datasets/CUB_200_2011 --dump_features output_CUB0.1/eval_knn0.5 --pretrained_weights output_CUB0.1/checkpoint0270.pth --checkpoint_key teacher --batch_size_per_gpu 8 --arch swin_tiny --cfg experiments/imagenet/swin/swin_tiny_patch4_window14_224.yaml --num_workers 1 MODEL.NUM_CLASSES 0
@REM python -m torch.distributed.launch --nproc_per_node=2 eval_linear_ratio0.5.py --data_path E:/FineGrained/datasets/StanfordCars --output_dir output_Cars0.1/eval_linear0.5 --pretrained_weights output_Cars0.1/checkpoint0270.pth --checkpoint_key teacher --batch_size_per_gpu 8 --arch swin_tiny --cfg experiments/imagenet/swin/swin_tiny_patch4_window14_224.yaml --n_last_blocks 4 --num_labels 196 --num_workers 1 MODEL.NUM_CLASSES 0
@REM python -m torch.distributed.launch --nproc_per_node=2 eval_knn_ratio0.5.py --data_path E:/FineGrained/datasets/StanfordCars --dump_features output_Cars0.1/eval_knn0.5 --pretrained_weights output_Cars0.1/checkpoint0270.pth --checkpoint_key teacher --batch_size_per_gpu 8 --arch swin_tiny --cfg experiments/imagenet/swin/swin_tiny_patch4_window14_224.yaml --num_workers 1 MODEL.NUM_CLASSES 0
@REM python -m torch.distributed.launch --nproc_per_node=2 eval_linear_ratio0.2.py --data_path E:/FineGrained/datasets/FGVC-Aircrafts --output_dir output_Aircrafts0.1/eval_linear0.2 --pretrained_weights output_Aircrafts0.1/checkpoint0270.pth --checkpoint_key teacher --batch_size_per_gpu 8 --arch swin_tiny --cfg experiments/imagenet/swin/swin_tiny_patch4_window14_224.yaml --n_last_blocks 4 --num_labels 100 --num_workers 1 MODEL.NUM_CLASSES 0
@REM python -m torch.distributed.launch --nproc_per_node=2 eval_knn_ratio0.2.py --data_path E:/FineGrained/datasets/FGVC-Aircrafts --dump_features output_Aircrafts0.1/eval_knn0.2 --pretrained_weights output_Aircrafts0.1/checkpoint0270.pth --checkpoint_key teacher --batch_size_per_gpu 8 --arch swin_tiny --cfg experiments/imagenet/swin/swin_tiny_patch4_window14_224.yaml --num_workers 1 MODEL.NUM_CLASSES 0
@REM 1python -m torch.distributed.launch --nproc_per_node=2 eval_linear_ratio0.2.py --data_path E:/FineGrained/datasets/CUB_200_2011 --output_dir output_CUB0.1/eval_linear0.2 --pretrained_weights output_CUB0.1/checkpoint0270.pth --checkpoint_key teacher --batch_size_per_gpu 8 --arch swin_tiny --cfg experiments/imagenet/swin/swin_tiny_patch4_window14_224.yaml --n_last_blocks 4 --num_labels 200 --num_workers 1 MODEL.NUM_CLASSES 0
@REM 1python -m torch.distributed.launch --nproc_per_node=2 eval_knn_ratio0.2.py --data_path E:/FineGrained/datasets/CUB_200_2011 --dump_features output_CUB0.1/eval_knn0.2 --pretrained_weights output_CUB0.1/checkpoint0270.pth --checkpoint_key teacher --batch_size_per_gpu 8 --arch swin_tiny --cfg experiments/imagenet/swin/swin_tiny_patch4_window14_224.yaml --num_workers 1 MODEL.NUM_CLASSES 0
@REM python -m torch.distributed.launch --nproc_per_node=2 eval_linear_ratio0.2.py --data_path E:/FineGrained/datasets/StanfordCars --output_dir output_Cars0.1/eval_linear0.2 --pretrained_weights output_Cars0.1/checkpoint0270.pth --checkpoint_key teacher --batch_size_per_gpu 8 --arch swin_tiny --cfg experiments/imagenet/swin/swin_tiny_patch4_window14_224.yaml --n_last_blocks 4 --num_labels 196 --num_workers 1 MODEL.NUM_CLASSES 0
@REM python -m torch.distributed.launch --nproc_per_node=2 eval_knn_ratio0.2.py --data_path E:/FineGrained/datasets/StanfordCars --dump_features output_Cars0.1/eval_knn0.2 --pretrained_weights output_Cars0.1/checkpoint0270.pth --checkpoint_key teacher --batch_size_per_gpu 8 --arch swin_tiny --cfg experiments/imagenet/swin/swin_tiny_patch4_window14_224.yaml --num_workers 1 MODEL.NUM_CLASSES 0