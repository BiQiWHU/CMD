# Modified by Chunyuan Li (chunyl@microsoft.com)
#
# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse
import json
from pathlib import Path
from sklearn import metrics
from sklearn.preprocessing import label_binarize
import numpy as np

import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms as pth_transforms

from torchvision import models as torchvision_models
torchvision_archs = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

import utils
import models.vision_transformer as vits
from models.vision_transformer import DINOHead
from models import build_model

from config import config
from config import update_config
from config import save_config

def eval_linear(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ preparing data ... ============
    train_transform = pth_transforms.Compose([
        pth_transforms.RandomResizedCrop(224),
        pth_transforms.RandomHorizontalFlip(),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    val_transform = pth_transforms.Compose([
        pth_transforms.Resize(256, interpolation=3),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    if args.zip_mode:

        from .zipdata import ZipData

        datapath_train = os.path.join(config.DATA.DATA_PATH, 'train.zip')
        data_map_train = os.path.join(config.DATA.DATA_PATH, 'train_map.txt')

        datapath_val = os.path.join(config.DATA.DATA_PATH, 'val.zip')
        data_map_val = os.path.join(config.DATA.DATA_PATH, 'val_map.txt')
        
        dataset_train = ZipData(datapath_train, data_map_train, train_transform)
        dataset_val = ZipData(datapath_val, data_map_val, val_transform)
        
    else:
        dataset_train = datasets.ImageFolder(os.path.join(args.data_path, "train"), transform=train_transform)
        dataset_val = datasets.ImageFolder(os.path.join(args.data_path, "val"), transform=val_transform)

    sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")

    # ============ building network ... ============
    # if the network is a 4-stage vision transformer (i.e. swin)
    if 'swin' in args.arch :
        update_config(config, args)
        model = build_model(config, is_teacher=True)

        swin_spec = config.MODEL.SPEC
        embed_dim=swin_spec['DIM_EMBED']
        depths=swin_spec['DEPTHS']
        num_heads=swin_spec['NUM_HEADS'] 

        num_features = []
        for i, d in enumerate(depths):
            num_features += [int(embed_dim * 2 ** i)] * d 
        
        print(num_features)
        num_features_linear = sum(num_features[-args.n_last_blocks:])

        print(f'num_features_linear {num_features_linear}')

        linear_classifier = LinearClassifier(num_features_linear, args.num_labels)

    # if the network is a 4-stage vision transformer (i.e. longformer)
    elif 'vil' in args.arch :
        update_config(config, args)
        model = build_model(config, is_teacher=True)

        msvit_spec = config.MODEL.SPEC
        arch = msvit_spec.MSVIT.ARCH

        layer_cfgs = model.layer_cfgs
        num_stages = len(model.layer_cfgs)
        depths = [cfg['n'] for cfg in model.layer_cfgs]
        dims = [cfg['d'] for cfg in model.layer_cfgs]
        out_planes = model.layer_cfgs[-1]['d']
        Nglos = [cfg['g'] for cfg in model.layer_cfgs]

        print(dims)

        num_features = []
        for i, d in enumerate(depths):
            num_features += [ dims[i] ] * d 
        
        print(num_features)
        num_features_linear = sum(num_features[-args.n_last_blocks:])

        print(f'num_features_linear {num_features_linear}')

        linear_classifier = LinearClassifier(num_features_linear, args.num_labels)


    # if the network is a 4-stage vision transformer (i.e. CvT)
    elif 'cvt' in args.arch :
        update_config(config, args)
        model = build_model(config, is_teacher=True)

        cvt_spec = config.MODEL.SPEC
        embed_dim=cvt_spec['DIM_EMBED']
        depths=cvt_spec['DEPTH']
        num_heads=cvt_spec['NUM_HEADS'] 


        print(f'embed_dim {embed_dim} depths {depths}')
        num_features = []
        for i, d in enumerate(depths):
            num_features += [int(embed_dim[i])] * int(d) 
        
        print(num_features)
        num_features_linear = sum(num_features[-args.n_last_blocks:])

        print(f'num_features_linear {num_features_linear}')

        linear_classifier = LinearClassifier(num_features_linear, args.num_labels)



    # if the network is a vanilla vision transformer (i.e. deit_tiny, deit_small, vit_base)
    elif args.arch in vits.__dict__.keys():
        depths = []
        model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
        linear_classifier = LinearClassifier(model.embed_dim * (args.n_last_blocks + int(args.avgpool_patchtokens)), args.num_labels)

    model.cuda()
    model.eval()
    print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    # load weights to evaluate
    utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)

    
    linear_classifier = linear_classifier.cuda()
    linear_classifier = nn.parallel.DistributedDataParallel(linear_classifier, device_ids=[args.gpu])

    # set optimizer
    optimizer = torch.optim.SGD(
        linear_classifier.parameters(),
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256., # linear scaling rule
        momentum=0.9,
        weight_decay=0, # we do not apply weight decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)

    if args.output_dir and dist.get_rank() == 0:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)  
              

    # Optionally resume from a checkpoint
    to_restore = {"epoch": 0, "best_acc": 0.}
    utils.restart_from_checkpoint(
        'E:/selfsup/esvit-my_v3/output_{}/eval_linear_more/checkpoint.pth.tar'.format(args.data_path.split('/')[-1]),
        run_variables=to_restore,
        state_dict=linear_classifier,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    start_epoch = 0
    best_acc = to_restore["best_acc"]

    for epoch in range(start_epoch, args.epochs):
        validate_network(val_loader, model, linear_classifier, args.n_last_blocks, args.avgpool_patchtokens, depths)
        exit()

        train_loader.sampler.set_epoch(epoch)

        train_stats = train(model, linear_classifier, optimizer, train_loader, epoch, args.n_last_blocks, args.avgpool_patchtokens, depths)
        scheduler.step()

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            test_stats = validate_network(val_loader, model, linear_classifier, args.n_last_blocks, args.avgpool_patchtokens, depths)
            print(f"Accuracy at epoch {epoch} of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
            best_acc = max(best_acc, test_stats["acc1"])
            print(f'Max accuracy so far: {best_acc:.2f}%')
            log_stats = {**{k: v for k, v in log_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()}}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            save_dict = {
                "epoch": epoch + 1,
                "state_dict": linear_classifier.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_acc": best_acc,
            }
            torch.save(save_dict, os.path.join(args.output_dir, "checkpoint.pth.tar"))
    print("Training of the supervised linear classifier on frozen features completed.\n"
                "Top-1 test accuracy: {acc:.1f}".format(acc=best_acc))


def train(model, linear_classifier, optimizer, loader, epoch, n, avgpool, depths):
    linear_classifier.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    for (inp, target) in metric_logger.log_every(loader, 20, header):
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # forward
        with torch.no_grad():
            output = model.forward_return_n_last_blocks(inp, n, avgpool, depths)
        
        # print(f'output {output.shape}')
        output = linear_classifier(output)

        # compute cross entropy loss
        loss = nn.CrossEntropyLoss()(output, target)

        # compute the gradients
        optimizer.zero_grad()
        loss.backward()

        # step
        optimizer.step()

        # log 
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate_network(val_loader, model, linear_classifier, n, avgpool, depths):
    linear_classifier.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    ground_truths_multiclass = []
    ground_truths_onehot = []
    predictions_class = []

    sample_num = 0
    for inp, target in metric_logger.log_every(val_loader, 20, header):
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        batch_size = inp.shape[0]

        # compute output
        feature, feature_region = model.forward_feature_maps(inp) # [8, 768], [8, 49, 768]
        # output = model.forward_return_n_last_blocks(inp, n, avgpool, depths)
        # output = linear_classifier(output) # [8, 200]
        # output = feature
        B, N, C = feature_region.shape
        output = feature_region.sum(axis=1)
        # output = feature_region.reshape(B, N, int(C/3), 3).sum(axis=-1)
        if sample_num == 0:
            output_fea = output.detach().cpu().numpy()
        else:
            output_fea = np.concatenate((output_fea, output.detach().cpu().numpy()), axis=0)
        sample_num += batch_size
        # if sample_num == 160:
        #     np.save('E:/selfsup/esvit_md/vis/feat/{}_fea.npy'.format(args.data_path.split('/')[-1]), output_fea)
        #     exit()
        np.save('E:/selfsup/esvit_md/vis/feat/{}_fea.npy'.format(args.data_path.split('/')[-1]), output_fea)
    #     loss = nn.CrossEntropyLoss()(output, target)

    #     acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
    #     acc1, acc5 = utils.accuracy(output, target, topk=(1, 2))

    #     batch_size = inp.shape[0]
    #     metric_logger.update(loss=loss.item())
    #     metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
    #     metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    #     ##### my
    #     # outputs_class = utils.softmax(output.data.cpu().numpy())
    #     # ground_truths.extend(target.data.cpu().numpy())
    #     # predictions_class.extend(outputs_class)

    #     labels_multiclass = target.clone()
    #     labels_onehot = target.clone()
    #     labels_onehot = torch.nn.functional.one_hot(labels_onehot, output.shape[-1])

    #     outputs_class = utils.softmax(output.data.cpu().numpy())
    #     ground_truths_multiclass.extend(labels_multiclass.data.cpu().numpy())
    #     ground_truths_onehot.extend(labels_onehot.data.cpu().numpy())
    #     predictions_class.extend(outputs_class)

    # gts = np.asarray(ground_truths_multiclass)
    # probs = np.asarray(predictions_class)
    # preds = np.argmax(probs, axis=1)
    # accuracy = metrics.accuracy_score(gts, preds)

    # gts2 = np.asarray(ground_truths_onehot)
    # trues = np.asarray(gts2).flatten()
    # probs2 = np.asarray(probs).flatten()
    # AUC = metrics.roc_auc_score(trues, probs2)

    # wKappa = metrics.cohen_kappa_score(gts, preds, weights='quadratic')
    # wF1 = metrics.f1_score(gts, preds, average='weighted')

    # # metric_logger.meters['recall'] = recall
    # # metric_logger.meters['precision'] = precision
    # ##### my

    # print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
    #       .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    # print('--- Acc {} AUC {} wF1 {} wKappa {} ---'.format(accuracy, AUC, wF1, wKappa))
    # # return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    # result = {}
    # result['acc1'] = metric_logger.acc1.global_avg
    # result['acc5'] = metric_logger.acc1.global_avg
    # result['Acc'] = accuracy
    # result['AUC'] = AUC
    # result['wF1'] = wF1
    # result['wKappa'] = wKappa
    # return result


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=1000):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        return self.linear(x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with linear classification on ImageNet')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        type=str)

    parser.add_argument('--arch', default='deit_small', type=str,
        choices=['cvt_tiny', 'swin_tiny','swin_small', 'swin_base', 'swin_large', 'swin', 'vil', 'vil_1281', 'vil_2262', 'deit_tiny', 'deit_small', 'vit_base'] + torchvision_archs,
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using deit_tiny or deit_small.""")

    parser.add_argument('--n_last_blocks', default=4, type=int, help="""Concatenate [CLS] tokens
        for the `n` last blocks. We use `n=4` when evaluating DeiT-Small and `n=1` with ViT-Base.""")
    parser.add_argument('--avgpool_patchtokens', default=False, type=utils.bool_flag,
        help="""Whether ot not to concatenate the global average pooled features to the [CLS] token.
        We typically set this to False for DeiT-Small and to True with ViT-Base.""")

    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument("--lr", default=0.001, type=float, help="""Learning rate at the beginning of
        training (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.
        We recommend tweaking the LR depending on the checkpoint evaluated.""")
    parser.add_argument('--batch_size_per_gpu', default=128, type=int, help='Per-GPU batch-size')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--data_path', default='/path/to/imagenet/', type=str)
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--val_freq', default=1, type=int, help="Epoch frequency for validation.")
    parser.add_argument('--output_dir', default=".", help='Path to save logs and checkpoints')

    # Dataset
    parser.add_argument('--zip_mode', type=utils.bool_flag, default=False, help="""Whether or not
        to use zip file.""")

    parser.add_argument('--num_labels', default=1000, type=int, help='number of classes in a dataset')

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)    

    args = parser.parse_args()


    eval_linear(args)
