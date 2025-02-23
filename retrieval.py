import os
import numpy as np
import cv2
from tqdm import tqdm
import torch
import argparse
import os
import random



if __name__ == '__main__':

    split_mode = 'ratio' # ratio / origin
    dataset = 'NABirds'

    # all_train_feature = np.array(torch.load('E:/selfsup/esvit_md/output_{0}0.1/eval_knn_more/trainfeat.pth'.format(dataset)))
    # all_test_feature = np.array(torch.load('E:/selfsup/esvit_md/output_{0}0.1/eval_knn_more/testfeat.pth'.format(dataset)))
    # all_train_labels = np.array(torch.load('E:/selfsup/esvit_md/output_{0}0.1/eval_knn_more/trainlabels.pth'.format(dataset)))
    # all_test_labels = np.array(torch.load('E:/selfsup/esvit_md/output_{0}0.1/eval_knn_more/testlabels.pth'.format(dataset)))
    all_train_feature = np.array(torch.load('E:/selfsup/esvit_md/output_{0}/eval_knn_more/trainfeat.pth'.format(dataset)))
    all_test_feature = np.array(torch.load('E:/selfsup/esvit_md/output_{0}/eval_knn_more/testfeat.pth'.format(dataset)))
    all_train_labels = np.array(torch.load('E:/selfsup/esvit_md/output_{0}/eval_knn_more/trainlabels.pth'.format(dataset)))
    all_test_labels = np.array(torch.load('E:/selfsup/esvit_md/output_{0}/eval_knn_more/testlabels.pth'.format(dataset)))

    # take train as gallary, test as query
    if split_mode == 'origin':
        gallary_feats = all_train_feature
        query_feats = all_test_feature
        gallary_labels = all_train_labels
        query_labels = all_test_labels

    # take samples in val according to ratio as query, the rest val + train as gallary
    if split_mode == 'ratio':
        # query_ratio = 0.1
        all_feats = np.concatenate((all_train_feature, all_test_feature), axis=0)
        all_labels = np.concatenate((all_train_labels, all_test_labels), axis=0)
        # random_index = np.arange(all_test_labels.shape[0]) + all_train_labels.shape[0]
        # random.shuffle(random_index)
        # query_index = random_index[:int(len(random_index)*query_ratio)]
        query_index = []
        for i in range(all_test_labels.shape[0]):
            if i % 9 == 0:
                query_index.append(all_train_labels.shape[0] + i) 
        gallary_feats = []
        gallary_labels = []
        query_feats = []
        query_labels = []
        for ind in range(all_labels.shape[0]):
            if ind in query_index:
                query_feats.append(all_feats[ind])
                query_labels.append(all_labels[ind])
            else:
                gallary_feats.append(all_feats[ind])
                gallary_labels.append(all_labels[ind])
        gallary_feats = np.array(gallary_feats)
        gallary_labels = np.array(gallary_labels)
        query_feats = np.array(query_feats)
        query_labels = np.array(query_labels)
        print('Query samples: {0}, Gallary samples: {1}'.format(query_labels.shape[0], gallary_labels.shape[0]))

    rank1 = 0
    rank5 = 0
    mAP = 0
    for i in tqdm(range(query_labels.shape[0])):
        query_feat = query_feats[i, :]
        scores = np.dot(query_feat, gallary_feats.transpose())
        rank_ID = np.argsort(scores)[::-1]
        rank_score = scores[rank_ID]
        top_k = 5
        r_label = [gallary_labels[index] for index in rank_ID[:top_k]]
        # print('Query-{0} | R1-{1} R2-{2} R3-{3} R4-{4} R5-{5}'.format(query_labels[i], r_label[0], 
        # r_label[1], r_label[2], r_label[3], r_label[4]))
        # calculate rank@1
        if r_label[0] == query_labels[i]:
            rank1 += 1
        # calculate rank@5
        for j in range(top_k):
            if r_label[j] == query_labels[i]:
                rank5 += 1
                break
        # calculate AP
        at_map = 100
        ap = 0
        pos_num = 0
        precision_sum = 0
        pos_set = []
        all_r_label = [gallary_labels[index] for index in rank_ID[:]]
        for j in range(len(all_r_label)):
            if all_r_label[j] == query_labels[i]:
                pos_set.append(j)
        for j in range(len(all_r_label)):
            if j in pos_set and j < at_map:
                pos_num += 1
                precision_sum += pos_num / (j+1)
        if pos_num != 0:
            ap = precision_sum / pos_num
        else:
            ap = 0
        mAP += ap
    rank1 /= query_labels.shape[0]
    rank5 /= query_labels.shape[0]
    mAP /= query_labels.shape[0]
    print('Rank-1: {0:.2f} Rank-5: {1:.2f} mAP: {2:.2f}'.format(rank1*100, rank5*100, mAP*100))
    