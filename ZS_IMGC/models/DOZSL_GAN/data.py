import numpy as np
import scipy.io as scio
import torch
from sklearn import preprocessing
import os
import time



def readTxt(file_name):
    class_list = list()
    wnids = open(file_name, 'rU')
    try:
        for line in wnids:
            class_list.append(line[:-1])
    finally:
        wnids.close()
    return class_list

#
#
# imagenet_split_file = '/Users/geng/Desktop/DATA/ImageNet/split.mat'
# # imagenet_split_file = '/Users/geng/Desktop/DATA/ImageNet/w2v.mat'
# # # imagenet_split_file = '/Users/geng/Downloads/imagenet_class_splits.mat'
# # matcontent = scio.loadmat(imagenet_split_file)
# # print(matcontent.keys())
#
# matcontent = scio.loadmat(imagenet_split_file)
# wnids = matcontent['allwnids'].squeeze().tolist()
# words = matcontent['allwords'].squeeze()
#
# seen_file = os.path.join('/Users/geng/Desktop/ZSL_DATA/ImageNet', 'ImNet_A', 'seen.txt')
# unseen_file = os.path.join('/Users/geng/Desktop/ZSL_DATA/ImageNet', 'ImNet_A', 'unseen.txt')
#
# seen_wnids = readTxt(seen_file)
# unseen_wnids = readTxt(unseen_file)
#
# for wnid in seen_wnids:
#     if wnid in wnids[:1000]:
#         continue
#     else:
#         print(wnid, wnids.index(wnid)+1, words[wnids.index(wnid)])
#
# print('******* unseen *****')
# for wnid in unseen_wnids:
#     if wnid in wnids[1000:]:
#         continue
#     else:
#         print(wnid, wnids.index(wnid)+1, words[wnids.index(wnid)])
# #


o2v_file = '/Users/geng/PycharmProjects/ZSL-FSL/OntoZSL/data/ImageNet/ImNet_O/onto_file/embeddings/o2v-imagenet-o.mat'
matcontent = scio.loadmat(o2v_file)
print(matcontent.keys())