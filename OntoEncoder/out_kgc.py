import os
import json
import numpy as np
import torch
from torch.nn.init import xavier_normal_
from torch.nn import Parameter



def get_param(shape):
    param = Parameter(torch.Tensor(*shape));
    xavier_normal_(param.data)
    return param


###########################

def loadDict(file_name):
    entities = list()
    wnids = open(file_name, 'rU')
    try:
        for line in wnids:
            line = line[:-1]
            index, cls = line.split('\t')
            entities.append(cls)
    finally:
        wnids.close()
    print(len(entities))
    return entities



if __name__ == '__main__':

    datadir = '../../ZS_KGC/data'

    dataset, SemEmbed = 'NELL', 'TransE'

    DATASET_DIR = os.path.join(datadir, dataset)

    embed_path = os.path.join('../../data/', dataset)



    if dataset == 'NELL':
        rel2id = json.load(open(os.path.join(DATASET_DIR, 'relation2ids')))
        if SemEmbed == 'TransE':
            save_name, embed_file, save_file, w_dim = 'TransE_D200_NELL', 'entity_55000.npy', 'TransE_55000.npz', 200
        if SemEmbed == 'RGAT':
            save_name, embed_file, save_file, w_dim = 'RGAT_D200_NELL', '3400_3360_ent_embeddings.npy', 'RGAT_3400_3360.npz', 200
        if SemEmbed == 'DisenE':
            save_name, embed_file, save_file, w_dim = 'DisenE_K4_D200_NELL', '5400_4950_ent_embeddings.npy', 'DisenE_5400_4950.npz', 800
        if SemEmbed == 'DisenKGAT':
            save_name, embed_file, save_file, w_dim = 'DisenKAGT_TransE_mult_K2_D200_NELL', '5600_5525_ent_embeddings.npy', 'DisenKGAT_5600_5525.npz', 400
        if SemEmbed == 'DOZSL_RD':
            save_name, embed_file, save_file, w_dim = 'DOZSL_Random_K4_D200_NELL', '6000_5985_ent_embeddings.npy', 'DOZSL_RD_6000_5985.npz', 800
        if SemEmbed == 'DOZSL_AGG':
            save_name, embed_file, save_file, w_dim = 'DOZSL_AGG_K9_D200_NELL', '2000_1911_ent_embeddings.npy', 'DOZSL_AGG_2000_1911.npz', 1800
        if SemEmbed == 'DOZSL_AGG_sub':
            save_name, embed_file, save_file, w_dim = 'DOZSL_AGG_sub_K8_D200_NELL', '3000_2975_ent_embeddings.npy', 'DOZSL_AGG_sub_3000_2975.npz', 1600




    if dataset == 'Wiki':
        rel2id = json.load(open(os.path.join(DATASET_DIR, 'relation2ids_1')))
        if SemEmbed == 'TransE':
            save_name, embed_file, save_file, w_dim = 'TransE_D200_NELL', 'entity_65000.npy', 'TransE_65000.npz', 200
        if SemEmbed == 'RGAT':
            save_name, embed_file, save_file, w_dim = 'RGAT_D200_NELL', '6400_6357_ent_embeddings.npy', 'RGAT_6400_6357.npz', 200
        if SemEmbed == 'DisenE':
            save_name, embed_file, save_file, w_dim = 'DisenE_K2_D200_NELL', '10600_10152_ent_embeddings.npy', 'DisenE_10600_10152.npz', 400
        if SemEmbed == 'DisenKGAT':
            save_name, embed_file, save_file, w_dim = 'DisenKAGT_TransE_mult_K4_D200_NELL', '4200_4084_ent_embeddings.npy', 'DisenKGAT_4200_4084.npz', 800
        if SemEmbed == 'DOZSL_RD':
            save_name, embed_file, save_file, w_dim = 'DOZSL_Random_K4_D200_NELL', '14400_14058_ent_embeddings.npy', 'DOZSL_RD_14400_14058.npz', 800
        if SemEmbed == 'DOZSL_AGG':
            save_name, embed_file, save_file, w_dim = 'DOZSL_AGG_K9_D200_NELL', '2600_2598_ent_embeddings.npy', 'DOZSL_AGG_2600_2598.npz', 1800
        if SemEmbed == 'DOZSL_AGG_sub':
            save_name, embed_file, save_file, w_dim = 'DOZSL_AGG_sub_K8_D200_NELL', '2400_2366_ent_embeddings.npy', 'DOZSL_AGG_sub_2400_2366.npz', 1600




    embed_file = os.path.join(embed_path, save_name, embed_file)
    # load entity dict
    entity_file = os.path.join(embed_path, 'ent2id.txt')
    ent2id = json.load(open(entity_file))
    id2rel = {v: k for k, v in rel2id.items()}
    id2rel = {k: id2rel[k] for k in sorted(id2rel.keys())}



    all_feats = np.zeros((len(rel2id.keys()), w_dim), dtype=np.float)
    ent_embeds = np.load(embed_file)

    print(ent_embeds.shape)
    for i, rel in id2rel.items():
        if dataset == 'NELL':
            rel = rel.replace('concept:', 'NELL:')
        if dataset == 'Wiki':
            rel = 'Wikidata:' + rel

        rel = rel.lower()
        # load embeddings
        if rel in ent2id:
            rel_embed = ent_embeds[ent2id[rel]].astype('float32')
            all_feats[i] = rel_embed.reshape(-1, w_dim)
        else:
            print('not found:', rel)


    all_feats = np.array(all_feats)  # 229, 51

    np.savez(os.path.join(DATASET_DIR, 'embeddings', save_file), relaM=all_feats)









