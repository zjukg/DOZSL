import os
import json
import sys
import pickle as pkl
import numpy as np
import scipy.io as scio





def readTxt(file_name):
    class_list = list()
    wnids = open(file_name, 'rU')
    try:
        for line in wnids:
            line = line[:-1]
            class_list.append(line)
    finally:
        wnids.close()
    print(len(class_list))
    return class_list

def load_class():
    seen = readTxt(seen_file)
    unseen = readTxt(unseen_file)
    return seen, unseen


def save_embed_awa(filename, wnids, names):


    # load embeddings
    embeds = np.load(filename)
    # save to .mat file
    matcontent = scio.loadmat(os.path.join(DATASET_DIR, 'att_splits.mat'))
    all_names = matcontent['allclasses_names'].squeeze().tolist()

    # embed_size = embeds.shape[1]
    vectors = np.zeros((len(all_names), embed_size), dtype=np.float)
    for i in range(len(all_names)):
        name = all_names[i][0]
        wnid = wnids[names.index(name)]
        wnid = namespace + wnid
        wnid = wnid.lower()
        if wnid in ent2id:

            vectors[i] = embeds[ent2id[wnid]].reshape(-1, embed_size)
        else:
            print('not found:', wnid)

    print(vectors.shape)

    embed_file = os.path.join(DATASET_DIR, 'embeddings', save_file)
    scio.savemat(embed_file, {'embeddings': vectors})

def save_embed(filename, classes):

    # load embeddings
    embeds = np.load(filename)
    # save to .mat file
    matcontent = scio.loadmat(os.path.join(datadir, 'ImageNet', 'w2v.mat'))
    wnids = matcontent['wnids'].squeeze().tolist()
    wnids = wnids[:2549]

    vectors = np.zeros((len(wnids), embed_size), dtype=np.float)

    print(vectors.shape)
    for i, wnid in enumerate(wnids):
        wnid = wnid[0]
        if wnid in classes:

            wnid = namespace + wnid
            wnid = wnid.lower()
            if wnid in ent2id:
                vectors[i] = embeds[ent2id[wnid]].reshape(-1, embed_size)
            else:
                print('not found:', wnid)
        else:
            continue
    wnids_cell = np.empty((len(wnids), 1), dtype=np.object)
    for i in range(len(wnids)):
        wnids_cell[i][0] = np.array(wnids[i])

    embed_file = os.path.join(DATASET_DIR, 'embeddings', save_file)
    scio.savemat(embed_file, {'embeddings': vectors, 'wnids': wnids_cell})




if __name__ == '__main__':

    datadir = '../../ZS_IMGC/data'

    # providing the target dataset and target embedding methods as following
    dataset, SemEmbed = 'AwA2', 'TransE'
    # dataset, SemEmbed = 'ImNet_A', 'RGAT'
    # dataset, SemEmbed = 'ImNet_O', 'DisenE'




    if dataset == 'AwA2':
        DATASET_DIR = os.path.join(datadir, dataset)
        embed_path = os.path.join('../../data/', 'AwA')
        namespace = 'AwA:'

        if SemEmbed == 'TransE':
            save_name, embed_file, save_file, embed_size = 'TransE_D100_AwA', 'entity_65000.npy', 'TransE_65000.mat', 100
        if SemEmbed == 'RGAT':
            save_name, embed_file, save_file, embed_size = 'RGAT_D100_AwA', '9200_9191_ent_embeddings.npy', 'RGAT_9200_9191.mat', 100
        if SemEmbed == 'DisenE':
            save_name, embed_file, save_file, embed_size = 'DisenE_K2_D100_AwA', '5600_5583_ent_embeddings.npy', 'DisenE_5600_5583.mat', 200
        if SemEmbed == 'DisenKGAT':
            save_name, embed_file, save_file, embed_size = 'DisenKAGT_TransE_mult_K4_D100_AwA', '9800_9667_ent_embeddings.npy', 'DisenKGAT_9800_9667.mat', 400
        if SemEmbed == 'DOZSL_RD':
            save_name, embed_file, save_file, embed_size = 'DOZSL_Random_K2_D100_AwA', '4800_4666_ent_embeddings.npy', 'DOZSL_RD_4800_4666.mat', 200
        if SemEmbed == 'DOZSL_AGG':
            save_name, embed_file, save_file, embed_size = 'DOZSL_AGG_K5_D100_AwA', '5200_5125_ent_embeddings.npy', 'DOZSL_AGG_5200_5125.mat', 500
        if SemEmbed == 'DOZSL_AGG_sub':
            save_name, embed_file, save_file, embed_size = 'DOZSL_AGG_sub_K4_D100_AwA', '5400_5291_ent_embeddings.npy', 'DOZSL_AGG_sub_5400_5291.mat', 400

    else:
        DATASET_DIR = os.path.join(datadir, 'ImageNet', dataset)
        embed_path = os.path.join('../../data/', dataset)
        if dataset == 'ImNet_A':
            namespace = 'ImNet-A:'
            if SemEmbed == 'TransE':
                save_name, embed_file, save_file, embed_size = 'TransE_D100_ImNet_A', 'entity_65000.npy', 'TransE_65000.mat', 100
            if SemEmbed == 'RGAT':
                save_name, embed_file, save_file, embed_size = 'RGAT_D100_ImNet_A', '6200_6164_ent_embeddings.npy', 'RGAT_6200_6164.mat', 100
            if SemEmbed == 'DisenE':
                save_name, embed_file, save_file, embed_size = 'DisenE_K2_D100_ImNet_A', '5000_4553_ent_embeddings.npy', 'DisenE_5000_4553.mat', 200
            if SemEmbed == 'DisenKGAT':
                save_name, embed_file, save_file, embed_size = 'DisenKAGT_TransE_mult_K2_D100_ImNet_A', '2400_2356_ent_embeddings.npy', 'DisenKGAT_2400_2356.mat', 200
            if SemEmbed == 'DOZSL_RD':
                save_name, embed_file, save_file, embed_size = 'DOZSL_Random_K2_D100_ImNet_A', '6000_5550_ent_embeddings.npy', 'DOZSL_RD_6000_5550.mat', 200
            if SemEmbed == 'DOZSL_AGG':
                save_name, embed_file, save_file, embed_size = 'DOZSL_AGG_K5_D100_ImNet_A', '2200_2191_ent_embeddings.npy', 'DOZSL_AGG_2200_2191.mat', 500
            if SemEmbed == 'DOZSL_AGG_sub':
                save_name, embed_file, save_file, embed_size = 'DOZSL_AGG_sub_K4_D100_ImNet_A', '2000_1894_ent_embeddings.npy', 'DOZSL_AGG_sub_2000_1894.mat', 400

        if dataset == 'ImNet_O':
            namespace = 'ImNet-O:'
            if SemEmbed == 'TransE':
                save_name, embed_file, save_file, embed_size = 'TransE_D100_ImNet_O', 'entity_55000.npy', 'TransE_55000.mat', 100
            if SemEmbed == 'RGAT':
                save_name, embed_file, save_file, embed_size = 'RGAT_D100_ImNet_O', '3000_2869_ent_embeddings.npy', 'RGAT_3000_2869.mat', 100
            if SemEmbed == 'DisenE':
                save_name, embed_file, save_file, embed_size = 'DisenE_K2_D100_ImNet_O', '2800_2342_ent_embeddings.npy', 'DisenE_2800_2342.mat', 200
            if SemEmbed == 'DisenKGAT':
                save_name, embed_file, save_file, embed_size = 'DisenKAGT_TransE_mult_K2_D100_ImNet_O', '3000_2980_ent_embeddings.npy', 'DisenKGAT_3000_2980.mat', 200
            if SemEmbed == 'DOZSL_RD':
                save_name, embed_file, save_file, embed_size = 'DOZSL_Random_K2_D100_ImNet_O', '3600_3598_ent_embeddings.npy', 'DOZSL_RD_3600_3598.mat', 200
            if SemEmbed == 'DOZSL_AGG':
                save_name, embed_file, save_file, embed_size = 'DOZSL_AGG_K5_D100_ImNet_O', '2000_1753_ent_embeddings.npy', 'DOZSL_AGG_2000_1753.mat', 500
            if SemEmbed == 'DOZSL_AGG_sub':
                save_name, embed_file, save_file, embed_size = 'DOZSL_AGG_sub_K4_D100_ImNet_O', '4200_4037_ent_embeddings.npy', 'DOZSL_AGG_sub_4200_4037.mat', 400




    # load entity dict
    entity_file = os.path.join(embed_path, 'ent2id.txt')
    ent2id = json.load(open(entity_file))

    embed_file = os.path.join(embed_path, save_name, embed_file)


    if dataset == 'AwA2':
        class_file = os.path.join(DATASET_DIR, 'class.json')
        classes = json.load(open(class_file, 'r'))
        wnids = list()
        names = list()
        for wnid, name in classes['seen'].items():
            wnids.append(wnid)
            names.append(name)
        for wnid, name in classes['unseen'].items():
            wnids.append(wnid)
            names.append(name)

        save_embed_awa(embed_file, wnids, names)

    else:
        seen_file = os.path.join(DATASET_DIR, 'seen.txt')
        unseen_file = os.path.join(DATASET_DIR, 'unseen.txt')
        seen, unseen = load_class()
        classes = seen + unseen

        save_embed(embed_file, classes)












