# DOZSL
Code and Data for the paper: "Disentangled Ontology Embedding for Zero-shot Learning".
In this work, we focus on ontologies for augmenting ZSL, and propose to learn disentangled ontology embeddings to capture and utilize more fine-grained class relationships in different aspects.
We also contribute a new ZSL framework named DOZSL, which contains two new ZSL solutions based on generative models and graph propagation models respectively,
for effectively utilizing the disentangled ontology embeddings for zero-shot image classification (ZS-IMGC) and zero-shot KG completion (ZS-KGC) with unseen relations.

### Requirements
- `python 3.5`
- `PyTorch >= 1.5.0`

### Dataset Preparation

#### AwA2
Download public pre-trained image features and dataset split for [AwA2](http://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip), uncompress it and put the files in **AWA2** folder to our folder `ZS_IMGC/data/AwA2/`.


#### ImageNet (ImNet-A, ImNet-O)

Download pre-trained image features of ImageNet classes and their class splits from [here](https://drive.google.com/drive/folders/1An6nLXRRvlKSCbJoKKlqTNDvgN7PyvvW?usp=sharing) and put them to the folder `ZS_IMGC/data/ImageNet/`.


#### NELL-ZS & Wiki-ZS
Download from [here](https://github.com/Panda0406/Zero-shot-knowledge-graph-relational-learning) and put them to the corresponding data folder.

### Basic Training and Testing

#### Disentangled Ontology Encoder

The first thing you need to do is to train the disentangled ontology encoder, using the codes in the folders `OntoEncoder/DOZSL_RD` (for **RD** variants) and `OntoEncoder/DOZSL_AGG` (for **AGG** variants).

**Steps:**
1. Running `run.py` in each method folder to obtain the disentangled concept embeddings;
2. Selecting target **class** or **relation** embeddings from the trained concept embeddings by running `python out_imgc.py` for ZS-IMGC task and `python out_kgc.py` for ZS-KGC task.


#### Entangled ZSL Learner
With the selected class embedding or relation embedding, you can take it to perform downstream ZSL tasks using the generative model or graph propagation model.

The codes for generative model are in folder `ZS_IMGC/models/DOZSL_GAN` and `ZS_KGC/models/DOZSL_GAN` for ZS-IMGC and ZS-KGC tasks, respectively,
for propagation model are in folder `ZS_IMGC/models/DOZSL_GCN` and `ZS_KGC/models/DOZSL_GCN`.

*Note: you can skip the step of training ontology encoder if you just want to use the ontology embedding we learned, the embedding files have already been attached in the corresponding directories*.

#### Baselines
- The baselines for different ZSL methods are in the folders `ZS_IMGC/models` and `ZS_KGC/models` for ZS-IMGC and ZS-KGC tasks, respectively.
- The baselines for different ontology embedding methods are in the folder `OntoEncoder`.


### How to Cite
If you find this code useful, please consider citing the following paper.
```bigquery
@inproceedings{geng2022dozsl,
  author    = {Yuxia Geng and
               Jiaoyan Chen and
               Wen Zhang and
               Yajing Xu and
               Zhuo Chen and
               Jeff Z. Pan and
               YUfeng Huang and
               Feiyu Xiong and
               Huajun Chen},
  title     = {Disentangled Ontology Embedding for Zero-shot Learning},
  booktitle = {{KDD} '22: 28th ACM SIGKDD
Conference on Knowledge Discovery and Data Mining / Washington DC, USA, August 14-18, 2022},
  publisher = {{ACM} / {IW3C2}},
  year      = {2022}
}
```