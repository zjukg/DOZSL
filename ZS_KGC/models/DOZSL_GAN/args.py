import argparse
import random
import os

'''
command:


python gan_kgc.py --dataset NELL --embed_model TransE --fc1_dim 250


evaluate start at epoch:
NELL: 1500
Wiki: 1000

'''
def read_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", default="../../data", type=str)

    parser.add_argument("--dataset", default="NELL", type=str)
    parser.add_argument("--embed_model", default='TransE', type=str)


    # embedding dimension
    parser.add_argument("--embed_dim", default=100, type=int, help='dimension of triple embedding')
    parser.add_argument("--ep_dim", default=200, type=int, help='dimension of entity pair embedding')
    parser.add_argument("--fc1_dim", default=250, type=int, help='dimension of entity pair embedding')
    parser.add_argument("--noise_dim", default=15, type=int)

    # feature extractor pretraining related
    parser.add_argument("--pretrain_batch_size", default=64, type=int)
    parser.add_argument("--pretrain_few", default=30, type=int)
    parser.add_argument("--pretrain_subepoch", default=20, type=int)
    parser.add_argument("--pretrain_margin", default=10.0, type=float, help='pretraining margin loss')
    parser.add_argument("--pretrain_times", default=16000, type=int, help='total training steps for pretraining')
    parser.add_argument("--pretrain_loss_every", default=500, type=int)

    # adversarial training related
    # batch size
    parser.add_argument("--D_batch_size", default=256, type=int)
    parser.add_argument("--G_batch_size", default=256, type=int)
    parser.add_argument("--gan_batch_rela", default=2, type=int)
    # learning rate
    parser.add_argument("--lr_G", default=0.0001, type=float)
    parser.add_argument("--lr_D", default=0.0001, type=float)
    parser.add_argument("--lr_E", default=0.0005, type=float)
    # training times
    parser.add_argument("--train_times", default=8000, type=int)
    parser.add_argument("--D_epoch", default=5, type=int)
    parser.add_argument("--G_epoch", default=1, type=int)
    # log
    parser.add_argument("--loss_every", default=50, type=int)
    parser.add_argument("--eval_every", default=500, type=int)
    parser.add_argument("--start_eval", default=2000, type=int)
    # hyper-parameter
    parser.add_argument("--test_sample", default=20, type=int, help='number of synthesized samples')
    parser.add_argument("--dropout", default=0.5, type=float)
    parser.add_argument('--REG_W', default=0.001, type=float)
    parser.add_argument('--REG_Wz', default=0.0001, type=float)
    parser.add_argument("--max_neighbor", default=50, type=int, help='neighbor number of each entity')
    parser.add_argument("--grad_clip", default=5.0, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)

    parser.add_argument("--fine_tune", action='store_true')
    parser.add_argument("--aggregate", default='max', type=str)
    parser.add_argument("--no_meta", action='store_true')

    # switch
    parser.add_argument("--pretrain_feature_extractor", action='store_true')
    parser.add_argument("--load_trained_embed", action='store_true', help='load well trained kg embeddings, such as DistMult')
    parser.add_argument("--trained_embed_path", default='')

    parser.add_argument("--SemEmbed", default='TransE', help='{TransE, DisenE, DisenKGAT, DOZSL_RD, DOZSL_AGG, DOZSL_AGG_sub}')

    parser.add_argument("--semantic_of_rel", default='')
    parser.add_argument("--input_dim", default=300, type=int)

    parser.add_argument("--train_data", default='')

    parser.add_argument("--seed", type=int, default=6096)
    parser.add_argument('--device', type=int, default=2, help='device to use for iterate data, -1 means cpu [default: 0]')

    args = parser.parse_args()

    args.save_path = os.path.join(args.datadir, args.dataset, 'expri_data', 'models_train')
    args.trained_embed_path = os.path.join('expri_data', 'Embed_used')
    args.train_data = os.path.join('datasplit', 'train_tasks.json')



    if args.seed is None:
        args.seed = random.randint(1, 10000)



    print("------HYPERPARAMETERS-------")
    print('relation embedding: ' + str(args.semantic_of_rel))
    print('input rel embedding dimension: ' + str(args.input_dim))
    print('random seed: ' + str(args.seed))

    print("----------------------------")

    if args.dataset == 'NELL':
        if args.SemEmbed == 'TransE':
            args.semantic_of_rel = os.path.join('embeddings', 'TransE_55000.npz')
            args.input_dim = 200
        if args.SemEmbed == 'RGAT':
            args.semantic_of_rel = os.path.join('embeddings', 'RGAT_3400_3360.npz')
            args.input_dim = 200
        if args.SemEmbed == 'DisenE':
            args.semantic_of_rel = os.path.join('embeddings', 'DisenE_5400_4950.npz')
            args.input_dim = 800
        if args.SemEmbed == 'DisenKGAT':
            args.semantic_of_rel = os.path.join('embeddings', 'DisenKGAT_5600_5525.npz')
            args.input_dim = 400
        if args.SemEmbed == 'DOZSL_RD':
            args.semantic_of_rel = os.path.join('embeddings', 'DOZSL_RD_6000_5985.npz')
            args.input_dim = 800
        if args.SemEmbed == 'DOZSL_AGG':
            args.semantic_of_rel = os.path.join('embeddings', 'DOZSL_AGG_2000_1911.npz')
            args.input_dim = 1800
        if args.SemEmbed == 'DOZSL_AGG_sub':
            args.semantic_of_rel = os.path.join('embeddings', 'DOZSL_AGG_sub_3000_2975.npz')
            args.input_dim = 1600

    if args.dataset == 'Wiki':
        if args.SemEmbed == 'TransE':
            args.semantic_of_rel = os.path.join('embeddings', 'TransE_65000.npz')
            args.input_dim = 200
        if args.SemEmbed == 'RGAT':
            args.semantic_of_rel = os.path.join('embeddings', 'RGAT_6400_6357.npz')
            args.input_dim = 200
        if args.SemEmbed == 'DisenE':
            args.semantic_of_rel = os.path.join('embeddings', 'DisenE_10600_10152.npz')
            args.input_dim = 400
        if args.SemEmbed == 'DisenKGAT':
            args.semantic_of_rel = os.path.join('embeddings', 'DisenKGAT_4200_4084.npz')
            args.input_dim = 800
        if args.SemEmbed == 'DOZSL_RD':
            args.semantic_of_rel = os.path.join('embeddings', 'DOZSL_RD_14400_14058.npz')
            args.input_dim = 800
        if args.SemEmbed == 'DOZSL_AGG':
            args.semantic_of_rel = os.path.join('embeddings', 'DOZSL_AGG_2600_2598.npz')
            args.input_dim = 1800
        if args.SemEmbed == 'DOZSL_AGG_sub':
            args.semantic_of_rel = os.path.join('embeddings', 'DOZSL_AGG_sub_2400_2366.npz')
            args.input_dim = 1600

    return args

if __name__ == "__main__":
    read_options()

