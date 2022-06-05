from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import json
import os
import numpy as np
import torch

from torch.utils.data import DataLoader


from model import KGEModel

from dataloader import TrainDataset
from dataloader import BidirectionalOneShotIterator


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--cuda', action='store_true', help='use GPU', default=True)
    parser.add_argument('--device', default='0', help='')

    parser.add_argument('--do_train', action='store_true', default=True)
    parser.add_argument('--do_valid', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--evaluate_train', action='store_true', help='Evaluate on training data', default=True)

    parser.add_argument('--datadir', type=str, default='../data')
    parser.add_argument('--dataset', type=str, default='Wiki')


    parser.add_argument('--save_name', type=str, default='')

    parser.add_argument('--model', default='TransE', type=str)

    parser.add_argument('--negative_sample_size', default=1024, type=int)
    parser.add_argument('--hidden_dim', default=100, type=int)
    parser.add_argument('--gamma', default=12, type=float)
    parser.add_argument('--negative_adversarial_sampling', action='store_true', default=True)
    parser.add_argument('--adversarial_temperature', default=1, type=float)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--regularization', default=0.0, type=float)
    parser.add_argument('--test_batch_size', default=8, type=int, help='valid/test batch size')
    parser.add_argument('--uni_weight', action='store_true',
                        help='Otherwise use subsampling weighting like in word2vec')

    parser.add_argument('--learning_rate', default=0.00005, type=float)
    parser.add_argument('--cpu_num', default=10, type=int)


    parser.add_argument('--max_steps', default=80000, type=int)
    parser.add_argument('--warm_up_steps', default=None, type=int)

    parser.add_argument('--save_steps', default=1000, type=int)
    parser.add_argument('--valid_steps', default=1000, type=int)
    parser.add_argument('--print_steps', default=100, type=int, help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=1000, type=int, help='valid/test log every xx steps')

    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')

    return parser.parse_args(args)


def save_embeddings(model, step, args):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''
    file_name = 'entity_' + str(step)
    entity_embedding = model.entity_embedding.detach().cpu().numpy()

    np.save(
        os.path.join(args.save_path, file_name),
        entity_embedding
    )

    rel_file_name = 'relation_' + str(step)
    relation_embedding = model.relation_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, rel_file_name),
        relation_embedding
    )



def read_triple(file_path, entity2id, relation2id):
    '''
    Read triples and map them into ids.
    '''
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples


def log_metrics(mode, step, metrics):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        print('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    args.data_path = os.path.join(args.data_dir, args.dataset)
    kg_file = os.path.join(args.data_path, 'triples.txt')


    save_path = os.path.join(args.data_path, args.save_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # load data

    ent_set, rel_set = OrderedSet(), OrderedSet()
    for line in open(kg_file):
        sub, rel, obj = map(str.lower, line.strip().split('\t'))
        ent_set.add(sub)
        rel_set.add(rel)
        ent_set.add(obj)

    if args.dataset == 'Wiki':
        Wiki_relation_file = '../../ZS_KGC/data/Wiki/relation2ids_1'

        dataset_rels = list(json.load(open(Wiki_relation_file)).keys())
        for rel in dataset_rels:
            rel = 'wikidata:' + rel.lower()
            if rel not in ent_set:
                ent_set.add(rel)

    entity2id = {ent: idx for idx, ent in enumerate(ent_set)}
    relation2id = {rel: idx for idx, rel in enumerate(rel_set)}
    json.dump(entity2id, open('{}/{}.txt'.format(args.data_path, 'ent2id'), 'w'))
    json.dump(relation2id, open('{}/{}.txt'.format(args.data_path, 'rel2id'), 'w'))


    nentity = len(entity2id)
    nrelation = len(relation2id)

    args.nentity = nentity
    args.nrelation = nrelation

    print('Model: %s' % args.model)
    print('#entity num: %d' % nentity)
    print('#relation num: %d' % nrelation)

    all_triples = read_triple(kg_file, entity2id,
                              relation2id)
    print('#total triples num: %d' % len(all_triples))


    # All true triples
    all_true_triples = all_triples

    kge_model = KGEModel(
        model_name=args.model,
        nentity=nentity,
        nrelation=nrelation,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma
    )


    if args.cuda:
        kge_model = kge_model.cuda()

    if args.do_train:
        # Set training dataloader iterator
        train_dataloader_head = DataLoader(
            TrainDataset(all_triples, nentity, nrelation, args.negative_sample_size, 'head-batch'),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=TrainDataset.collate_fn
        )

        train_dataloader_tail = DataLoader(
            TrainDataset(all_triples, nentity, nrelation, args.negative_sample_size, 'tail-batch'),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=TrainDataset.collate_fn
        )

        train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)

        # Set training configuration
        current_learning_rate = args.learning_rate

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, kge_model.parameters()),
            lr=current_learning_rate
        )
        if args.warm_up_steps:
            warm_up_steps = args.warm_up_steps
        else:
            warm_up_steps = args.max_steps // 2

    print('Ramdomly Initializing %s Model...' % args.model)

    # step = init_step

    print('------ Start Training...')
    print('batch_size = %d' % args.batch_size)
    print('negative sample size = %d' % args.negative_sample_size)
    print('hidden_dim = %d' % args.hidden_dim)
    print('gamma = %f' % args.gamma)
    print('negative_adversarial_sampling = %s' % str(args.negative_adversarial_sampling))

    if args.negative_adversarial_sampling:
        print('adversarial_temperature = %f' % args.adversarial_temperature)

    print("learning rate = %f" % current_learning_rate)

    # Set valid dataloader as it would be evaluated during training

    if args.do_train:

        train_losses = []

        # Training Loop
        for step in range(1, args.max_steps + 1):

            loss_values = kge_model.train_step(kge_model, optimizer, train_iterator, args)

            train_losses.append(loss_values)

            if step >= warm_up_steps:
                current_learning_rate = current_learning_rate / 10
                print('Change learning_rate to %f at step %d' % (current_learning_rate, step))
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, kge_model.parameters()),
                    lr=current_learning_rate
                )
                warm_up_steps = warm_up_steps * 3

            if step % args.print_steps == 0:
                pos_sample_loss = sum([losses['pos_sample_loss'] for losses in train_losses]) / len(train_losses)
                neg_sample_loss = sum([losses['neg_sample_loss'] for losses in train_losses]) / len(train_losses)
                loss1 = sum([losses['loss'] for losses in train_losses]) / len(train_losses)

                # log_metrics('Training average', step, metrics)
                print('Training Step: %d; average -> pos_sample_loss: %f; neg_sample_loss: %f; loss: %f' %
                      (step, pos_sample_loss, neg_sample_loss, loss1))
                train_losses = []

            if step % args.save_steps == 0:
                save_embeddings(kge_model, step, args)

            if args.evaluate_train and step % args.valid_steps == 0:
                print('------ Evaluating on Training Dataset...')
                metrics = kge_model.test_step(kge_model, all_triples, all_true_triples, args)
                log_metrics('Test', step, metrics)


if __name__ == '__main__':


    main(parse_args())
