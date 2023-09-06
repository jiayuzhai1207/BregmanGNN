import torch
import argparse
import numpy as np
import os
import pandas as pd
from activation import *
from data_proc import load_data
from model import *
from itertools import product

def build_model(args, dataset):
    if args.model == 'gcn':
        model = GCNNet(dataset.num_node_features,
                    dataset.num_classes,
                    nhid=args.nhid,
                    activation=args.activation,
                    dropout_prob=args.dropout,
                    version = args.version).to(device)
    elif args.model == 'gat':
        model = GATNet(dataset.num_node_features,
                    dataset.num_classes,
                    nhid=args.nhid,
                    activation=args.activation,
                    dropout_prob=args.dropout,
                    version = args.version).to(device)
    elif args.model == 'sage':
        model = SAGENet(dataset.num_node_features,
                    dataset.num_classes,
                    nhid=args.nhid,
                    activation=args.activation,
                    dropout_prob=args.dropout,
                    version = args.version).to(device)
    elif args.model == 'cheb':
        model = ChebNet(dataset.num_node_features,
                    dataset.num_classes,
                    nhid=args.nhid,
                    activation=args.activation,
                    dropout_prob=args.dropout,
                    version = args.version).to(device)
    elif args.model == 'appnp':
        model = APPNPNet(dataset.num_node_features,
                         dataset.num_classes,
                         nhid=args.nhid,
                         activation=args.activation,
                         K=10,
                         alpha=0.1,
                         dropout_prob=args.dropout,
                         version = args.version).to(device)
    elif args.model == 'gin':
        model = GINNet(dataset.num_node_features,
                       dataset.num_classes,
                       nhid=args.nhid,
                       activation=args.activation,
                       dropout_prob=args.dropout,
                       version=args.version).to(device)

    return model



def train_homo(args):
    for rep in range(num_reps):
        print('****** Rep {}: training start ******'.format(rep + 1))
        max_acc = 0.0
        record_test_acc = 0.0

        # initialize the model: setting cutoff to True makes the first layer as hard high-frequency cut-off
        model = build_model(args, dataset).to(device)

        # initialize the optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        for epoch in range(num_epochs):
            # training mode
            model.train()
            optimizer.zero_grad()
            out = model(data, edge_index)
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()

            # evaluation mode
            model.eval()
            out = model(data, edge_index)
            for i, mask in data('train_mask', 'val_mask', 'test_mask'):
                pred = out[mask].max(dim=1)[1]
                correct = float(pred.eq(data.y[mask]).sum().item())
                e_acc = correct / mask.sum().item()
                epoch_acc[i][rep, epoch] = e_acc

                e_loss = F.nll_loss(out[mask], data.y[mask])
                epoch_loss[i][rep, epoch] = e_loss

            # scheduler.step(epoch_loss['val_mask'][rep, epoch])

            # print out results
            if (epoch + 1) % 2 == 0:
                print('Epoch: {:3d}'.format(epoch + 1),
                      'train_loss: {:.4f}'.format(epoch_loss['train_mask'][rep, epoch]),
                      'train_acc: {:.4f}'.format(epoch_acc['train_mask'][rep, epoch]),
                      'val_loss: {:.4f}'.format(epoch_loss['val_mask'][rep, epoch]),
                      'val_acc: {:.4f}'.format(epoch_acc['val_mask'][rep, epoch]),
                      'test_loss: {:.4f}'.format(epoch_loss['test_mask'][rep, epoch]),
                      'test_acc: {:.4f}'.format(epoch_acc['test_mask'][rep, epoch]))

            # save model
            if epoch > 10:  # and epoch < 171:
                if epoch_acc['val_mask'][rep, epoch] > max_acc:
                    # torch.save(model.state_dict(), SaveResultFilename + '.pth')
                    print('Epoch: {:3d}'.format(epoch + 1),
                          'train_loss: {:.4f}'.format(epoch_loss['train_mask'][rep, epoch]),
                          'train_acc: {:.4f}'.format(epoch_acc['train_mask'][rep, epoch]),
                          'val_loss: {:.4f}'.format(epoch_loss['val_mask'][rep, epoch]),
                          'val_acc: {:.4f}'.format(epoch_acc['val_mask'][rep, epoch]),
                          'test_loss: {:.4f}'.format(epoch_loss['test_mask'][rep, epoch]),
                          'test_acc: {:.4f}'.format(epoch_acc['test_mask'][rep, epoch]))
                    print('=== Model saved at epoch: {:3d}'.format(epoch + 1))
                    max_acc = epoch_acc['val_mask'][rep, epoch]
                    record_test_acc = epoch_acc['test_mask'][rep, epoch]

        saved_model_val_acc[rep] = max_acc
        saved_model_test_acc[rep] = record_test_acc
        print('#### Rep {0:2d} Finished! val acc: {1:.4f}, test acc: {2:.4f} ####\n'.format(rep + 1, max_acc,
                                                                                            record_test_acc))


def train_hetero(args):
    for split in range(10):
        print('****** Rep {}: training start ******'.format(split + 1))
        max_acc = 0.0
        record_test_acc = 0.0

        # initialize the model: setting cutoff to True makes the first layer as hard high-frequency cut-off
        model = build_model(args, dataset).to(device)

        # initialize the optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # initialize the learning rate scheduler: change lr overtime
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

        train_mask = data.train_mask[:, split]
        val_mask = data.val_mask[:, split]
        test_mask = data.test_mask[:, split]

        # training
        for epoch in range(num_epochs):
            # training mode
            model.train()
            optimizer.zero_grad()
            out = model(data, edge_index)  # training result (forward pass)
            loss = F.nll_loss(out[train_mask], data.y[train_mask])  # negative llh loss
            loss.backward()  # backward pass
            optimizer.step()  # update parameters

            # evaluation mode
            model.eval()
            out = model(data, edge_index)  # eval result
            for i, mask in data('train_mask', 'val_mask', 'test_mask'):
                pred = out[mask[:,split]].max(dim=1)[1]
                correct = float(pred.eq(data.y[mask[:,split]]).sum().item())  # number of correct prediction
                e_acc = correct / mask[:,split].sum().item()  # #correct/#all
                epoch_acc[i][split, epoch] = e_acc

                e_loss = F.nll_loss(out[mask[:,split]], data.y[mask[:,split]])
                epoch_loss[i][split, epoch] = e_loss

            # scheduler.step(epoch_loss['val_mask'][rep, epoch])

            # print out results
            if (epoch + 1) % 2 == 0:
                print('Epoch: {:3d}'.format(epoch + 1),
                      'train_loss: {:.4f}'.format(epoch_loss['train_mask'][split, epoch]),
                      'train_acc: {:.4f}'.format(epoch_acc['train_mask'][split, epoch]),
                      'val_loss: {:.4f}'.format(epoch_loss['val_mask'][split, epoch]),
                      'val_acc: {:.4f}'.format(epoch_acc['val_mask'][split, epoch]),
                      'test_loss: {:.4f}'.format(epoch_loss['test_mask'][split, epoch]),
                      'test_acc: {:.4f}'.format(epoch_acc['test_mask'][split, epoch]))

            # save model
            if epoch > 10:  # and epoch < 171:
                if epoch_acc['val_mask'][split, epoch] > max_acc:
                    # torch.save(model.state_dict(), SaveResultFilename + '.pth')
                    print('Epoch: {:3d}'.format(epoch + 1),
                          'train_loss: {:.4f}'.format(epoch_loss['train_mask'][split, epoch]),
                          'train_acc: {:.4f}'.format(epoch_acc['train_mask'][split, epoch]),
                          'val_loss: {:.4f}'.format(epoch_loss['val_mask'][split, epoch]),
                          'val_acc: {:.4f}'.format(epoch_acc['val_mask'][split, epoch]),
                          'test_loss: {:.4f}'.format(epoch_loss['test_mask'][split, epoch]),
                          'test_acc: {:.4f}'.format(epoch_acc['test_mask'][split, epoch]))
                    print('=== Model saved at epoch: {:3d}'.format(epoch + 1))
                    max_acc = epoch_acc['val_mask'][split, epoch]
                    record_test_acc = epoch_acc['test_mask'][split, epoch]
        # for name, param in model.named_parameters():
        #    if param.requires_grad:
        #        print(name, param.data)
        saved_model_val_acc[split] = max_acc
        saved_model_test_acc[split] = record_test_acc
        print('#### Rep {0:2d} Finished! val acc: {1:.4f}, test acc: {2:.4f} ####\n'.format(split + 1, max_acc,
                                                                                            record_test_acc))

if __name__ == '__main__':

    dataset = ['texas','cora']
    model = ['gcn']
    lr = [0.1, 0.5, 0.01, 0.05, 0.001, 0.005]
    wd = [0.01, 0.05, 0.1, 0.005, 0.001]
    activation = ['relu', 'isru', 'sigmoid', 'tanh', 'scaled_tanh', 'atan', 'sin', 'asinh', 'softplus']
    nhid = [16, 32, 64, 96]
    dropout =[0.4, 0.5, 0.6, 0.7]
    version = ['bregman']

    grid = product(dataset, model,activation, lr, wd, nhid, dropout, version)
    print(grid)
    for dataset, model,activation, lr, wd, nhid, dropout, version in grid:
        print(dataset, model,activation, lr, wd, nhid, dropout, version)

        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset', type=str, default='wisconsin',
                            help='name of dataset (default: cora)')
        parser.add_argument('--model', type=str, default='gat',
                            help='name of model (default: gcn)')
        parser.add_argument('--reps', type=int, default=10,
                            help='number of repetitions (default: 10)')
        parser.add_argument('--epochs', type=int, default=200,
                            help='number of epochs to train (default: 200)')
        parser.add_argument('--lr', type=float, default=0.005,
                            help='learning rate (default: 5e-3)')
        parser.add_argument('--wd', type=float, default=0.01,
                            help='weight decay (default: 5e-3)')
        parser.add_argument('--nhid', type=int, default=16,
                            help='number of hidden units (default: 16)')
        parser.add_argument('--activation', type=str, default='tanh',
                            help='name of dataset (default: Cora)')
        parser.add_argument('--dropout', type=float, default=0.7,
                            help='dropout probability (default: 0.3)')
        parser.add_argument('--version', type=str, default='bregman',
                            help='model version (default: standard)')  # bregman, standard
        parser.add_argument('--seed', type=int, default=1000,
                            help='random seed (default: 1000)')
        parser.add_argument('--ExpNum', type=int, default='1',
                            help='The Experiment Number (default: 1)')

        args = parser.parse_args()
        args.dataset, args.model,args.activation, args.lr, args.wd, args.nhid, args.dropout, args.version = dataset, model,activation, lr, wd, nhid, dropout, version

        print(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        # Training on CPU/GPU device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)

        # load dataset and prepare noise through high frequence of Laplacian
        dataset, data, num_nodes, nfeatures, edge_index = load_data(args)

        '''
        Training Scheme
        '''
        # Hyper-parameter Settings
        learning_rate = args.lr  # default=0.005
        weight_decay = args.wd  # weight decay (default: 5e-3)
        nhid = args.nhid  # number of hidden units (default: 16)

        # create result matrices
        num_epochs = args.epochs  # number of epochs to train (default: 200)
        num_reps = args.reps  # number of repetitions (default: 4)
        epoch_loss = dict()
        epoch_acc = dict()
        splits = 10
        epoch_loss['train_mask'] = np.zeros((splits, num_epochs))
        epoch_acc['train_mask'] = np.zeros((splits, num_epochs))
        epoch_acc['val_mask'] = np.zeros((splits, num_epochs))
        epoch_loss['val_mask'] = np.zeros((splits, num_epochs))
        epoch_loss['test_mask'] = np.zeros((splits, num_epochs))
        epoch_acc['test_mask'] = np.zeros((splits, num_epochs))
        saved_model_val_acc = np.zeros(splits)
        saved_model_test_acc = np.zeros(splits)

        # SaveResultFilename = 'ResultExpBregman_Exp{0:03d}'.format(args.ExpNum)
        ResultCSV = args.dataset + args.version + args.model + 'Repeat{0:03d}'.format(num_reps) + '.csv'

        if args.dataset in {'cora', 'citeseer', 'pubmed', 'computers', 'photo', 'cs', 'physics'}:
            train_homo(args)
        elif args.dataset in {'cornell', 'texas', 'wisconsin', 'actor', 'chameleon', 'squirrel'}:
            train_hetero(args)
        else:
            raise Exception('Invalid Dataset')

        if os.path.isfile(ResultCSV):
            df = pd.read_csv(ResultCSV)
        else:
            outputs_names = {name: type(value).__name__ for (name, value) in args._get_kwargs()}
            outputs_names.update({'Replicate{0:2d}'.format(ii): 'float' for ii in range(1, num_reps + 1)})
            outputs_names.update({'Ave_Test_Acc': 'float', 'Test_Acc_std': 'float'})
            df = pd.DataFrame({c: pd.Series(dtype=t) for c, t in outputs_names.items()})

        new_row = {name: value for (name, value) in args._get_kwargs()}
        new_row.update({'Replicate{0:2d}'.format(ii): saved_model_test_acc[ii - 1] for ii in range(1, num_reps + 1)})
        new_row.update({'Ave_Test_Acc': np.mean(saved_model_test_acc), 'Test_Acc_std': np.std(saved_model_test_acc)})
        df = df.append(new_row, ignore_index=True)
        df.to_csv(ResultCSV, index=False)

        # save the results

        message = 'Experiment with seed {0:5d}: Average test accuracy over {1:2d} reps: {2:.4f} with stdev {3:.4f}\n'.format(
            args.seed, num_reps, np.mean(saved_model_test_acc), np.std(saved_model_test_acc))
        message += 'dataset: {0}; model: {1}; epochs: {2:3d}; reps: {3:2d}; learning_rate: {4:.5f}; weight_decay: {5:.4f}; nhid: {6:3d}; dropout: {7:.2f}; version: {8};\n' \
            .format(args.dataset, args.model, args.epochs, args.reps, args.lr, args.wd, args.nhid, args.dropout,
                    args.version)

        print(
            '***************************************************************************************************************************')
        print(message)
        print(
            '***************************************************************************************************************************')







