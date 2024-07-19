from sklearn.model_selection import train_test_split
import numpy as np
import argparse
import yaml
from tqdm import tqdm
import traceback
from collections import OrderedDict, defaultdict
import sys
import time
import random
import os
import pickle
import pprint


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset

from torch.optim.lr_scheduler import MultiStepLR

from loss.loss1 import *

DEVICE = torch.device("cuda:0")


def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


class DictAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super(DictAction, self).__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        input_dict = eval(f'dict({values})')  # pylint: disable=W0123
        output_dict = getattr(namespace, self.dest)
        for k in input_dict:
            output_dict[k] = input_dict[k]
        setattr(namespace, self.dest, output_dict)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)










class MyDataSet(Dataset):
    def __init__(self, X_list, label_list, debug):
        if debug == True:
            self.X_list = X_list[:100]
            self.label_list = label_list[:100]
        else:
            self.X_list = X_list
            self.label_list = label_list

    def __getitem__(self, index):
        x = self.X_list[index]
        y = self.label_list[index]
        return x, y

    def __len__(self):
        return len(self.X_list)

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label_list)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description='Spatial Temporal Graph Convolution Network For HAR')

    parser.add_argument(
        '--work-dir',
        help='the work folder for storing results')

    parser.add_argument(
        '--config',
        default='config/train_pamap2.yaml',
        help='path to the configuration file')

    # processor

    parser.add_argument(
        '--debug',
        type=str2bool,
        default=True,
        help='if ture, the classification score will be stored')

    # visulize and debug
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed for pytorch')

    parser.add_argument(
        '--print-log',
        type=str2bool,
        default=True,
        help='print logging or not')

    # data
    parser.add_argument(
        '--data-path', default='feeder.feeder', help='data loader will be used')

    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument(
        '--model-args',
        action=DictAction,
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--weights',
        default=None,
        help='the weights for network initialization')

    # optim
    parser.add_argument(
        '--base-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument(
        '--step',
        type=int,
        default=[20, 40, 60],
        nargs='+',
        help='the epoch where optimizer reduce the learning rate')

    parser.add_argument(
        '--feats-split',
        type=int,
        default=[12, 6],
        nargs='+',
        help='the epoch where optimizer reduce the learning rate')
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='the indexes of GPUs for training or testing')

    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')

    parser.add_argument(
        '--batch-size', type=int, default=256, help='training batch size')
    parser.add_argument(
        '--test-batch-size', type=int, default=256, help='test batch size')

    parser.add_argument(
        '--start-epoch',
        type=int,
        default=0,
        help='start training from which epoch')
    parser.add_argument(
        '--num-epoch',
        type=int,
        default=80,
        help='stop training in which epoch')
    parser.add_argument(
        '--flag',
        type=int,
        default=1,
        help='')
    parser.add_argument(
        '--Dili',
        type=int,
        default=0,
        help='')
    parser.add_argument(
        '--DS',
        type=int,
        default=0,
        help='')
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0005,
        help='weight decay for optimizer')
    parser.add_argument(
        '--lr-decay-rate',
        type=float,
        default=0.1,
        help='decay rate for learning rate')
    # nesterov 牛顿运算
    parser.add_argument(
        '--nesterov', type=str2bool, default=False, help='use nesterov or not')

    parser.add_argument('--checkpoint', default=None)

    parser.add_argument(
        '--show-topk',
        type=int,
        default=[1, 5],
        nargs='+',
        help='which Top K accuracy will be shown')
    return parser


class Processor():
    """Processor for Skeleton-based Action Recgnition"""

    def __init__(self, arg):
        if not os.path.exists(arg.work_dir):
            os.makedirs(arg.work_dir)
        self.arg = arg
        self.load_model()
        self.load_param_groups()
        self.load_optimizer()



        self.load_data()

        self.lr = self.arg.base_lr

        self.best_acc = 0
        self.best_acc_epoch = 0

        self.train_acc = []
        self.train_loss = []

        self.test_acc = []
        self.test_loss = []

    def load_model(self):

        output_device = self.arg.device[0] if type(
            self.arg.device) is list else self.arg.device
        self.output_device = output_device

        # import model and loss
        Model = import_class(self.arg.model)

        self.model = Model(**self.arg.model_args).cuda(output_device)
        self.lossf = nn.CrossEntropyLoss().cuda(output_device)
        self.print_log(f'Model total number of params: {count_params(self.model)}')

        if self.arg.weights:
            try:
                self.base_epoch = int(self.arg.weights[:-3].split('-')[1])
            except:
                print('Cannot parse base_epoch from model weights filename')
                self.base_epoch = 0
            self.print_log(f'Loading weights from {self.arg.weights}')

            weights = torch.load(self.arg.weights)

            weights = OrderedDict(
                [[k.split('module.')[-1],
                  v.cuda(output_device)] for k, v in weights.items()])

            try:
                self.model.load_state_dict(weights)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                self.print_log('Can not find these weights:')
                for d in diff:
                    self.print_log('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)

    def load_param_groups(self):
        """
        Template function for setting different learning behaviour
        (e.g. LR, weight decay) of different groups of parameters
        """
        self.param_groups = defaultdict(list)

        for name, params in self.model.named_parameters():
            self.param_groups['other'].append(params)

        self.optim_param_groups = {
            'other': {'params': self.param_groups['other']}
        }

    def load_optimizer(self):
        params = list(self.optim_param_groups.values())
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                params,
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov
            )
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                params,
                lr=self.arg.base_lr)
        else:
            raise ValueError('Unsupported optimizer: {}'.format(self.arg.optimizer))

            # Load optimizer states if any

    
    def load_data(self):

        self.data_loader = dict()

        def worker_seed_fn(worker_id):
            return init_seed(self.arg.seed + worker_id + 1)

        train_x = np.load(self.arg.data_path.format('x_train'))
        train_y = np.load(self.arg.data_path.format('y_train'))
        test_x = np.load(self.arg.data_path.format('x_test'))
        test_y = np.load(self.arg.data_path.format('y_test'))

        source_node = []
        target_node = []

        for node in range(train_x.shape[1] - 1):
            source_node.append(node)
            target_node.append(node + 1)

        self.edges = torch.tensor((source_node, target_node),dtype=torch.long).to(DEVICE)


        trainset = MyDataSet(train_x, train_y, self.arg.debug)
        testset = MyDataSet(test_x, test_y, self.arg.debug)

        self.data_loader['train'] = torch.utils.data.DataLoader(
            dataset=trainset,
            batch_size=self.arg.batch_size,
            shuffle=True,
            drop_last=True,
            worker_init_fn=worker_seed_fn)

        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=testset,
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            drop_last=False,
            worker_init_fn=worker_seed_fn)

    def print_log(self, s, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            s = f'[ {localtime} ] {s}'
        print(s)
        if self.arg.print_log:
            with open(os.path.join(self.arg.work_dir, 'log.txt'), 'a') as f:
                print(s, file=f)

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log(f'Local current time: {localtime}')

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def save_states(self, states, out_folder, out_name):
        out_folder_path = os.path.join(self.arg.work_dir, out_folder)
        out_path = os.path.join(out_folder_path, out_name)
        os.makedirs(out_folder_path, exist_ok=True)
        torch.save(states, out_path)

    def getU(self, alpha):
        num_classes = alpha.shape[-1]
        S = np.sum(alpha,axis=-1,keepdims=True)
        u = num_classes / S

        return np.mean(u)
    def DS_Combin(self, alpha):
        num_classes = alpha[0].shape[-1]
        def DS_Combin_two(alpha1, alpha2):
            alpha = dict()
            alpha[0], alpha[1] = alpha1, alpha2
            b, S, E, u = dict(), dict(), dict(), dict()
            for v in range(2):
                S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
                E[v] = alpha[v] - 1
                b[v] = E[v] / (S[v].expand(E[v].shape))
                u[v] = num_classes / S[v]

            # b^0 @ b^(0+1)
            bb = torch.bmm(b[0].view(-1, num_classes, 1), b[1].view(-1, 1, num_classes))
            # b^0 * u^1
            uv1_expand = u[1].expand(b[0].shape)
            bu = torch.mul(b[0], uv1_expand)
            # b^1 * u^0
            uv_expand = u[0].expand(b[0].shape)
            ub = torch.mul(b[1], uv_expand)
            # calculate C
            bb_sum = torch.sum(bb, dim=(1, 2), out=None)
            bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
            # bb_diag1 = torch.diag(torch.mm(b[v], torch.transpose(b[v+1], 0, 1)))
            C = bb_sum - bb_diag

            # calculate b^a
            b_a = (torch.mul(b[0], b[1]) + bu + ub) / ((1 - C).view(-1, 1).expand(b[0].shape))
            # calculate u^a
            u_a = torch.mul(u[0], u[1]) / ((1 - C).view(-1, 1).expand(u[0].shape))

            # calculate new S
            S_a = num_classes / u_a
            # calculate new e_k
            e_a = torch.mul(b_a, S_a.expand(b_a.shape))
            alpha_a = e_a + 1
            return alpha_a

        for v in range(len(alpha) - 1):
            if v == 0:
                alpha_a = DS_Combin_two(alpha[0], alpha[1])
            else:
                alpha_a = DS_Combin_two(alpha_a, alpha[v + 1])
        return alpha_a
    def train(self, epoch):
        self.model.train()

        loader = self.data_loader['train']
        loss_values = []
        epoch_acc = []
        self.print_log('Now epoch:{}'.format(epoch + 1))
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)

        current_lr = self.optimizer.param_groups[0]['lr']
        self.print_log(f'Training is staring. Training epoch: {epoch + 1}, LR: {current_lr:.4f}')

        process = tqdm(loader, dynamic_ncols=True)

        for batch_idx, (x_data, label) in enumerate(process):
            self.optimizer.zero_grad()
            # get data
            label = label.long().to(DEVICE)
            if self.arg.flag == 1:
                x_data = torch.tensor(x_data,dtype=torch.float32).to(DEVICE)
                timer['dataloader'] += self.split_time()
                alpha = self.model(self.edges, x_data)
                if self.arg.Dili == 1:
                    alpha = alpha + 1
            if self.arg.flag == 3:   
                feats1 = x_data[:,:, :self.arg.feats_split[0]]
                feats1 = torch.tensor(feats1,dtype=torch.float32).to(DEVICE)
    
                feats2 = x_data[:,:, self.arg.feats_split[0]:self.arg.feats_split[1]]
                feats2 = torch.tensor(feats2,dtype=torch.float32).to(DEVICE)
    
                feats3 = x_data[:,:, self.arg.feats_split[1]:]
                feats3 = torch.tensor(feats3,dtype=torch.float32).to(DEVICE)
                timer['dataloader'] += self.split_time()
                out1,out2,out3 = self.model(self.edges, feats1, feats2, feats3)
                alpha = out1 + out2 + out3
                if self.arg.Dili == 1:
                    if self.arg.DS == 1:
                        alpha = self.DS_Combin([out1+1,out2+1,out3+1])
                    else:
                        alpha = alpha + 1
                  
                
                #额外处理
            
            # backward
            if self.arg.Dili == 1:
                loss = mse_loss(alpha, label, epoch + 1)
            else:
                loss = self.lossf(alpha,label)
            
            loss.backward()
            loss_values.append(loss.item())
            timer['model'] += self.split_time()

            # Display loss

            value, predict_label = torch.max(alpha, 1)
            acc = torch.mean((predict_label == label).float())

            epoch_acc.append(acc)

            #####################################

            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2)
            self.optimizer.step()

            # statistics
            self.lr = self.optimizer.param_groups[0]['lr']

            timer['statistics'] += self.split_time()

        # statistics of time consumption and loss
        proportion = {
            k: f'{int(round(v * 100 / sum(timer.values()))):02d}%'
            for k, v in timer.items()
        }

        mean_loss = np.mean(loss_values)
        mean_acc = np.mean(np.array([item.cpu().numpy() for item in epoch_acc]))

        self.print_log(f'\tMean training loss: {mean_loss:.4f}')
        self.print_log(
            f'\tMean training acc: {mean_acc * 100:.2f}%.')
        self.print_log('\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(**proportion))


        self.train_acc.append(mean_acc)
        self.train_loss.append(mean_loss)

    def eval(self, epoch):
        with torch.no_grad():
            self.model = self.model.cuda(self.output_device)
            self.model.eval()
            self.print_log(f'Eval is staring. Eval epoch: {epoch + 1}')
            
            loss_values = []
            score_batches = []
            ln = 'test'
            step = 0
            process = tqdm(self.data_loader[ln], dynamic_ncols=True)

            for batch_idx, (x_data, label) in enumerate(process):
                # get data
                label = label.long().to(DEVICE)
                if self.arg.flag == 1:
                    x_data = torch.tensor(x_data,dtype=torch.float32).to(DEVICE)
                    output = self.model(self.edges, x_data)
                    if self.arg.Dili == 1:
                        output = output + 1
                if self.arg.flag == 3:   
                    feats1 = x_data[:,:, :self.arg.feats_split[0]]
                    feats1 = torch.tensor(feats1,dtype=torch.float32).to(DEVICE)
        
                    feats2 = x_data[:,:, self.arg.feats_split[0]:self.arg.feats_split[1]]
                    feats2 = torch.tensor(feats2,dtype=torch.float32).to(DEVICE)
        
                    feats3 = x_data[:,:, self.arg.feats_split[1]:]
                    feats3 = torch.tensor(feats3,dtype=torch.float32).to(DEVICE)
                    out1,out2,out3 = self.model(self.edges, feats1, feats2, feats3)
                    output = out1 + out2 + out3
                    if self.arg.Dili == 1:
                        if self.arg.DS == 1:
                            output = self.DS_Combin([out1+1,out2+1,out3+1])
                        else:
                            output = output + 1
            
                if self.arg.Dili == 1:
                    tloss = mse_loss(output, label, epoch + 1)
                else:
                    tloss = self.lossf(output,label)

                score_batches.append(output.data.cpu().numpy())
                loss_values.append(tloss.item())

                _, predict_label = torch.max(output.data, 1)
                step += 1

            score = np.concatenate(score_batches)
            loss = np.mean(loss_values)
            accuracy = self.data_loader[ln].dataset.top_k(score, 1)
            
            self.test_acc.append(accuracy)
            self.test_loss.append(loss)

            self.print_log(
                f'\t work_dir: {self.arg.work_dir}.')
            self.print_log(
                f'\tMean test acc: {accuracy * 100:.2f}%.')
            self.print_log(f'\tMean {ln} loss of {len(self.data_loader[ln])} batches: {np.mean(loss_values)}.')
            if self.arg.Dili == 1:
                self.print_log(
                f'\tMean test u {self.getU(score)}.')
            for k in self.arg.show_topk:
                self.print_log(f'\tTop {k}: {100 * self.data_loader[ln].dataset.top_k(score, k):.2f}%')

            if accuracy > self.best_acc:
                self.print_log(f'Last epoch to save weight and checkpoint.!!!!!!!  Epoch number is {epoch + 1}')
                self.best_acc = accuracy
                self.best_acc_epoch = epoch + 1

                weight_state_dict = self.model.state_dict()
                weights = OrderedDict([
                    [k.split('module.')[-1], v.cpu()]
                    for k, v in weight_state_dict.items()
                ])

                weights_name = f'best_acc_weights.pt'
                self.save_states(weights, './', weights_name)


        # Empty cache after evaluation
        torch.cuda.empty_cache()

    def start(self):

        self.print_log(f'Parameters:\n{pprint.pformat(vars(self.arg))}\n')
        self.print_log(f'Model total number of params: {count_params(self.model)}')

        for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
            self.train(epoch)
            self.eval(epoch)

        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.print_log(f'Best accuracy: {self.best_acc}')
        self.print_log(f'Epoch number: {self.best_acc_epoch}')
        self.print_log(f'Model work_dir: {self.arg.work_dir}')
        self.print_log(f'Model total number of params: {num_params}')
        self.print_log(f'Weight decay: {self.arg.weight_decay}')
        self.print_log(f'Base LR: {self.arg.base_lr}')
        self.print_log(f'Batch Size: {self.arg.batch_size}')
        self.print_log(f'Test Batch Size: {self.arg.test_batch_size}')

        self.print_log('Done.\n')


if __name__ == '__main__':
    print(torch.__version__)
    print(torch.cuda.is_available())
    parser = get_parser()
    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f, Loader=yaml.FullLoader)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG:', k)
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    init_seed(arg.seed)
    processor = Processor(arg)
    processor.start()