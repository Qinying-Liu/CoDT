import argparse
import numpy as np
import random

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# torchlight
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class

from .processor import Processor

from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score


def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    # return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size
    return w[ind].sum() / y_pred.size


def init_seed(seed=1):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1 or classname.find('Conv2d') != -1 or classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def kmeans(data, k):
    indices = torch.randperm(data.size()[0])[:k]
    centroids = data[indices]
    classes = torch.arange(k).to(centroids.device)
    for i in range(1000):
        distances = torch.cdist(data, centroids)
        labels = torch.argmin(distances, -1)
        mask = (classes[:, None] == labels).float()  # K, N
        new_centroids = torch.matmul(mask, data) / mask.sum(1, keepdim=True).clamp(1, )
        if torch.abs(centroids - new_centroids).mean() < 1e-10:
            break
        centroids = new_centroids
    return centroids, labels


class CLS_Processor(Processor):
    def load_weights(self):
        if self.arg.weights:
            self.model = self.io.load_weights(self.model, self.arg.weights, self.arg.ignore_weights)
        if self.arg.weights1:
            self.model = self.io.load_weights(self.model, self.arg.weights1, self.arg.ignore_weights1)

    def load_model(self):
        self.model = self.io.load_model(self.arg.model, **self.arg.model_args)
        self.model.apply(weights_init)

        self.loss = nn.CrossEntropyLoss()

    def load_data(self):
        self.data_loader = dict()

        source_feeder = import_class(self.arg.source_feeder)
        self.data_loader['source'] = torch.utils.data.DataLoader(
            dataset=source_feeder(**self.arg.source_feeder_args),
            batch_size=self.arg.batch_size,
            shuffle=True,
            pin_memory=True,  # set True when memory is abundant
            num_workers=self.arg.num_worker * torchlight.ngpu(
                self.arg.device),
            drop_last=True,
            worker_init_fn=init_seed)

        target_feeder = import_class(self.arg.target_feeder)
        self.data_loader['target'] = torch.utils.data.DataLoader(
            dataset=target_feeder(**self.arg.target_feeder_args),
            batch_size=self.arg.batch_size,
            shuffle=True,
            pin_memory=True,  # set True when memory is abundant
            num_workers=self.arg.num_worker * torchlight.ngpu(
                self.arg.device),
            drop_last=True,
            worker_init_fn=init_seed)

        train_feeder = import_class(self.arg.train_feeder)
        self.data_loader['train'] = torch.utils.data.DataLoader(
            dataset=train_feeder(**self.arg.train_feeder_args),
            batch_size=self.arg.batch_size,
            shuffle=False,
            pin_memory=True,  # set True when memory is abundant
            num_workers=self.arg.num_worker * torchlight.ngpu(
                self.arg.device),
            drop_last=False,
            worker_init_fn=init_seed)

        test_feeder = import_class(self.arg.test_feeder)
        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=test_feeder(**self.arg.test_feeder_args),
            batch_size=self.arg.batch_size,
            shuffle=False,
            pin_memory=True,  # set True when memory is abundant
            num_workers=self.arg.num_worker * torchlight.ngpu(self.arg.device),
            drop_last=False,
            worker_init_fn=init_seed)

    def load_optimizer(self):
        parameters = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                parameters,
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                parameters,
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

    def adjust_lr(self):
        if self.arg.optimizer == 'SGD' and self.arg.step:
            lr = self.arg.base_lr * (
                    0.1 ** np.sum(self.meta_info['epoch'] > np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.lr = lr
        else:
            self.lr = self.arg.base_lr

    def show_acc(self):
        targets = self.test_label.cpu().numpy()
        for i in range(self.test_result.size(-1)):
            acc_list_tmp = []
            nmi_list_tmp = []
            ari_list_tmp = []
            for t in range(10):
                _, preds = kmeans(self.test_result[..., i], self.arg.model_args['num_cluster'])
                preds = preds.cpu().numpy()
                acc, nmi, ari = cluster_acc(targets, preds), nmi_score(targets, preds), ari_score(targets, preds)
                acc_list_tmp.append(acc)
                nmi_list_tmp.append(nmi)
                ari_list_tmp.append(ari)
            acc = sum(acc_list_tmp) / 10
            nmi = sum(nmi_list_tmp) / 10
            ari = sum(ari_list_tmp) / 10
            self.io.print_log(
                '\tKmeans: Test acc {:.2f}, nmi {:.2f}, ari {:.2f}'.format(acc * 100, nmi * 100, ari * 100))
        for i in range(self.test_pred.size(-1)):
            preds = torch.argmax(self.test_pred[..., i], dim=1).cpu().numpy()
            acc, nmi, ari = cluster_acc(targets, preds), nmi_score(targets, preds), ari_score(targets, preds)
            self.io.print_log('\tPred: Test acc {:.2f}, nmi {:.2f}, ari {:.2f}'.format(acc * 100, nmi * 100, ari * 100))

    def train(self, epoch):
        target_loader = self.data_loader['target']
        source_loader = self.data_loader['source']
        self.model.train()
        self.adjust_lr()

        loader_iter = iter(source_loader)
        for [data_q, data_k, data_q1, data_k1], _, ids in target_loader:
            try:
                data_s, label_s = next(loader_iter)
            except:
                loader_iter = iter(source_loader)
                data_s, label_s = next(loader_iter)

            data_q = data_q.float().to(self.dev, non_blocking=True)
            data_k = data_k.float().to(self.dev, non_blocking=True)

            data_q1 = data_q1.float().to(self.dev, non_blocking=True)
            data_k1 = data_k1.float().to(self.dev, non_blocking=True)

            ids = ids.long().to(self.dev, non_blocking=True)
            data_s = data_s.float().to(self.dev, non_blocking=True)
            label_s = label_s.long().to(self.dev, non_blocking=True)

            logit_s, dec_err_s = self.model(data_s, mask_p=self.arg.mask_p)
            loss_ce = F.cross_entropy(logit_s, label_s)

            output = self.model(data_q, data_k, data_q1, data_k1,
                                lam=self.arg.lam,
                                ids=ids,
                                mask_p=self.arg.mask_p)
            output_un = output[0]
            target = output[1]
            loss_un = F.cross_entropy(output_un, target)
            dec_err_t = output[2]
            loss_dec = dec_err_s + dec_err_t
            loss_cls = torch.zeros([]).to(self.dev)
            loss_cls1 = torch.zeros([]).to(self.dev)
            if self.arg.co_training and epoch > self.arg.co_epoch:
                if epoch == self.arg.co_epoch:
                    train_loader = self.data_loader['train']
                    train_result, _, _ = self.run(train_loader)
                    queue_feat = F.normalize(train_result[..., 0], dim=1)
                    queue_feat1 = F.normalize(train_result[..., 1], dim=1)
                    protosj0, _ = kmeans(queue_feat, self.arg.model_args['num_cluster'])
                    protosj1, _ = kmeans(queue_feat1, self.arg.model_args['num_cluster'])
                    self.model.cf_q_j0.weight.data.copy_(protosj0)
                    self.model.cf_q_j1.weight.data.copy_(protosj1)
                    self.model.cf_k_j0.weight.data.copy_(protosj0)
                    self.model.cf_k_j1.weight.data.copy_(protosj1)

                cf_q_j0, cf_q_j1, cf_k_j0, cf_k_j1 = output[3:]
                cf_v_j0, cf_l_j0 = cf_k_j0.max(dim=1)
                cf_v_j1, cf_l_j1 = cf_k_j1.max(dim=1)

                loss_cls += F.cross_entropy(cf_q_j0, cf_l_j0)
                loss_cls += F.cross_entropy(cf_q_j1, cf_l_j1)

                sim_l_j1 = (cf_l_j1[:, None] == cf_l_j1[None, :]).float()
                cf_q_j0_co = cf_q_j0.softmax(dim=1)
                sim_f_j0 = cf_q_j0_co @ cf_q_j0_co.t()
                sim_f_j0 = sim_f_j0.clamp(1e-8, 1. - 1e-8)
                sim_f_j0 = F.normalize(sim_f_j0, p=1, dim=1)
                loss_tmp = (sim_f_j0.log() * sim_l_j1).sum(1) / sim_l_j1.sum(1)
                loss_cls1 += - loss_tmp.mean()

                sim_l_j0 = (cf_l_j0[:, None] == cf_l_j0[None, :]).float()
                cf_q_j1_co = cf_q_j1.softmax(dim=1)
                sim_f_j1 = cf_q_j1_co @ cf_q_j1_co.t()
                sim_f_j1 = sim_f_j1.clamp(1e-8, 1. - 1e-8)
                sim_f_j1 = F.normalize(sim_f_j1, p=1, dim=1)
                loss_tmp = (sim_f_j1.log() * sim_l_j0).sum(1) / sim_l_j0.sum(1)
                loss_cls1 += - loss_tmp.mean()

            loss = self.arg.w_ce * loss_ce + self.arg.w_un * loss_un + self.arg.w_cls * loss_cls + \
                   self.arg.w_dec * loss_dec + self.arg.w_cls1 * loss_cls1

            self.global_step += 1
            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.iter_info['loss'] = loss.data.item()
            self.iter_info['loss_ce'] = loss_ce.data.item()
            self.iter_info['loss_un'] = loss_un.data.item()
            self.iter_info['loss_cls'] = loss_cls.data.item()
            self.iter_info['loss_cls1'] = loss_cls1.data.item()
            self.iter_info['loss_dec'] = loss_dec.data.item()
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)

            self.show_iter_info()
            self.meta_info['iter'] += 1

    def test(self, epoch):
        self.model.eval()
        test_loader = self.data_loader['test']
        self.test_result, self.test_label, self.test_pred = self.run(test_loader)
        self.show_acc()

    def run(self, loader):
        result_frag = []
        label_frag = []
        pred_frag = []
        for [data, data1], label, _ in loader:
            data = data.float().to(self.dev, non_blocking=True)
            data1 = data1.float().to(self.dev, non_blocking=True)
            label = label.long().to(self.dev, non_blocking=True)
            with torch.no_grad():
                output = self.model.encoder_q(data)[0]
                output1 = self.model.encoder_q1(data1)[0]
                pred_j0 = self.model.cf_q_j0(output)
                pred_j1 = self.model.cf_q_j1(output1)
                output = torch.stack((output, output1), dim=-1)
                pred = torch.stack((pred_j0, pred_j1), dim=-1)
            result_frag.append(output.data)
            label_frag.append(label.data)
            pred_frag.append(pred.data)
        result_frag = torch.cat(result_frag, 0)
        label_frag = torch.cat(label_frag, 0)
        pred_frag = torch.cat(pred_frag, 0)
        return result_frag, label_frag, pred_frag

    @staticmethod
    def get_parser(add_help=False):

        # parameter priority: command line > config > default
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Spatial Temporal Graph Convolution Network')

        # region arguments yapf: disable
        # evaluation
        parser.add_argument('--show_topk', type=int, default=[1, 5], nargs='+',
                            help='which Top K accuracy will be shown')
        # optim
        parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--step', type=int, default=[], nargs='+',
                            help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
        parser.add_argument('--weights1', help='type of optimizer')
        parser.add_argument('--ignore_weights1', default=None, help='type of optimizer')
        parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')
        parser.add_argument('--view', type=str, default='joint', help='the view of input')
        parser.add_argument('--dual_feeder', type=str, default='joint', help='the view of input')
        parser.add_argument('--dual_feeder_args', type=str, default='joint', help='the view of input')
        parser.add_argument('--single_feeder', type=str, default='joint', help='the view of input')
        parser.add_argument('--single_feeder_args', type=str, default='joint', help='the view of input')
        parser.add_argument('--source_feeder', type=str, default='joint', help='the view of input')
        parser.add_argument('--source_feeder_args', type=str, default='joint', help='the view of input')
        parser.add_argument('--target_feeder', type=str, default='joint', help='the view of input')
        parser.add_argument('--target_feeder_args', type=str, default='joint', help='the view of input')
        parser.add_argument('--cross_epoch', type=int, default=1e6, help='the starting epoch of cross-view training')
        parser.add_argument('--context', type=str2bool, default=True, help='using context knowledge')
        parser.add_argument('--topk', type=int, default=1, help='topk samples in cross-view training')
        parser.add_argument('--num_cluster', type=int, default=20, help='topk samples in cross-view training')
        parser.add_argument('--co_training', type=str2bool, default=True, help='topk samples in cross-view training')
        parser.add_argument('--is_hard', type=str2bool, default=True, help='topk samples in cross-view training')
        parser.add_argument('--on_target', type=str2bool, default=True, help='topk samples in cross-view training')
        parser.add_argument('--w_ce', type=float, default=1, help='weight decay for optimizer')
        parser.add_argument('--w_un', type=float, default=1, help='weight decay for optimizer')
        parser.add_argument('--w_cls', type=float, default=1, help='weight decay for optimizer')
        parser.add_argument('--w_cls1', type=float, default=1, help='weight decay for optimizer')
        parser.add_argument('--w_dec', type=float, default=10, help='weight decay for optimizer')
        parser.add_argument('--mask_p', type=float, default=20, help='initial learning rate')
        parser.add_argument('--lam', type=float, default=10, help='initial learning rate')
        parser.add_argument('--co_epoch', type=int, default=-1, help='initial learning rate')
        return parser
