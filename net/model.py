import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlight import import_class


class CoDT(nn.Module):
    """ Referring to the code of MOCO, https://arxiv.org/abs/1911.05722 """

    def __init__(self, base_encoder=None, base_encoder1=None, feature_dim=128, queue_size=32768,
                 momentum=0.999, Temperature=0.07, in_channels=3, hidden_channels=64,
                 hidden_dim=256, num_class=60, num_cluster=30, dropout=0.5,
                 graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'},
                 graph_args1={'layout': 'ntu-rgb+d', 'strategy': 'spatial'},
                 edge_importance_weighting=True,
                 bias=True,
                 **kwargs):
        """
        K: queue size; number of negative keys (default: 32768)
        m: momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """

        super().__init__()
        base_encoder = import_class(base_encoder)
        base_encoder1 = import_class(base_encoder1)

        self.K = queue_size

        self.m = momentum
        self.T = Temperature

        if graph_args['layout'] == 'skeletics':
            self.parts = [[2, 3, 4], [5, 6, 7], [0, 1, 8], [9, 10, 11], [12, 13, 14]]
        elif graph_args['layout'] == 'ntu-rgb+d':
            self.parts = [[3 - 1, 4 - 1, 1 - 1, 2 - 1, 21 - 1],
                          [5 - 1, 6 - 1, 7 - 1, 8 - 1, 22 - 1, 23 - 1],
                          [9 - 1, 10 - 1, 11 - 1, 12 - 1, 24 - 1, 25 - 1],
                          [13 - 1, 14 - 1, 15 - 1, 16 - 1],
                          [17 - 1, 18 - 1, 19 - 1, 20 - 1]]
        elif graph_args['layout'] == 'openpose':
            self.parts = [[2, 3, 4, 1],
                          [5, 6, 7, 1],
                          [8, 9, 10, 1],
                          [11, 12, 13, 1],
                          [0, 14, 15, 16, 17]]
        elif graph_args['layout'] == 'skeletics25':
            self.parts = [[2, 3, 4, 1],
                          [5, 6, 7, 1],
                          [8, 9, 10, 11],
                          [8, 12, 13, 14],
                          [0, 15, 16, 17, 18]]

        self.encoder_q = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                      hidden_dim=hidden_dim, dropout=dropout, graph_args=graph_args,
                                      edge_importance_weighting=edge_importance_weighting,
                                      **kwargs)
        self.encoder_k = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                      hidden_dim=hidden_dim, dropout=dropout, graph_args=graph_args,
                                      edge_importance_weighting=edge_importance_weighting,
                                      **kwargs)

        self.encoder_q1 = base_encoder1(in_channels=in_channels, hidden_channels=hidden_channels,
                                        hidden_dim=hidden_dim, dropout=dropout, graph_args=graph_args1,
                                        edge_importance_weighting=edge_importance_weighting,
                                        **kwargs)
        self.encoder_k1 = base_encoder1(in_channels=in_channels, hidden_channels=hidden_channels,
                                        hidden_dim=hidden_dim, dropout=dropout, graph_args=graph_args1,
                                        edge_importance_weighting=edge_importance_weighting,
                                        **kwargs)

        self.fc_q = nn.Linear(hidden_dim, num_class)

        self.fc_q1 = nn.Linear(hidden_dim, feature_dim)
        self.fc_k1 = nn.Linear(hidden_dim, feature_dim)

        for param_k in self.encoder_k.parameters():
            param_k.requires_grad = False

        for param_k in self.encoder_k1.parameters():
            param_k.requires_grad = False

        for param_k in self.fc_k1.parameters():
            param_k.requires_grad = False

        # create the queue
        self.register_buffer("queue1", torch.randn(self.K, hidden_dim))
        self.register_buffer("queue", torch.randn(self.K, hidden_dim))

        self.decoder = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                     nn.ReLU(inplace=True),
                                     nn.Dropout(0.5),
                                     nn.Linear(hidden_dim, in_channels * self.encoder_q.A.size(1)))

        self.cf_q_j0 = nn.Linear(hidden_dim, num_cluster, bias=bias)
        self.cf_k_j0 = nn.Linear(hidden_dim, num_cluster, bias=bias)

        self.cf_q_j1 = nn.Linear(hidden_dim, num_cluster, bias=bias)
        self.cf_k_j1 = nn.Linear(hidden_dim, num_cluster, bias=bias)

        for param_k in self.cf_k_j0.parameters():
            param_k.requires_grad = False

        for param_k in self.cf_k_j1.parameters():
            param_k.requires_grad = False

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _momentum_update_key_encoder1(self):
        for param_q, param_k in zip(self.encoder_q1.parameters(), self.encoder_k1.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _momentum_update_fc1(self):
        for param_q, param_k in zip(self.fc_q1.parameters(), self.fc_k1.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _momentum_update_cf(self):
        for param_q, param_k in zip(self.cf_q_j0.parameters(), self.cf_k_j0.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        for param_q, param_k in zip(self.cf_q_j1.parameters(), self.cf_k_j1.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, ids):
        self.queue[ids] = keys

    @torch.no_grad()
    def _dequeue_and_enqueue1(self, keys, ids):
        self.queue1[ids] = keys

    @torch.no_grad()
    def distributed_sinkhorn(self, PS, lam):
        PS = F.softmax(PS, dim=1)
        N, K = PS.shape
        PS = PS.T  # now it is K x N
        r = torch.ones((K, 1), device=PS.device) / K
        c = torch.ones((N, 1), device=PS.device) / N
        PS = torch.pow(PS, lam)
        inv_K = 1. / K
        inv_N = 1. / N
        err = 1e3
        _counter = 0
        while err > 1e-2 and _counter < 100:
            r = inv_K / (PS @ c)  # (KxN)@(N,1) = K x 1
            c_new = inv_N / (r.T @ PS).T  # ((1,K)@(KxN)).t() = N x 1
            if _counter % 10 == 0:
                err = torch.nansum(torch.abs(c / c_new - 1))
            c = c_new
            _counter += 1
        PS *= torch.squeeze(c)
        PS = PS.T
        PS *= torch.squeeze(r)
        PS *= N
        return PS

    def masking(self, im_q, mask_p):
        N, C, T, V, M = im_q.size()
        mask = torch.rand(N, 1, T, len(self.parts), M) > mask_p
        mask = mask.to(im_q.device)
        for i in range(len(self.parts)):
            im_q[:, :, :, self.parts[i]] = im_q[:, :, :, self.parts[i]] * mask[:, :, :, i:(i + 1)]
        return im_q

    def regression(self, im_q, dec):
        N, C, T, V, M = im_q.size()
        T1 = dec.size(1)
        ori = F.adaptive_avg_pool3d(im_q, (T1, V, M))
        ori = ori.permute(0, 4, 2, 3, 1).contiguous()
        ori = ori.view(N * M, T1, V * C)
        dec_err = F.mse_loss(dec, ori)
        return dec_err

    def forward(self, im_q=None, im_k=None, im_q1=None, im_k1=None, ids=None, lam=10, mask_p=0.0):
        if im_k is None:
            ori = im_q.clone()
            if mask_p > 0:
                im_q = self.masking(im_q, mask_p)
            feat_q, dec = self.encoder_q(im_q)
            dec = dec.permute(0, 2, 1).contiguous()
            dec = self.decoder(dec)
            dec_err = self.regression(ori, dec)
            fc_q = self.fc_q(feat_q)
            return fc_q, dec_err

        ori = im_q.clone()
        if mask_p > 0:
            im_q = self.masking(im_q, mask_p)
        feat_q, dec = self.encoder_q(im_q)
        # decoding
        dec = dec.permute(0, 2, 1).contiguous()
        dec = self.decoder(dec)
        dec_err = self.regression(ori, dec)

        if mask_p > 0:
            im_q1 = self.masking(im_q1, mask_p)
        feat_q1, _ = self.encoder_q1(im_q1)
        fc_q1 = self.fc_q1(feat_q1)
        fc_q1 = F.normalize(fc_q1, dim=1)

        with torch.no_grad():
            self._momentum_update_key_encoder()
            self._momentum_update_key_encoder1()
            self._momentum_update_fc1()

            feat_k, _ = self.encoder_k(im_k)
            feat_k1, _ = self.encoder_k1(im_k1)
            fc_k1 = self.fc_k1(feat_k1)
            fc_queue1 = self.fc_k1(self.queue1.clone().detach())
            fc_k1 = F.normalize(fc_k1, dim=1)
            fc_queue1 = F.normalize(fc_queue1, dim=1)

        ############################# unsupervised #################
        l_pos = torch.einsum('nc,nc->n', fc_q1, fc_k1).unsqueeze(-1)
        l_neg = torch.einsum('nc,kc->nk', fc_q1, fc_queue1)
        l_neg.scatter_(1, ids.unsqueeze(-1), -1e10)

        logits_j = torch.cat((l_pos, l_neg), dim=1)
        logits_j /= self.T

        labels = torch.zeros(logits_j.shape[0], dtype=torch.long, device=logits_j.device)
        self._dequeue_and_enqueue1(feat_k1, ids)

        cf_q_j0 = self.cf_q_j0(feat_q)
        cf_q_j1 = self.cf_q_j1(feat_q1)

        with torch.no_grad():  # no gradient to keys
            self._momentum_update_cf()
            self._dequeue_and_enqueue(feat_k, ids)
            cf_queue_j0 = self.cf_k_j0(self.queue.clone().detach())
            cf_queue_j1 = self.cf_k_j1(self.queue1.clone().detach())
            cf_k_j0 = self.distributed_sinkhorn(cf_queue_j0, lam)[ids]
            cf_k_j1 = self.distributed_sinkhorn(cf_queue_j1, lam)[ids]
        return [logits_j, labels, dec_err, cf_q_j0, cf_q_j1, cf_k_j0, cf_k_j1]
