import json
import copy
import torch
from float_embedding import FloatEmbedding


with open('inversion_config.json', 'r') as f:
    inv_config = json.load(f)

target_label = inv_config['target_label']


class LookupEmbedding(torch.nn.Module):

    def __init__(self, uid_all, iid_all, emb_dim):
        super().__init__()
        self.uid_embedding = torch.nn.Embedding(uid_all, emb_dim)
        self.iid_embedding = torch.nn.Embedding(iid_all, emb_dim)

    def forward(self, x):
        uid_emb = self.uid_embedding(x[:, 0].unsqueeze(1))
        iid_emb = self.iid_embedding(x[:, 1].unsqueeze(1))
        emb = torch.cat([uid_emb, iid_emb], dim=1)
        return emb


class FloatLookupEmbedding(torch.nn.Module):

    def __init__(self, uid_all, iid_all, emb_dim):
        super().__init__()
        self.uid_embedding = torch.nn.Embedding(uid_all, emb_dim)
        self.iid_embedding = FloatEmbedding(iid_all, emb_dim)

    def forward(self, x, stage=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if x.dtype is torch.float32:
            uid_idx = x[:, 0].type(torch.LongTensor).to(device)
            uid_emb = self.uid_embedding(uid_idx)
            iid_emb = self.iid_embedding(x[:, 1:])
            iid_emb = torch.sum(iid_emb, dim=1).unsqueeze(1)
            if stage is None:
                emb = torch.cat([uid_emb, iid_emb], dim=1)
                return emb
            if stage is 'save_inversed_iid':
                a = uid_idx.unsqueeze(2)
                emb = torch.cat([uid_idx.unsqueeze(2), iid_emb], dim=2)
                return emb
        else:
            uid_emb = self.uid_embedding(x[:, 0].unsqueeze(1))
            iid_emb = self.iid_embedding(x[:, 1].unsqueeze(1))
            emb = torch.cat([uid_emb, iid_emb], dim=1)
            return emb


class GMFBase(torch.nn.Module):
    def __init__(self, uid_all, iid_all, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.embedding = LookupEmbedding(uid_all, iid_all, emb_dim)
        self.linear = torch.nn.Linear(emb_dim, 1, False)

    def forward(self, x):
        emb = self.embedding.forward(x)
        x = emb[:, 0, :] * emb[:, 1, :]
        x = self.linear(x)
        return x.squeeze(1)


class FloatGMFBase(torch.nn.Module):
    def __init__(self, uid_all, iid_all, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.embedding = FloatLookupEmbedding(uid_all, iid_all, emb_dim)
        self.linear = torch.nn.Linear(emb_dim, 1, False)

    def forward(self, x, stage):
        emb = self.embedding.forward(x, stage)
        x = emb[:, 0, :] * emb[:, 1, :]
        x = self.linear(x)
        return x.squeeze(1)


class DNNBase(torch.nn.Module):
    def __init__(self, uid_all, iid_all, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.embedding = LookupEmbedding(uid_all, iid_all, emb_dim)
        self.linear = torch.nn.Linear(emb_dim, emb_dim)

    def forward(self, x):
        emb = self.embedding.forward(x)
        x = torch.sum(self.linear(emb[:, 0, :]) * emb[:, 1, :], 1)
        return x


class FloatDNNBase(torch.nn.Module):
    def __init__(self, uid_all, iid_all, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.embedding = FloatLookupEmbedding(uid_all, iid_all, emb_dim)
        self.linear = torch.nn.Linear(emb_dim, emb_dim)

    def forward(self, x, stage):
        emb = self.embedding.forward(x, stage)
        x = torch.sum(self.linear(emb[:, 0, :]) * emb[:, 1, :], 1)
        return x


class MFBasedModel(torch.nn.Module):
    # TODO change
    def __init__(self, field_dims_src, field_dims_tgt, num_fields, emb_dim, topk):
        super().__init__()
        self.num_fields = num_fields
        self.emb_dim = emb_dim
        self.src_model = FloatLookupEmbedding(field_dims_src['uid_src'], field_dims_src['iid_src'], emb_dim)
        # TODO change
        self.tgt_model = FloatLookupEmbedding(field_dims_tgt['uid_tgt'], field_dims_tgt['iid_tgt'], emb_dim)
        # TODO change
        self.topk = topk
        self.rp_mapping = torch.nn.Linear(emb_dim, emb_dim, False)
        self.mapping = torch.nn.Linear(emb_dim, emb_dim, False)

    def forward(self, x, stage, inversion_stage=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # TODO change
        if stage in ['train_src', 'src_inversion']:
            emb = self.src_model.forward(x, inversion_stage)
            if inversion_stage is None:
                x = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
                return x
            if inversion_stage == 'save_inversed_iid':
                return emb
        # TODO change
        elif stage in ['tgt_inversion']:
            emb = self.tgt_model.forward(x, inversion_stage)
            if inversion_stage is None:
                x = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
                return x
            if inversion_stage == 'save_inversed_iid':
                return emb
        elif stage in ['train_tgt', 'test_tgt']:
            emb = self.tgt_model.forward(x)
            x = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
            return x
        elif stage == 'train_map':
            src_emb = self.src_model.uid_embedding(x.unsqueeze(1)).squeeze()
            src_emb = self.mapping.forward(src_emb)
            tgt_emb = self.tgt_model.uid_embedding(x.unsqueeze(1)).squeeze()
            return src_emb, tgt_emb
        elif stage == 'test_map':
            uid_emb = self.mapping.forward(self.src_model.uid_embedding(x[:, 0].unsqueeze(1)).squeeze())
            emb = self.tgt_model.forward(x)
            ###
            # ground_truth = emb[:, 0, :].detach().cpu().numpy()
            # transform = uid_emb.detach().cpu().numpy()
            # a = ground_truth[:100, ]
            # np.savetxt('./emb/emcdr_gt.csv', ground_truth[:100, ], delimiter=',')
            # np.savetxt('./emb/emcdr_t.csv', transform[:100, ], delimiter=',')
            ###
            emb[:, 0, :] = uid_emb
            x = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
            return x
        elif stage == 'train_source_free':
            # input x organized with [uid, src_rate_pre, tgt_rate_pre]
            src_rate_pre = x[:, 1:self.emb_dim+1]
            tgt_rate_pre = x[:, self.emb_dim+1:]
            src_rate_pre = self.rp_mapping.forward(src_rate_pre)
            return tgt_rate_pre, src_rate_pre
        elif stage == 'test_source_free':
            # input x organized with [iid, src_rate_pre]
            # Our aim is 1: transfer src_rate_pre to tgt domain; 2: find topk users in tgt domain
            # 3: vote mechanism to get the rate
            iid = x[:, 0].type(torch.LongTensor).to(device)
            iid_emb = self.tgt_model.iid_embedding(iid.unsqueeze(1))
            src_rate_pre = x[:, 1:self.emb_dim+1]
            src_rate_pre = self.rp_mapping.forward(src_rate_pre)
            tgt_user_embedding = copy.deepcopy(self.tgt_model.uid_embedding.weight)
            predict_rate = torch.mm(tgt_user_embedding, src_rate_pre.T) - torch.tensor((target_label))
            predict_rate = torch.abs(predict_rate)
            sorted, indices = torch.sort(predict_rate, dim=0)
            topk = 50000
            topk_indices = indices[:topk, :]
            topk_indices = topk_indices.T.reshape(-1, 1).squeeze()
            topk_uid_emb = self.tgt_model.uid_embedding(topk_indices).view(len(x), topk, self.emb_dim)
            voted_rating = torch.sum(topk_uid_emb * iid_emb, dim=2)
            mean_rating = torch.mean(voted_rating, dim=1)

            return mean_rating


class GMFBasedModel(torch.nn.Module):
    def __init__(self, field_dims_src, field_dims_tgt, num_fields, emb_dim):
        super().__init__()
        self.num_fields = num_fields
        self.emb_dim = emb_dim
        self.src_model = FloatGMFBase(field_dims_src['uid_src'], field_dims_src['iid_src'], emb_dim)
        self.tgt_model = FloatGMFBase(field_dims_tgt['uid_tgt'], field_dims_tgt['iid_tgt'], emb_dim)
        self.rp_mapping = torch.nn.Linear(emb_dim, emb_dim, False)
        self.mapping = torch.nn.Linear(emb_dim, emb_dim, False)

    def forward(self, x, stage, inversion_stage=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if stage in ['train_src', 'src_inversion']:
            emb = self.src_model.embedding.forward(x, inversion_stage)
            if inversion_stage is None:
                x = emb[:, 0, :] * emb[:, 1, :]
                x = self.src_model.linear(x)
                return x.squeeze(1)
            if inversion_stage == 'save_inversed_iid':
                return emb
        elif stage in ['tgt_inversion']:
            emb = self.tgt_model.embedding.forward(x, inversion_stage)
            if inversion_stage is None:
                x = emb[:, 0, :] * emb[:, 1, :]
                x = self.tgt_model.linear(x)
                return x.squeeze(1)
            if inversion_stage == 'save_inversed_iid':
                return emb
        elif stage in ['train_tgt', 'test_tgt']:
            x = self.tgt_model.forward(x, inversion_stage)
            return x
        elif stage == 'train_map':
            src_emb = self.src_model.embedding.uid_embedding(x.unsqueeze(1)).squeeze()
            src_emb = self.mapping.forward(src_emb)
            tgt_emb = self.tgt_model.embedding.uid_embedding(x.unsqueeze(1)).squeeze()
            return src_emb, tgt_emb
        elif stage == 'test_map':
            uid_emb = self.mapping.forward(self.src_model.embedding.uid_embedding(x[:, 0].unsqueeze(1)))
            iid_emb = self.tgt_model.embedding.iid_embedding(x[:, 1].unsqueeze(1))
            emb = torch.cat([uid_emb, iid_emb], 1)
            x = self.tgt_model.linear(emb[:, 0, :] * emb[:, 1, :])
            return x.squeeze(1)
        elif stage in ['test_meta', 'train_task_oriented_meta']:
            # input x is organized with [test_uid, tgt_test_iid, personalized embedding(10dim)]
            iid = x[:, 1].type(torch.LongTensor).to(device)
            uid = x[:, 0].type(torch.LongTensor).to(device)
            iid_emb = self.tgt_model.embedding.iid_embedding(iid.unsqueeze(1))
            uid_emb_src = self.src_model.embedding.uid_embedding(uid.unsqueeze(1))
            mapping = self.meta_net.forward(x[:, 2:]).view(-1, self.emb_dim, self.emb_dim)
            uid_emb = torch.bmm(uid_emb_src, mapping)
            emb = torch.cat([uid_emb, iid_emb], 1)
            output = self.tgt_model.linear(emb[:, 0, :] * emb[:, 1, :]).squeeze(1)
            return output
        elif stage == 'train_source_free':
            # input x organized with [uid, src_rate_pre, tgt_rate_pre]
            src_rate_pre = x[:, 1:self.emb_dim+1]
            tgt_rate_pre = x[:, self.emb_dim+1:]
            src_rate_pre = self.rp_mapping.forward(src_rate_pre)
            return tgt_rate_pre, src_rate_pre
        elif stage == 'test_source_free':
            # input x organized with [iid, src_rate_pre]
            # Our aim is 1: transfer src_rate_pre to tgt domain; 2: find topk users in tgt domain
            # 3: vote mechanism to get the rate
            iid = x[:, 0].type(torch.LongTensor).to(device)
            iid_emb = self.tgt_model.embedding.iid_embedding(iid.unsqueeze(1))
            src_rate_pre = x[:, 1:self.emb_dim+1]
            src_rate_pre = self.rp_mapping.forward(src_rate_pre)
            tgt_user_embedding = copy.deepcopy(self.tgt_model.embedding.uid_embedding.weight)
            predict_rate = torch.mm(tgt_user_embedding, src_rate_pre.T) - torch.tensor((target_label))
            predict_rate = torch.abs(predict_rate)
            sorted, indices = torch.sort(predict_rate, dim=0)
            topk = 10000
            topk_indices = indices[:topk, :]
            topk_indices = topk_indices.T.reshape(-1, 1).squeeze()
            topk_uid_emb = self.tgt_model.embedding.uid_embedding(topk_indices).view(len(x), topk, self.emb_dim)
            a = topk_uid_emb * iid_emb
            voted_rating = self.tgt_model.linear((topk_uid_emb * iid_emb).view(len(x)*topk, self.emb_dim))
            voted_rating = voted_rating.view(len(x), topk)
            mean_rating = torch.mean(voted_rating, dim=1)

            return mean_rating



class DNNBasedModel(torch.nn.Module):
    def __init__(self, field_dims_src, field_dims_tgt, num_fields, emb_dim, topk):
        super().__init__()
        self.num_fields = num_fields
        self.emb_dim = emb_dim
        self.src_model = FloatDNNBase(field_dims_src['uid_src'], field_dims_src['iid_src'], emb_dim)
        self.tgt_model = FloatDNNBase(field_dims_tgt['uid_tgt'], field_dims_tgt['iid_tgt'], emb_dim)
        self.topk = topk
        self.rp_mapping = torch.nn.Linear(emb_dim, emb_dim, False)
        self.mapping = torch.nn.Linear(emb_dim, emb_dim, False)

    def forward(self, x, stage, inversion_stage=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if stage in ['train_src', 'src_inversion']:
            emb = self.src_model.embedding.forward(x, inversion_stage)
            if inversion_stage is None:
                x = torch.sum(self.src_model.linear(emb[:, 0, :]) * emb[:, 1, :], 1)
                return x
            if inversion_stage == 'save_inversed_iid':
                return emb
        if stage in ['tgt_inversion']:
            emb = self.tgt_model.embedding.forward(x, inversion_stage)
            if inversion_stage is None:
                x = torch.sum(self.tgt_model.linear(emb[:, 0, :]) * emb[:, 1, :], 1)
                return x
            if inversion_stage == 'save_inversed_iid':
                return emb
        elif stage in ['train_tgt', 'test_tgt']:
            x = self.tgt_model.forward(x, inversion_stage)
            return x
        elif stage == 'train_map':
            src_emb = self.src_model.linear(self.src_model.embedding.uid_embedding(x.unsqueeze(1)).squeeze())
            src_emb = self.mapping.forward(src_emb)
            tgt_emb = self.tgt_model.linear(self.tgt_model.embedding.uid_embedding(x.unsqueeze(1)).squeeze())
            return src_emb, tgt_emb
        elif stage == 'test_map':
            uid_emb = self.mapping.forward(self.src_model.linear(self.src_model.embedding.uid_embedding(x[:, 0].unsqueeze(1))))
            iid_emb = self.tgt_model.embedding.iid_embedding(x[:, 1].unsqueeze(1))
            emb = torch.cat([uid_emb, iid_emb], 1)
            x = torch.sum(emb[:, 0, :] * emb[:, 1, :], 1)
            return x
        elif stage == 'train_source_free':
            # input x organized with [uid, src_rate_pre, tgt_rate_pre]
            src_rate_pre = x[:, 1:self.emb_dim+1]
            tgt_rate_pre = x[:, self.emb_dim+1:]
            src_rate_pre = self.rp_mapping.forward(src_rate_pre)
            return tgt_rate_pre, src_rate_pre
        elif stage == 'test_source_free':
            # input x organized with [iid, src_rate_pre]
            # Our aim is 1: transfer src_rate_pre to tgt domain; 2: find topk users in tgt domain
            # 3: vote mechanism to get the rate
            iid = x[:, 0].type(torch.LongTensor).to(device)
            iid_emb = self.tgt_model.embedding.iid_embedding(iid.unsqueeze(1))
            src_rate_pre = x[:, 1:self.emb_dim+1]
            src_rate_pre = self.rp_mapping.forward(src_rate_pre)
            tgt_user_embedding = copy.deepcopy(self.tgt_model.embedding.uid_embedding.weight)
            predict_rate = torch.mm(self.tgt_model.linear(tgt_user_embedding), src_rate_pre.T) - torch.tensor((target_label))
            predict_rate = torch.abs(predict_rate)
            sorted, indices = torch.sort(predict_rate, dim=0)
            topk = 15000
            topk_indices = indices[:topk, :]
            topk_indices = topk_indices.T.reshape(-1, 1).squeeze()
            topk_uid_emb = self.tgt_model.embedding.uid_embedding(topk_indices).view(len(x), topk, self.emb_dim)
            voted_rating = torch.sum(self.tgt_model.linear(topk_uid_emb) * iid_emb, dim=2)
            mean_rating = torch.mean(voted_rating, dim=1)
            return mean_rating

