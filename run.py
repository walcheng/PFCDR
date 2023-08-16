import os.path
import json
import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import tqdm
from models import MFBasedModel, GMFBasedModel, DNNBasedModel
from inversion import Inversion

class Run():
    def __init__(self,
                 config
                 ):
        self.config = config
        self.use_cuda = config['use_cuda']
        self.base_model = config['base_model']
        self.root = config['root']
        self.model_root = config['model_root']
        self.ratio = config['ratio']
        self.task = config['task']
        self.co_user_num = config["src_tgt_pairs"][self.task]['co_user_num']
        self.src = config['src_tgt_pairs'][self.task]['src']
        self.tgt = config['src_tgt_pairs'][self.task]['tgt']
        self.uid_src = config['src_tgt_pairs'][self.task]['uid_src']
        self.iid_src = config['src_tgt_pairs'][self.task]['iid_src']
        self.uid_tgt = config['src_tgt_pairs'][self.task]['uid_tgt']
        self.iid_tgt = config['src_tgt_pairs'][self.task]['iid_tgt']
        self.field_dims_src = {'uid_src': self.uid_src, "iid_src": self.iid_src}
        self.field_dims_tgt = {'uid_tgt': self.uid_tgt, "iid_tgt": self.iid_tgt}

        self.batchsize_src = config['src_tgt_pairs'][self.task]['batchsize_src']
        self.batchsize_tgt = config['src_tgt_pairs'][self.task]['batchsize_tgt']
        self.batchsize_meta = config['src_tgt_pairs'][self.task]['batchsize_meta']
        self.batchsize_map = config['src_tgt_pairs'][self.task]['batchsize_map']
        self.batchsize_test = config['src_tgt_pairs'][self.task]['batchsize_test']
        self.topk = config['src_tgt_pairs'][self.task]['topk']
        self.batchsize_aug = self.batchsize_src

        self.epoch = config['epoch']
        self.emb_dim = config['emb_dim']
        self.meta_dim = config['meta_dim']
        self.num_fields = config['num_fields']
        self.lr = config['lr']
        self.lr_prototype = config['lr_prototype']
        self.wd = config['wd']

        self.input_root = self.root + 'ready/_' + str(int(self.ratio[0] * 10)) + '_' + str(int(self.ratio[1] * 10)) + \
            '/tgt_' + self.tgt + '_src_' + self.src
        self.src_path = self.input_root + '/train_src.csv'
        self.tgt_path = self.input_root + '/train_tgt.csv'
        self.meta_path = self.input_root + '/train_meta.csv'
        self.test_path = self.input_root + '/test.csv'
        self.test_idx_path = self.input_root + '/test_list.csv'

        self.results = {'tgt_mae': 10, 'tgt_rmse': 10,
                        'emcdr_mae': 10, 'emcdr_rmse': 10,
                        'sfcdr_mae': 10, 'sfcdr_rmse': 10}

    def read_log_data(self, path, batchsize):
        cols = ['uid', 'iid', 'y']
        x_col = ['uid', 'iid']
        y_col = ['y']
        data = pd.read_csv(path, header=None)
        data.columns = cols
        X = torch.tensor(data[x_col].values, dtype=torch.long)
        y = torch.tensor(data[y_col].values, dtype=torch.long)
        if self.use_cuda:
            X = X.cuda()
            y = y.cuda()
        dataset = TensorDataset(X, y)
        data_iter = DataLoader(dataset, batchsize, shuffle=True)
        return data_iter

    def read_map_data(self, train_mapping_idx):
        X = torch.tensor(np.array(train_mapping_idx), dtype=torch.long)
        y = torch.tensor(np.array(range(X.shape[0])), dtype=torch.long)
        if self.use_cuda:
            X = X.cuda()
            y = y.cuda()
        dataset = TensorDataset(X, y)
        data_iter = DataLoader(dataset, self.batchsize_map, shuffle=True)
        return data_iter

    def read_rating_preference(self, path):
        #src_rate_pre consists by the user id and corresponding prototype
        src_rate_pre_file = path + '_train_src.csv'
        tgt_rate_pre_file = path + '_train_tgt.csv'
        co_uid_cols = ['co_uid']
        src_rate_pre_cols = ['src_rate_pre_'+str(x) for x in range(self.emb_dim)]
        src_rate_pre_cols = co_uid_cols + src_rate_pre_cols
        tgt_rate_pre_cols = ['tgt_rate_pre_'+str(x) for x in range(self.emb_dim)]
        tgt_rate_pre_cols = co_uid_cols + tgt_rate_pre_cols
        src_rate_pre = pd.read_csv(src_rate_pre_file, header=None, names=src_rate_pre_cols)
        tgt_rate_pre = pd.read_csv(tgt_rate_pre_file, header=None, names=tgt_rate_pre_cols)
        merged_rate_pre = pd.merge(src_rate_pre, tgt_rate_pre, on='co_uid')
        X = torch.tensor(merged_rate_pre.values, dtype=torch.float32)
        y = torch.tensor(np.array(range(merged_rate_pre.shape[0])), dtype=torch.long)
        print('map {} iter / batchsize = {} '.format(len(X), self.batchsize_meta))
        if self.use_cuda:
            X = X.cuda()
            y = y.cuda()
        dataset = TensorDataset(X, y)
        data_iter = DataLoader(dataset, self.batchsize_meta, shuffle=True)
        return data_iter

    def read_meta_test(self, per_iid_path, data_path):
        src_rate_pre_file = per_iid_path + '_test_src.csv'
        uid_cols = ['uid']
        iid_cols = ['iid']
        y_col = ['y']
        cols = uid_cols+iid_cols+y_col
        data_root = pd.read_csv(data_path, header=None, names=cols)
        per_iid_cols = [str(x) for x in range(self.emb_dim)]
        cols = uid_cols + per_iid_cols
        rate_per = pd.read_csv(src_rate_pre_file, header=None, names=cols)
        rate_per['uid'] = rate_per['uid'].astype(int)
        test_data_with_per_iid = pd.merge(data_root, rate_per, on='uid')
        test_data_with_per_iid = test_data_with_per_iid.astype(float)
        # test data x is [test_co_uid, test_iid, per_iid_emb]
        X = torch.tensor(test_data_with_per_iid[iid_cols+per_iid_cols].values, dtype=torch.float32)
        y = torch.tensor(test_data_with_per_iid[y_col].values, dtype=torch.long)
        if self.use_cuda:
            X = X.cuda()
            y = y.cuda()
        dataset = TensorDataset(X, y)
        data_iter = DataLoader(dataset, self.batchsize_test, shuffle=True)
        return data_iter

    def get_data(self, co_user_num, inversion=False):
        print('========Reading data========')

        data_test = self.read_log_data(self.test_path, self.batchsize_test)
        print('test {} iter / batchsize = {} '.format(len(data_test), self.batchsize_test))

        test_idx = pd.read_csv(self.test_idx_path, header=None, index_col=False).values.tolist()[0]
        mapping_idx = list(set(list(range(co_user_num))) - set(test_idx))

        data_map = self.read_map_data(mapping_idx)
        print('map {} iter / batchsize = {} '.format(len(data_map), self.batchsize_map))

        if inversion:
            return data_test, test_idx, mapping_idx
        else:
            data_src = self.read_log_data(self.src_path, self.batchsize_src)
            print('src {} iter / batchsize = {} '.format(len(data_src), self.batchsize_src))

            data_tgt = self.read_log_data(self.tgt_path, self.batchsize_tgt)
            print('tgt {} iter / batchsize = {} '.format(len(data_tgt), self.batchsize_tgt))

            return data_src, data_tgt, data_test, data_map, test_idx, mapping_idx

    def get_model(self):
        if self.base_model == 'MF':
            model = MFBasedModel(self.field_dims_src, self.field_dims_tgt, self.num_fields, self.emb_dim, self.topk)
        elif self.base_model == 'DNN':
            model = DNNBasedModel(self.field_dims_src, self.field_dims_tgt, self.num_fields, self.emb_dim, self.topk)
        elif self.base_model == 'GMF':
            model = GMFBasedModel(self.field_dims_src, self.field_dims_tgt, self.num_fields, self.emb_dim)
        else:
            raise ValueError('Unknown base model: ' + self.base_model)
        return model.cuda() if self.use_cuda else model

    def get_optimizer(self, model):
        optimizer_src = torch.optim.Adam(params=model.src_model.parameters(), lr=self.lr, weight_decay=self.wd)
        optimizer_tgt = torch.optim.Adam(params=model.tgt_model.parameters(), lr=self.lr, weight_decay=self.wd)
        optimizer_map = torch.optim.Adam(params=model.mapping.parameters(), lr=self.lr, weight_decay=self.wd)
        return optimizer_src, optimizer_tgt, optimizer_map

    def eval_mae(self, model, data_loader, stage):
        print('Evaluating MAE:')
        model.eval()
        targets, predicts = list(), list()
        loss = torch.nn.L1Loss()
        mse_loss = torch.nn.MSELoss()
        with torch.no_grad():
            for X, y in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
                pred = model(X, stage)
                targets.extend(y.squeeze(1).tolist())
                predicts.extend(pred.tolist())
        targets = torch.tensor(targets).float()
        predicts = torch.tensor(predicts)
        return loss(targets, predicts).item(), torch.sqrt(mse_loss(targets, predicts)).item()

    def train(self, data_loader, model, criterion, optimizer, epoch, stage, mapping=False):
        print('Training Epoch {}:'.format(epoch + 1))
        model.train()
        for X, y in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            if mapping:
                src_emb, tgt_emb = model(X, stage)
                loss = criterion(src_emb, tgt_emb)
            else:
                pred = model(X, stage)
                loss = criterion(pred, y.squeeze().float())
            model.zero_grad()
            loss.backward()
            optimizer.step()

    def update_results(self, mae, rmse, phase):
        if mae < self.results[phase + '_mae']:
            self.results[phase + '_mae'] = mae
        if rmse < self.results[phase + '_rmse']:
            self.results[phase + '_rmse'] = rmse

    def TgtOnly(self, model, data_tgt, data_test, criterion, optimizer):
        print('=========TgtOnly========')
        for i in range(self.epoch):
            self.train(data_tgt, model, criterion, optimizer, i, stage='train_tgt')
            mae, rmse = self.eval_mae(model, data_test, stage='test_tgt')
            self.update_results(mae, rmse, 'tgt')
            print('MAE: {} RMSE: {}'.format(mae, rmse))

    def EMCDR(self, model, data_src, data_map, data_test,
              criterion, optimizer_src, optimizer_map):
        print('=====EMCDR Pretraining=====')
        for i in range(self.epoch):
            self.train(data_src, model, criterion, optimizer_src, i, stage='train_src')
        print('==========EMCDR==========')
        for i in range(self.epoch):
            # self.train(data_map, model, criterion, optimizer_map, i, stage='test_map')
            self.train(data_map, model, criterion, optimizer_map, i, stage='train_map', mapping=True)
            mae, rmse = self.eval_mae(model, data_test, stage='test_map')
            self.update_results(mae, rmse, 'emcdr')
            print('MAE: {} RMSE: {}'.format(mae, rmse))

    def SFCDR(self, model, inverse_idx, test_idx, criterion, optimizer):
        # TODO: 利用inverse_idx逆向出personalized的item embedding， 并保存
        # prepare inversion parameters
        print('=====SFCDR Preparing=====')
        mapping = False
        inversion_config_path = 'inversion_config.json'
        with open(inversion_config_path, 'r') as f:
            inversion_config = json.load(f)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # personalized user embedding root
        per_iid_emb_root = inversion_config['syn_root'] + str(int(self.ratio[0] * 10)) + '_' \
                           + str(int(self.ratio[1] * 10)) + '/tgt_' + self.tgt + '_src_' + self.src + '/' + \
                           str(inversion_config['target_label']) + '_' + self.base_model
        '''没有生成personalized user embedding则执行Inversion'''
        inversion_config["task"] = self.task
        inversion_config["src"] = self.src
        inversion_config["tgt"] = self.tgt
        inversion_config['ratio'] = self.ratio
        inversion_config['base_model'] = self.base_model
        inversion_config['src_item_dims'] = self.iid_src
        inversion_config['tgt_item_dims'] = self.iid_tgt
        if not os.path.exists(per_iid_emb_root + '_train_src.csv') or \
                not os.path.exists(per_iid_emb_root + '_test_src.csv'):
            Inversion(inversion_config, model, inverse_idx, device, stage="train", inv_goal='src').main()
            Inversion(inversion_config, model, test_idx, device, stage="test", inv_goal='src').main()
        if not os.path.exists(per_iid_emb_root + '_train_tgt.csv'):
            Inversion(inversion_config, model, inverse_idx, device, stage="train", inv_goal='tgt').main()
            # During inversion, model parameters were set requires_grad = False
            for p in model.parameters():
                p.requires_grad = True

        data_rate_pre_train = self.read_rating_preference(per_iid_emb_root)
        mapping = True
        data_rate_pre_test = self.read_meta_test(per_iid_emb_root, self.test_path)
        for i in range(self.epoch):
            self.train(data_rate_pre_train, model, criterion, optimizer, i, stage='train_source_free', mapping=mapping)
            mae, rmse = self.eval_mae(model, data_rate_pre_test, stage='test_source_free')
            self.update_results(mae, rmse, 'sfcdr')
            print('MAE: {} RMSE: {}'.format(mae, rmse))

    def get_pretrained(self, root_path, str_ratio):
        old_model = torch.load(f"{root_path}{str_ratio}{self.base_model}.pt")
        old_model.eval()
        if self.use_cuda:
            old_model.cuda()

        return old_model

    def main(self):
        str_ratio = str(int(self.ratio[0]*10))+'_'+str(int(self.ratio[1]*10))+'_'
        model_root = self.model_root + '/tgt_' + self.tgt + '_src_' + self.src + '/'
        criterion = torch.nn.MSELoss()
        if os.path.exists(f"{model_root}{str_ratio}{self.base_model}.pt"):
            INVERSION = True
            model = self.get_pretrained(model_root, str_ratio)
            data_test, test_idx, inverse_idx = self.get_data(self.co_user_num, inversion=INVERSION)
            mae, rmse = self.eval_mae(model, data_test, stage='test_tgt')
            self.update_results(mae, rmse, 'tgt')
            mae, rmse = self.eval_mae(model, data_test, stage='test_map')
            self.update_results(mae, rmse, 'emcdr')
        else:
            model = self.get_model()
            if self.use_cuda:
                model.cuda()
            data_src, data_tgt, data_test, data_map, test_idx, inverse_idx = self.get_data(self.co_user_num)
            optimizer_src, optimizer_tgt, optimizer_map = self.get_optimizer(model)
            self.TgtOnly(model, data_tgt, data_test, criterion, optimizer_tgt)
            self.EMCDR(model, data_src, data_map, data_test,
                       criterion, optimizer_src, optimizer_map)
            '''model save for SFCDR'''
            if not os.path.exists(model_root):
                os.makedirs(model_root)
            torch.save(model, f"{model_root}{str_ratio}{self.base_model}.pt")
        # optimizer_rp_map for mapping rating preference from source to target
        optimizer_rp_map = torch.optim.Adam(params=model.rp_mapping.parameters(), lr=self.lr_prototype, weight_decay=self.wd)
        self.SFCDR(model, inverse_idx, test_idx, criterion, optimizer_rp_map)
        # torch.save(model, f"{model_root}{str_ratio}{self.base_model}.pt")
        print(model)
        print(self.results)
