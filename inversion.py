import torch
import numpy as np
import os
from scipy.special import softmax


class Inversion(object):
    def __init__(self, config, model, inverse_idx, device, stage, inv_goal):
        self.syn_root = config['syn_root']
        self.iter = config['iter']
        self.batch = config['batch']
        self.r_feature = config['r_feature']
        self.lr = config['lr']
        self.weight_decay = config['weight_decay']
        self.clean_gen_data = config['clean_gen_data']
        self.target_label = config['target_label']
        self.store_best_results = config['store_best_results']
        self.base_model = config['base_model']
        self.src = config['src']
        self.tgt = config['tgt']
        self.ratio = config['ratio']
        self.src_item_dims = config['src_item_dims']
        self.tgt_item_dims = config['tgt_item_dims']
        self.inverse_idx = inverse_idx
        self.model = model
        self.model.eval()
        self.device = device
        self.stage = stage
        self.inv_goal = inv_goal
        self.criterion = torch.nn.MSELoss()

    def input_init(self, sub_idx, item_field, device):
        sub_inputs_num = len(sub_idx)
        '''one-hot representation of inversed users'''
        user_inputs_arr = np.array([sub_idx]).reshape(len(sub_idx), -1)
        item_inputs_arr = np.random.randn(sub_inputs_num, item_field)
        item_inputs_arr = softmax(item_inputs_arr, axis=1)
        sub_inputs = np.hstack((user_inputs_arr, item_inputs_arr))
        sub_inputs = sub_inputs.reshape(sub_inputs_num, item_field+1, -1).astype(np.float32)
        user_inputs, item_inputs = torch.split(torch.tensor(sub_inputs, device=device), [1, item_field], dim=1)
        return user_inputs, item_inputs

    def lr_cosine_policy(self, base_lr, warmup_length, epochs):
        def _lr_fn(iteration, epoch):
            if epoch < warmup_length:
                lr = base_lr * (epoch + 1) / warmup_length
            else:
                e = epoch - warmup_length
                es = epochs - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            return lr

        return self.lr_policy(_lr_fn)

    def lr_policy(self, lr_fn):
        def _alr(optimizer, iteration, epoch):
            lr = lr_fn(iteration, epoch)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        return _alr

    def emb_save(self, inputs, model, path):
        with open(path, "ab") as f:
            emb = model(inputs, self.inv_goal+'_inversion', 'save_inversed_iid')
            np_emb = emb.detach().cpu().numpy().squeeze()
            np.savetxt(f, np_emb, delimiter=',')

    def main(self):
        save_root = self.syn_root + str(int(self.ratio[0] * 10)) + '_' + str(int(self.ratio[1] * 10)) + \
                      '/tgt_' + self.tgt + '_src_' + self.src
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        save_file = save_root + '/' + str(self.target_label) + '_' + self.base_model + \
                    '_' + self.stage + '_' + self.inv_goal + '.csv'
        print(save_root)
        if self.clean_gen_data:
            if os.path.isfile(save_file):
                os.remove(save_file)
        ''' initial v with batch to save the memory'''
        num_inverse_idx = len(self.inverse_idx)
        all_targets = torch.FloatTensor([self.target_label] * num_inverse_idx).to(self.device)
        num_batches = (num_inverse_idx - 1) // self.batch + 1
        for i in range(num_batches):
            if i != num_batches:
                sub_idx = self.inverse_idx[i*self.batch: (i+1)*self.batch]
            else:
                sub_idx = self.inverse_idx[i*self.batch:]
            print("*****Initialization new batches")
            best_loss = 1e4
            '''initial v and target rating'''
            if self.inv_goal == 'src':
                user_inputs, item_inputs = self.input_init(sub_idx, self.src_item_dims, self.device)
            elif self.inv_goal == 'tgt':
                user_inputs, item_inputs = self.input_init(sub_idx, self.tgt_item_dims, self.device)
            else:
                raise ValueError('Unknown inversion goal: ' + self.inv_goal + ', it should be src or tgt')
            targets = all_targets[i*self.batch: (i+1)*self.batch]
            item_inputs.requires_grad_()
            inputs = torch.cat((user_inputs, item_inputs), dim=1)
            best_inputs = inputs
            optimizer = torch.optim.Adam([item_inputs], lr=self.lr, weight_decay=self.weight_decay)
            for p in self.model.parameters():
                p.requires_grad = False
            lr_scheduler = self.lr_cosine_policy(self.lr, 10, self.iter)

            for iter in range(self.iter):
                lr_scheduler(optimizer, iter, iter)
                y = self.model(inputs, self.inv_goal+"_inversion")
                loss = self.criterion(y, targets.float())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                inputs = torch.cat((user_inputs, item_inputs), dim=1)
                # record best results
                if best_loss > loss.item() or iter == 1:
                    best_inputs = inputs.data.clone()
                    best_loss = loss.item()
                if iter % 50 == 0:
                    print("------------iteration {}----------".format(iter))
                    print('predict y is:', y)
                    print('target is:', targets)
                    print("total loss", loss.item())
            if self.store_best_results:
                self.emb_save(best_inputs, self.model, save_file)
