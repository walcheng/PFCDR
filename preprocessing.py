import pandas as pd
import gzip
import json
import tqdm
import random
import os
import csv


class DataPreprocessingMid():
    def __init__(self,
                 root,
                 dealing):
        self.root = root
        self.dealing = dealing

    def main(self):
        print('Parsing ' + self.dealing + ' Mid...')
        re = []
        with gzip.open(self.root + 'raw/reviews_' + self.dealing + '_5.json.gz', 'rb') as f:
            for line in tqdm.tqdm(f, smoothing=0, mininterval=1.0):
                line = json.loads(line)
                re.append([line['reviewerID'], line['asin'], line['overall']])
        re = pd.DataFrame(re, columns=['uid', 'iid', 'y'])
        print(self.dealing + ' Mid Done.')
        re.to_csv(self.root + 'mid/' + self.dealing + '.csv', index=False)
        return re


class DataPreprocessingReady():
    def __init__(self,
                 root,
                 src_tgt_pairs,
                 task,
                 ratio):
        self.root = root
        self.src = src_tgt_pairs[task]['src']
        self.tgt = src_tgt_pairs[task]['tgt']
        self.ratio = ratio

    def read_mid(self, field):
        path = self.root + 'mid/' + field + '.csv'
        re = pd.read_csv(path)
        return re

    def mapper(self, src, tgt):
        print('Source inters: {}, uid: {}, iid: {}.'.format(len(src), len(set(src.uid)), len(set(src.iid))))
        print('Target inters: {}, uid: {}, iid: {}.'.format(len(tgt), len(set(tgt.uid)), len(set(tgt.iid))))
        set_uid_src = set(src.uid)
        set_uid_tgt = set(tgt.uid)
        set_iid_src = set(src.iid)
        set_iid_tgt = set(tgt.iid)
        co_uid = set_uid_src & set_uid_tgt
        out_uid_src = set_uid_src - co_uid
        out_uid_tgt = set_uid_tgt - co_uid
        all_uid = set_uid_src | set_uid_tgt
        print('All uid: {}, Co uid: {}.'.format(len(all_uid), len(co_uid)))
        # put co_user at the former
        uid_src_dict = dict(zip(list(co_uid) + list(out_uid_src), range(len(set_uid_src))))
        uid_tgt_dict = dict(zip(list(co_uid) + list(out_uid_tgt), range(len(set_uid_tgt))))
        iid_dict_src = dict(zip(set_iid_src, range(len(set_iid_src))))
        iid_dict_tgt = dict(zip(set_iid_tgt, range(len(set_iid_tgt))))
        src.uid = src.uid.map(uid_src_dict)
        src.iid = src.iid.map(iid_dict_src)
        tgt.uid = tgt.uid.map(uid_tgt_dict)
        tgt.iid = tgt.iid.map(iid_dict_tgt)
        return src, tgt, len(co_uid)

    def split(self, src, tgt, co_user_num):
        print('All iid: {}.'.format(len(set(src.iid) | set(tgt.iid))))
        tgt_users = set(tgt.uid.unique())
        co_users = set(list(range(co_user_num)))
        test_users = set(random.sample(co_users, round(self.ratio[1] * co_user_num)))
        train_src = src
        train_tgt = tgt[tgt['uid'].isin(tgt_users - test_users)]
        test = tgt[tgt['uid'].isin(test_users)]
        train_meta = tgt[tgt['uid'].isin(co_users - test_users)]
        return train_src, train_tgt, test, train_meta, list(test_users)

    def save(self, train_src, train_tgt, test, train_meta, test_list):
        output_root = self.root + 'ready/_' + str(int(self.ratio[0] * 10)) + '_' + str(int(self.ratio[1] * 10)) + \
                      '/tgt_' + self.tgt + '_src_' + self.src
        if not os.path.exists(output_root):
            os.makedirs(output_root)
        print(output_root)
        train_src.to_csv(output_root + '/train_src.csv', sep=',', header=None, index=False)
        train_tgt.to_csv(output_root + '/train_tgt.csv', sep=',', header=None, index=False)
        test.to_csv(output_root + '/test.csv', sep=',', header=None, index=False)
        train_meta.to_csv(output_root + '/train_meta.csv', sep=',', header=None, index=False)
        with open(output_root + '/test_list.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(test_list)

    def main(self):
        src = self.read_mid(self.src)
        tgt = self.read_mid(self.tgt)
        src, tgt, co_uid_num = self.mapper(src, tgt)
        train_src, train_tgt, test, train_meta, test_list = self.split(src, tgt, co_uid_num)
        self.save(train_src, train_tgt, test, train_meta, test_list)

