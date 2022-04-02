import math
import numpy as np
import os.path as osp

import torch
import torch.nn.functional as F

from torch.utils.data import Dataset
from sklearn.metrics import roc_curve
from scipy.interpolate import interp1d

from .utils import image_pipeline, get_metrics


class IJBDataset(Dataset):
    def __init__(self, name, data_dir, meta_dir,
            data_ann_file, tmpl_ann_file,
            gallery_ann_files, probe_ann_files,
            pair_ann_file, src_landmark, test_mode=True):
        super().__init__()

        self.name = name
        self.data_dir = data_dir
        self.data_ann_path = osp.join(meta_dir, data_ann_file)
        self.tmpl_ann_path = osp.join(meta_dir, tmpl_ann_file)
        self.gallery_ann_paths = [
            osp.join(meta_dir, g_ann_file)
            for g_ann_file in gallery_ann_files
        ]
        self.probe_ann_paths = [
            osp.join(meta_dir, p_ann_file)
            for p_ann_file in probe_ann_files
        ]
        self.pair_ann_path = osp.join(meta_dir, pair_ann_file)
        self.src_landmark = torch.tensor(
                src_landmark, dtype=torch.float32).view(5, 2)
        self.test_mode = test_mode

        self.get_data()
        self.get_label()

    def _parse_landmark_meta(self, path):
        # load 'template_pair_label' annotation file
        with open(path, 'r') as f:
            lines = f.readlines()

        image_paths = []
        landmarks = []
        facenesses = []
        for line in lines:
            terms = line.rstrip().split(' ')
            image_paths.append(terms[0])
            landmark = np.array(
                [float(t) for t in terms[1:-1]], dtype=np.float32)
            landmarks.append(landmark.reshape((5, 2)))
            facenesses.append(float(terms[-1]))

        assert len(image_paths) == len(landmarks) == len(facenesses)
        data_items = []
        for idx in range(len(image_paths)):
            data_item = {
                'path': image_paths[idx],
                'tgz_landmark': landmarks[idx],
                'faceness': facenesses[idx],
            }
            data_items.append(data_item)
    
        return data_items
    
    def _parse_template_media(self, path):
        # load 'tid_mid' annotation file`
        with open(path, 'r') as f:
            lines = f.readlines()

        tmpl_ids = []
        media_ids = []
        for line in lines:
            terms = line.rstrip().split(' ')
            tmpl_ids.append(int(terms[1]))
            media_ids.append(int(terms[2]))
    
        tmpl_items = {}
        for idx, (tmpl_id, media_id) in enumerate(zip(tmpl_ids, media_ids)):
            image_id = idx
            if tmpl_id not in tmpl_items:
                tmpl_items[tmpl_id] = {
                    'image_ids': [],
                    'media_ids': [],
                }
            tmpl_items[tmpl_id]['image_ids'].append(image_id)
            tmpl_items[tmpl_id]['media_ids'].append(media_id)
        
        for idx, tmpl_item in enumerate(tmpl_items.values()):
            tmpl_item['posn_id'] = idx
            media_ids = tmpl_item['media_ids']
            u_media_ids, counts = np.unique(media_ids, return_counts=True)
            media2count = dict(zip(u_media_ids.tolist(), counts.tolist()))
            weights = [1. / media2count[media_id] for media_id in media_ids]
            tmpl_item['weights'] = np.array(weights, dtype=np.float32)
        
        return tmpl_items
    
    def _parse_1n_meta(self, paths):
        tmpl_set = set()
        posn_ids = []
        subj_ids = []
        for path in paths:
            # `load '1N' annotation file`
            with open(path, 'r') as f:
                lines = f.readlines()

            for line in lines[1:]:
                terms = line.rstrip().split(',')
                tmpl_id, subj_id = int(terms[0]), int(terms[1])
                posn_id = self.tmpl_items[tmpl_id]['posn_id']
                if tmpl_id in tmpl_set:
                    continue
                tmpl_set.add(tmpl_id)
                posn_ids.append(posn_id)
                subj_ids.append(subj_id)
    
        return {'posn_ids': posn_ids,
                'subj_ids': subj_ids}

    def _parse_11_meta(self, path):
        # load 'template_pair_label' annotation file
        with open(path, 'r') as f:
            lines = f.readlines()

        posn_ids0 = []
        posn_ids1 = []
        labels = []
        for line in lines:
            terms = line.rstrip().split(' ')
            tmpl_id0, tmpl_id1 = int(terms[0]), int(terms[1])
            posn_ids0.append(self.tmpl_items[tmpl_id0]['posn_id'])
            posn_ids1.append(self.tmpl_items[tmpl_id1]['posn_id'])
            labels.append(int(terms[2]))
    
        return {'posn_ids0': posn_ids0,
                'posn_ids1': posn_ids1,
                'labels': labels}

    def get_data(self):
        # data_items and tmpl_items
        self.data_items = self._parse_landmark_meta(self.data_ann_path)
        self.tmpl_items = self._parse_template_media(self.tmpl_ann_path)

    def get_label(self):
        # annotations for identification and verification
        self.iden_info = {
            'g': self._parse_1n_meta(self.gallery_ann_paths),
            'p': self._parse_1n_meta(self.probe_ann_paths),
        }
        self.veri_info = self._parse_11_meta(self.pair_ann_path)

    def feat2template(self, feats):
        facenesses = torch.tensor(
            [item['faceness'] for item in self.data_items],
            dtype=torch.float32,
        )
        feats = feats * facenesses.view(-1, 1)

        tmpl_feats = torch.zeros(len(self.tmpl_items), feats.size(1))
        for tmpl_item in self.tmpl_items.values():
            idx = tmpl_item['posn_id']
            image_ids = tmpl_item['image_ids']
            weights = torch.tensor(
                tmpl_item['weights'], dtype=torch.float32).view(-1, 1)
            tmpl_feats[idx, :] = torch.mean(
                feats[image_ids] * weights, dim=0)

        return F.normalize(tmpl_feats, dim=1)

    def evaluate_11(self, tmpl_feats,
            FPRs=['1e{}'.format(p) for p in range(-6, 0)]):
        # pair-wise scores
        posn_ids0 = self.veri_info['posn_ids0']
        posn_ids1 = self.veri_info['posn_ids1']
        labels = self.veri_info['labels']

        step = 10000
        scores = []
        for idx in range(0, len(labels), step):
            indices0 = posn_ids0[idx:idx+step]
            indices1 = posn_ids1[idx:idx+step]
            feats0 = tmpl_feats[indices0, :]
            feats1 = tmpl_feats[indices1, :]
            scores.extend(torch.sum(feats0 * feats1, dim=1).tolist())

        # TPR @ FPR
        metrics = get_metrics(labels, scores, FPRs)
        TPRs = [metric for metric in metrics if 'TPR' in metric[0]]

        return TPRs

    def evaluate_1n(self, tmpl_feats, topk=[1, 5, 10],
            FPIRs=['1e{}'.format(p) for p in range(-2, 0)]):
        g_posn_ids = self.iden_info['g']['posn_ids']
        g_subj_ids = self.iden_info['g']['subj_ids']
        p_posn_ids = self.iden_info['p']['posn_ids']
        p_subj_ids = self.iden_info['p']['subj_ids']
        
        g_tmpl_feats = tmpl_feats[g_posn_ids, :]
        p_tmpl_feats = tmpl_feats[p_posn_ids, :]
        g_subj_ids = torch.tensor(g_subj_ids, dtype=torch.int)
        p_subj_ids = torch.tensor(p_subj_ids, dtype=torch.int)

        probe_size = p_subj_ids.size(0)

        # topk
        scores = p_tmpl_feats.mm(g_tmpl_feats.t())
        _, topk_indices = torch.topk(
                scores, max(topk), dim=1, largest=True, sorted=True)
        correct = g_subj_ids[topk_indices].eq(
                p_subj_ids.view(-1, 1).expand_as(topk_indices))

        topk_ACCs = []
        for k in topk:
            header = 'top{}'.format(k)
            correct_k = torch.any(correct[:, :k], dim=1)
            topk_ACC = 100. * torch.sum(correct_k) / probe_size
            topk_ACCs.append((header, topk_ACC.item()))

        # TPIR @ FPIR
        mask = p_subj_ids.view(-1, 1).eq(g_subj_ids.view(1, -1))
        pos_scores = torch.masked_select(scores, mask)
        neg_scores = torch.masked_select(scores, torch.logical_not(mask))
        
        ks = [math.ceil(float(FPIR) * probe_size) for FPIR in FPIRs]
        ths, _ = torch.topk(neg_scores, max(ks), largest=True, sorted=True)
        TPIRs = []
        for k, FPIR in zip(ks, FPIRs):
            header = 'TPIR@FPIR={}'.format(FPIR)
            TPIR = 100. * torch.sum(pos_scores > ths[k-1]) / probe_size
            TPIRs.append((header, TPIR.item()))

        return topk_ACCs, TPIRs

    def evaluate(self, feats):
        tmpl_feats = self.feat2template(feats)
        TPRs = self.evaluate_11(tmpl_feats)
        topk_ACCs, TPIRs = self.evaluate_1n(tmpl_feats)

        return TPRs + topk_ACCs + TPIRs


    def prepare(self, idx):
        # load image and pre-process (pipeline) from path
        path = self.data_items[idx]['path']
        tgz_landmark = self.data_items[idx]['tgz_landmark']
        item = {
            'path': osp.join(self.data_dir, path),
            'src_landmark': self.src_landmark,
            'tgz_landmark': tgz_landmark,
            'crop_size': [112, 112],
        }
        image = image_pipeline(item, self.test_mode)

        return image, idx

    def __len__(self):
        return len(self.data_items)

    def __getitem__(self, idx):
        return self.prepare(idx)
