import os
import os.path as osp
import yaml
import argparse
import torch
import numpy as np

from glob import glob
from face_alignment import align
from opensphere.model import Model

def parse_args():
    parser = argparse.ArgumentParser(
            description='A PyTorch project for face recognition.')
    parser.add_argument('--image_dir', default='./face_alignment/test_images/',
            help='path to the image directory')
    parser.add_argument('--proj_dir',
            help='the directory to load model')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    # get arguments
    args = parse_args()

    # get config of the trained model
    cfg_path = osp.join(args.proj_dir, 'config.yml')
    with open(cfg_path, 'r') as f:
        config = yaml.load(f, yaml.SafeLoader)
    # get model
    model = Model(config['model'])
    last_save = config['trainer']['save_steps'][-1]
    model_path = osp.join(args.proj_dir, 'checkpoint', f'model_{last_save}.pth')
    model_dict = torch.load(model_path)
    model.load_state_dict(model_dict)
    model.eval_mode()

    # inference
    image_paths = sorted(glob(osp.join(args.image_dir, '*.*')))
    features = []
    for image_path in image_paths:
        # get image
        aligned_image = align.get_aligned_face(image_path)
        if aligned_image is None:
            continue
        aligned_image = np.array(aligned_image, dtype=np.float32)
        aligned_image = (aligned_image - 127.5) / 127.5
        aligned_image = torch.tensor(aligned_image.transpose(2, 0, 1)).unsqueeze(0)

        # get feature
        feat = model.get_feature(aligned_image.cuda())
        feat = model.head.f_wrapping(feat)
        fused_feat = model.head.f_fusing(feat)
        features.append(fused_feat)
    features = torch.cat(features, dim=0)

    # calculate score
    score = model.head.f_scoring(features, features, all_pairs=True)
    print(score)
