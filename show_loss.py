import yaml
import os.path as osp
from glob import glob

# hfn v0 v1 v2

proj_dirs = [
]

proj_dirs += glob('project/backup/*')
proj_dirs = sorted(proj_dirs)
for proj_dir in proj_dirs:
    config_path = osp.join(proj_dir, 'config.yml')
    with open(config_path, 'r') as f:
        config = yaml.load(f, yaml.SafeLoader)
    bkb_param = config['model']['backbone']
    head_param = config['model']['head']

    print(proj_dir)
    print(bkb_param)
    print(head_param)
    result_path = osp.join(proj_dir, 'Combined.txt')
    with open(result_path, 'r') as f:
        lines = f.readlines()
    print(lines[-1])
    '''
    result_path = osp.join(proj_dir, 'IJB-B.txt')
    with open(result_path, 'r') as f:
        lines = f.readlines()
    print(lines[-1])
    result_path = osp.join(proj_dir, 'IJB-C.txt')
    with open(result_path, 'r') as f:
        lines = f.readlines()
    print(lines[-1])
    '''
    val_path = osp.join(proj_dir, 'val.log')
    with open(val_path, 'r') as f:
        lines = f.readlines()
        vggface2_50 = lines[-1].rstrip().split(', ')[2].split(': ')[-1]
    print(vggface2_50)


    print('--------')




