""" Make Zarr dataset with SAC files
"""
import os, shutil

# i/o paths
out_path = '/data/bigdata/eg_train-samples.zarr'
sac_root = '/data/bigdata/eg_train-samples_sac'
sar_prep_dir = '/home/zhouyj/software/LoSAR/preprocess'
shutil.copyfile('config_eg.py', os.path.join(sar_prep_dir, 'config.py'))
num_workers = 10

# sac2zarr
os.system("python {}/sac2zarr.py --out_path={} --sac_root={} --num_workers={}"\
    .format(sar_prep_dir, out_path, sac_root, num_workers))
