""" Make HDF5 dataset with SAC files
"""
import os, shutil

# i/o paths
out_path = '/data/bigdata/eg_train-samples.h5'
sac_root = '/data/bigdata/eg_train-samples_sac'
rsel_prep_dir = '/home/zhouyj/software/RSeL_TED/preprocess'
shutil.copyfile('config_eg.py', os.path.join(rsel_prep_dir, 'config.py'))
num_workers = 10

# sac2zarr
os.system("python {}/sac2hdf5.py --out_path={} --sac_root={} --num_workers={}"\
    .format(rsel_prep_dir, out_path, sac_root, num_workers))
