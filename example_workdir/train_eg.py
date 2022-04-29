""" Training models (main)
"""
import os, shutil
import warnings
warnings.filterwarnings("ignore")

cerp_dir = '/home/zhouyj/software/CERP_Pytorch'
shutil.copyfile('config_eg.py', os.path.join(cerp_dir, 'config.py'))
# train params
gpu_idx = 1
model = ['EventNet','PhaseNet'][0]
zarr_path = '/data/bigdata/eg_train-samples.zarr'
ckpt_dir = 'output/eg_ckpt'

# start training
if model=='EventNet':
  os.system("python {}/train_cnn.py --gpu_idx={} \
    --zarr_path={} --ckpt_dir={}/{} "\
    .format(cerp_dir, gpu_idx, zarr_path, ckpt_dir, model))
elif model=='PhaseNet':
  os.system("python {}/train_rnn.py --gpu_idx={}\
    --zarr_path={} --ckpt_dir={}/{} "\
    .format(cerp_dir, gpu_idx, zarr_path, ckpt_dir, model))
else: print('false model name!')

