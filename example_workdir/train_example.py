""" Training models (main)
"""
import os, shutil
import warnings
warnings.filterwarnings("ignore")

cerp_dir = '/home/zhouyj/software/CERP_Pytorch'
shutil.copyfile('config_example.py', os.path.join(cerp_dir, 'config.py'))
# train params
gpu_idx = 0
model = ['EventNet','PhaseNet'][1]
zarr_path = 'output/example.zarr'
ckpt_dir = ['output/example_ckpt/EventNet','output/example_ckpt/PhaseNet'][1]

# start training
if model=='EventNet':
  os.system("python {}/train_cnn.py --gpu_idx={} \
    --zarr_path={} --ckpt_dir={} "\
    .format(cerp_dir, gpu_idx, zarr_path, ckpt_dir))
elif model=='PhaseNet':
  os.system("python {}/train_rnn.py --gpu_idx={}\
    --zarr_path={} --ckpt_dir={} "\
    .format(cerp_dir, gpu_idx, zarr_path, ckpt_dir))
else: print('false model name!')

