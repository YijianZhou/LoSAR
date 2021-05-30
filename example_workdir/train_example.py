""" Training models (main)
"""
import os, shutil
import warnings
warnings.filterwarnings("ignore")

cerp_dir = '/home/zhouyj/software/CERP_Pytorch'
shutil.copyfile('config_example.py', os.path.join(cerp_dir, 'config.py'))
# train params
gpu_idx = 1
model = ['DetNet','PpkNet'][0]
zarr_path = 'output/example.zarr'
ckpt_dir = 'output/rc_ckpt/PpkNet'
resume = [False, True][0]

# start training
if model=='DetNet':
  os.system("python {}/train_cnn.py --gpu_idx={} \
    --zarr_path={} --ckpt_dir={} --resume={} "\
    .format(cerp_dir, gpu_idx, zarr_path, ckpt_dir, resume))
elif model=='PpkNet':
  os.system("python {}/train_rnn.py --gpu_idx={}\
    --zarr_path={} --ckpt_dir={} --resume={} "\
    .format(cerp_dir, gpu_idx, zarr_path, ckpt_dir, resume))
else: print('false model name!')

