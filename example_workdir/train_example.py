""" Training models (main)
"""
import os, shutil
import warnings
warnings.filterwarnings("ignore")

cdrp_dir = '/home/zhouyj/software/CERP_Pytorch'
shutil.copyfile('config_example.py', os.path.join(cdrp_dir, 'config.py'))
# train params
gpu_idx = 1
model = ['DetNet','PpkNet'][0]
zarr_path = '/data3/bigdata/zhouyj/RC_train/rc_scsn_win-20s_freq-2-40hz.zarr'
ckpt_dir = ['output/rc_ckpt/DetNet8','output/rc_ckpt/PpkNet6'][0]
resume = [False, True][0]

# start training
if model=='DetNet':
  os.system("python {}/train_cnn.py --gpu_idx={} \
    --zarr_path={} --ckpt_dir={} --resume={} "\
    .format(cdrp_dir, gpu_idx, zarr_path, ckpt_dir, resume))
elif model=='PpkNet':
  os.system("python {}/train_rnn.py --gpu_idx={}\
    --zarr_path={} --ckpt_dir={} --resume={} "\
    .format(cdrp_dir, gpu_idx, zarr_path, ckpt_dir, resume))
else: print('false model name!')

