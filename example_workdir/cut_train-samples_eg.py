""" Cut training samples in SAC
"""
import os, shutil

# i/o paths
data_dir = '/data/Example_data'
cerp_prep_dir = '/home/zhouyj/software/CERP_Pytorch/preprocess'
shutil.copyfile('config_eg.py', os.path.join(cerp_prep_dir, 'config.py'))
fpha = 'input/example.pha'
out_root = '/data/bigdata/eg_train-samples_sac'
cut_method = 'obspy'

# cut pos & neg
os.system("python {}/cut_pos_{}.py --data_dir={} --fpha={} --out_root={} "\
    .format(cerp_prep_dir, cut_method, data_dir, fpha, out_root))
os.system("python {}/cut_neg_{}.py --data_dir={} --fpha={} --out_root={} "\
    .format(cerp_prep_dir, cut_method, data_dir, fpha, out_root))
