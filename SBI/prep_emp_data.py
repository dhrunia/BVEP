# %%
import sys
import os
sys.path.append(os.path.join('/', *os.getcwd().split('/')[:-1]))
import lib.preprocess.feature_extract as fx
import numpy as np
import lib.io.stan
# %%
emp_data = fx.prepare_slp_data('../datasets/retro/id020_lma', 
                               'LM_crise3P_100415b-bex_0005.json',
                               'LM_crise3P_100415b-bex_0005.raw.fif',
                               parcellation='destrieux', hpf=10, lpf=0.05)
strct_con = emp_data['SC']
gain_mat = emp_data['gain']
slp = emp_data['slp']
map_estim = lib.io.stan.read_samples(['../results/exp10/exp10.86/id020_lma/samples_LM_crise3P_100415b-bex_0005_hpf10_lpf0.05.csv'])
for key in map_estim.keys():
    map_estim[key] = map_estim[key][0]
# %%
np.savez(os.path.join('../datasets/SBI_Meysam/id020_lma/data.npz'), 
         SC=strct_con, gain=gain_mat, obs_slp=slp, map_K=map_estim['K'],
         map_x_init=map_estim['x_init'], map_z_init=map_estim['z_init'], 
         map_tau=map_estim['tau0'], map_amplitude=map_estim['alpha'],
         map_offset=map_estim['beta'], map_x0=map_estim['x0'],
         map_eps_slp=map_estim['eps_slp'], 
         map_eps_snsr_pwr=map_estim['eps_snsr_pwr'],
         map_pred_slp=map_estim['mu_slp'],
         map_pred_snsr_pwr=map_estim['mu_snsr_pwr'])
# %%
emp_data_sbi = np.load(os.path.join('../datasets/SBI_Meysam/id004_bj/data.npz'))
# %%
