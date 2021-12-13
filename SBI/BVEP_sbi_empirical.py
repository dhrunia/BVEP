# %%
import os
import sys

from numpy.lib.function_base import interp
sys.path.append(os.path.join('/', *os.getcwd().split('/')[:-1]))
from BVEP_Simulator import VEP2Dmodel
import numpy as np
import torch
import lib.preprocess.feature_extract as fx
import matplotlib.pyplot as plt
import sbi.utils
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
import time
# %%

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Arial']})
# %%
emp_data = fx.prepare_slp_data('../datasets/retro/id004_bj', 
                               'BJcrise1le161128B-BEX_0002.json',
                               'BJcrise1le161128B-BEX_0002.raw.fif',
                               parcellation='dk', hpf=10, lpf=0.05)
strct_con = emp_data['SC']
gain_mat = emp_data['gain']


# fig = plt.figure(figsize=(7, 3), dpi=150, constrained_layout=True)
# gs = fig.add_gridspec(nrows=1, ncols=2)
# ax1 = plt.subplot(gs[0])
# axs_img = ax1.imshow(strct_con)
# ax1.set_xticks(np.r_[0:strct_con.shape[0]:20])
# ax1.set_yticks(np.r_[0:strct_con.shape[0]:20])
# plt.colorbar(axs_img, ax=ax1, use_gridspec=True, fraction=0.05)
# ax1.set_title('Connectome')
# ax2 = plt.subplot(gs[1])
# axs_img = ax2.imshow(gain_mat, aspect='auto')
# plt.colorbar(axs_img, ax=ax2, use_gridspec=True)
# ax2.set_title('Gain Matrix')
# fig = plt.figure(figsize=(6, 5), constrained_layout=True, dpi=150)
# gs = fig.add_gridspec(nrows=1, ncols=1)
# ax1 = plt.subplot(gs[0])
# axs_img = ax1.imshow(slp_emp, aspect='auto')
# plt.colorbar(axs_img, ax=ax1, use_gridspec=True)
# ax1.set_title('SEEG log. power')
# %%


def VEP2Dmodel_seeg_simulator_wrapper(params):
    # alpha = params[-2]
    # beta = params[-1]
    params = np.asarray(params)

    # fixed parameters
    sigma = 0.0
    dt = 0.1
    nt = 300
    constants = np.array([sigma, dt, nt])

    src_sig = VEP2Dmodel(params, constants, strct_con)
    # slp = alpha * np.log(np.matmul(gain_mat, np.exp(src_sig))) + beta
    slp = torch.from_numpy(np.log(np.matmul(gain_mat, np.exp(src_sig))))
    summ_stats = torch.as_tensor(fx.comp_summ_stats(slp, 5))

    return summ_stats

def VEP2Dmodel_seeg_simulator_wrapper_np(params):
    # alpha = params[-2]
    # beta = params[-1]
    params = np.asarray(params)

    # fixed parameters
    sigma = 0.0
    dt = 0.1
    nt = 300
    constants = np.array([sigma, dt, nt])

    src_sig = VEP2Dmodel(params, constants, strct_con)
    slp = np.log(np.matmul(gain_mat, np.exp(src_sig)))
    # slp = alpha * np.log(np.matmul(gain_mat, np.exp(src_sig))) + beta
    summ_stats = torch.as_tensor(
        fx.comp_summ_stats(torch.from_numpy(slp.copy()), 5))

    return summ_stats, src_sig


# %%
ns, nn = gain_mat.shape
hz_val = -3.5
pz_val = -1.9
ez_val = -1.5

ez_idx = np.array([6, 34],  dtype=np.int32)
pz_wplng_idx = np.array([5, 11], dtype=np.int32)
pz_kplng_idx = np.array([27], dtype=np.int32)
pz_idx = np.append(pz_kplng_idx, pz_wplng_idx)

eta_true = np.ones(nn)*hz_val
eta_true[ez_idx] = ez_val
eta_true[pz_idx] = pz_val

K_true = np.array([1.5])

x_init_true = -2.5*np.ones(nn)
z_init_true = 3.2*np.ones(nn)
y_init_true = np.concatenate((x_init_true, z_init_true))

tau_true = np.array([40.0])
# alpha_true = np.array([1.0])
# beta_true = np.array([0.0])

params_true = np.concatenate(
    (eta_true, y_init_true, K_true, tau_true))#, alpha_true, beta_true))

obs_summ_stats_syn, src_sig_true = VEP2Dmodel_seeg_simulator_wrapper_np(params_true)

# %%
plt.figure(figsize=(3, 6), dpi=150)
plt.imshow(obs_summ_stats_syn.reshape(
    ns, 6), aspect='auto', interpolation=None)
plt.colorbar()
plt.title('Summary statistics - Synthetic observations')

plt.figure(figsize=(7, 6), dpi=150)
plt.imshow(src_sig_true, aspect='auto', interpolation=None)
plt.colorbar()
plt.title('Source Signals')
# %%
nn = strct_con.shape[0]
prior_min_eta = -5.0 * np.ones(nn)
prior_min_xinit = -5.0 * np.ones(nn)
prior_min_zinit = 3.0 * np.ones(nn)
prior_min_K = 0.0 * np.ones(1)
prior_min_tau = 20.0 * np.ones(1)
# prior_min_alpha = 0.0 * np.ones(1)
# prior_min_beta = -5.0 * np.ones(1)

prior_max_eta = -1.0 * np.ones(nn)
prior_max_xinit = -2.0 * np.ones(nn)
prior_max_zinit = 6.0 * np.ones(nn)
prior_max_K = 10.0 * np.ones(1)
prior_max_tau = 100.0 * np.ones(1)
# prior_max_alpha = 5.0 * np.ones(1)
# prior_max_beta = 5.0 * np.ones(1)

prior_min = np.concatenate((prior_min_eta, prior_min_xinit,
                            prior_min_zinit, prior_min_K, prior_min_tau))
prior_max = np.concatenate((prior_max_eta, prior_max_xinit,
                            prior_max_zinit, prior_max_K, prior_max_tau))
prior = sbi.utils.torchutils.BoxUniform(
    low=torch.as_tensor(prior_min), high=torch.as_tensor(prior_max))

# %%
simulator, prior = prepare_for_sbi(VEP2Dmodel_seeg_simulator_wrapper, prior)
inference = SNPE(prior)
posteriors = []
proposal = prior
# %%
n_rounds = 1

start_time = time.time()
for _ in range(n_rounds):
    theta, x = simulate_for_sbi(simulator, proposal, 
                                num_simulations=50000, show_progress_bar=True)
    density_estimator = inference.append_simulations(theta, x, proposal=prior).train()
    posterior = inference.build_posterior(density_estimator)
    # posterior = inference.build_posterior(density_estimator, sample_with='mcmc',
    # mcmc_method='nuts', mcmc_parameters={'warmup_steps':1000, 'num_chains':1,
    # 'thin':1})
    posteriors.append(posterior)
    # proposal = posterior.set_default_x(obs_summ_stats_syn)

print("-"*60)
print("--- %s seconds ---" % (time.time() - start_time))
# %%
num_samples = 1000
posterior_samples = posterior.sample(
    (num_samples,), obs_summ_stats_syn, 
    sample_with='mcmc', mcmc_method='nuts',
    mcmc_parameters={'warmup_steps':1000, 'thin':1, 'num_chains':1}).numpy()
# %%
eta_pstr_samples = posterior_samples[:, 0:nn]
xinit_pstr_samples = posterior_samples[:, nn:2*nn]
zinit_pstr_samples = posterior_samples[:, 2*nn:3*nn]
K_pstr_samples = posterior_samples[:, 3*nn]
tau_pstr_samples = posterior_samples[:, 3*nn + 1]
# alpha_pstr_samples = posterior_samples[:, 3*nn + 2]
# beta_pstr_samples = posterior_samples[:, 3*nn + 3]

# y_init_pstr_samples = np.concatenate((xinit_pstr_samples, zinit_pstr_samples), axis=0)

# %%
plt.figure(figsize=(7, 7), dpi=150, constrained_layout=True)

plt.subplot(131)
plt.violinplot(eta_pstr_samples, positions=np.r_[1:nn+1], vert=False)
plt.yticks(np.r_[1:nn+1:3])
plt.xlabel(r'$\eta$')
plt.ylabel('Region')
plt.scatter(params_true[0:nn], np.r_[1:nn+1], s=10, c='red')

plt.subplot(132)
plt.violinplot(xinit_pstr_samples, positions=np.r_[1:nn+1], vert=False)
plt.yticks(np.r_[1:nn+1:3])
plt.xlabel(r'$x(t=0)$')
plt.ylabel('Region')
# plt.scatter(params_true[0:nn], np.r_[1:nn+1], s=10, c='red')

plt.subplot(133)
plt.violinplot(zinit_pstr_samples, positions=np.r_[1:nn+1], vert=False)
plt.yticks(np.r_[1:nn+1:3])
plt.xlabel(r'$z(t=0)$')
plt.ylabel('Region')
# plt.scatter(params_true[0:nn], np.r_[1:nn+1], s=10, c='red')

plt.figure(figsize=(7, 2), dpi=150, constrained_layout=True)
plt.subplot(141)
plt.violinplot(K_pstr_samples)
plt.scatter(1, params_true[3*nn], c='red')
plt.ylabel(r'$K$')
plt.title("Global scaling factor")

plt.subplot(142)
plt.violinplot(tau_pstr_samples)
plt.scatter(1, params_true[3*nn+1], c='red')
plt.ylabel(r'$\tau$')
plt.title("Time scaling")

# plt.subplot(143)
# plt.violinplot(alpha_pstr_samples)
# plt.scatter(1, params_true[3*nn+2], c='red')
# plt.ylabel(r'$\alpha$')
# plt.title("Amplitude")

# plt.subplot(144)
# plt.violinplot(beta_pstr_samples)
# plt.scatter(1, params_true[3*nn+3], c='red')
# plt.ylabel(r'$\beta$')
# plt.title("Offset")


# %%
pred_summ_stats_syn = np.zeros((1000, ns, 6))
pred_src_sig = np.zeros((1000, nn, 300))
for i, sample in enumerate(posterior_samples):
# pstr_samples_mean = posterior_samples.mean(axis=0)
    summ_stats, src_sig = VEP2Dmodel_seeg_simulator_wrapper_np(posterior_samples[i])
    pred_summ_stats_syn[i] = summ_stats.numpy().reshape(ns, 6)
    pred_src_sig[i] = src_sig
pred_summ_stats_syn_mean = pred_summ_stats_syn.mean(axis=0)
src_sig_mean = pred_src_sig.mean(axis=0)

# %%
plt.figure(figsize=(7, 6), dpi=150)
plt.subplot(121)
plt.imshow(obs_summ_stats_syn.reshape(
    ns, 6), aspect='auto', interpolation=None)
plt.colorbar()
plt.title('Summary statistics - Observed synthetic')

plt.subplot(122)
plt.imshow(pred_summ_stats_syn_mean, aspect='auto', interpolation=None)
plt.colorbar()
plt.title('Summary statistics - Predcted synthetic')

plt.figure(figsize=(7, 6), dpi=150)
plt.subplot(121)
plt.imshow(src_sig_true, aspect='auto', interpolation=None)
plt.colorbar()
plt.title('True Source Signals')

plt.subplot(122)
plt.imshow(src_sig_mean, aspect='auto', interpolation=None)
plt.colorbar()
plt.title('Predicted Source Signals')
# %%
# ds_freq = emp_data['slp'].shape[0]//300
# slp_emp = emp_data['slp'][::ds_freq, :].T
obs_summ_stats_emp = fx.comp_summ_stats(
    slp=torch.from_numpy(emp_data['slp'].copy().T), nbins=5)

plt.figure(figsize=(3, 6), dpi=150)
plt.imshow(obs_summ_stats_emp.reshape(
    ns, 6), aspect='auto', interpolation=None)
plt.colorbar()
plt.title('Summary statistics - empirical')
# %%
num_samples = 1000
posterior_samples = posterior.sample(
    (num_samples,), obs_summ_stats_emp, 
    sample_with='mcmc', mcmc_method='nuts',
    mcmc_parameters={'warmup_steps':1000, 'thin':1, 'num_chains':1}).numpy()
# %%
# %%
eta_pstr_samples = posterior_samples[:, 0:nn]
xinit_pstr_samples = posterior_samples[:, nn:2*nn]
zinit_pstr_samples = posterior_samples[:, 2*nn:3*nn]
K_pstr_samples = posterior_samples[:, 3*nn]
tau_pstr_samples = posterior_samples[:, 3*nn + 1]
# alpha_pstr_samples = posterior_samples[:, 3*nn + 2]
# beta_pstr_samples = posterior_samples[:, 3*nn + 3]

# y_init_pstr_samples = np.concatenate((xinit_pstr_samples, zinit_pstr_samples), axis=0)

# %%
plt.figure(figsize=(7, 7), dpi=150, constrained_layout=True)

plt.subplot(131)
plt.violinplot(eta_pstr_samples, positions=np.r_[1:nn+1], vert=False)
plt.yticks(np.r_[1:nn+1:3])
plt.xlabel(r'$\eta$')
plt.ylabel('Region')

plt.subplot(132)
plt.violinplot(xinit_pstr_samples, positions=np.r_[1:nn+1], vert=False)
plt.yticks(np.r_[1:nn+1:3])
plt.xlabel(r'$x(t=0)$')
plt.ylabel('Region')

plt.subplot(133)
plt.violinplot(zinit_pstr_samples, positions=np.r_[1:nn+1], vert=False)
plt.yticks(np.r_[1:nn+1:3])
plt.xlabel(r'$z(t=0)$')
plt.ylabel('Region')

plt.figure(figsize=(7, 2), dpi=150, constrained_layout=True)
plt.subplot(141)
plt.violinplot(K_pstr_samples)
plt.ylabel(r'$K$')
plt.title("Global scaling factor")

plt.subplot(142)
plt.violinplot(tau_pstr_samples)
plt.ylabel(r'$\tau$')
plt.title("Time scaling")

# plt.subplot(143)
# plt.violinplot(alpha_pstr_samples)
# plt.scatter(1, params_true[3*nn+2], c='red')
# plt.ylabel(r'$\alpha$')
# plt.title("Amplitude")

# plt.subplot(144)
# plt.violinplot(beta_pstr_samples)
# plt.scatter(1, params_true[3*nn+3], c='red')
# plt.ylabel(r'$\beta$')
# plt.title("Offset")


# %%
pred_summ_stats_syn = np.zeros((1000, ns, 6))
pred_src_sig = np.zeros((1000, nn, 300))
for i, sample in enumerate(posterior_samples):
    summ_stats, src_sig = VEP2Dmodel_seeg_simulator_wrapper_np(posterior_samples[i])
    pred_summ_stats_syn[i] = summ_stats.numpy().reshape(ns, 6)
    pred_src_sig[i] = src_sig
pred_summ_stats_syn_mean = pred_summ_stats_syn.mean(axis=0)
pred_src_sig_mean = pred_src_sig.mean(axis=0)

# %%
plt.figure(figsize=(7, 6), dpi=150)
plt.subplot(121)
plt.imshow(obs_summ_stats_emp.reshape(
    ns, 6), aspect='auto', interpolation=None)
plt.colorbar()
plt.title('Summary statistics - Observed empirical')

plt.subplot(122)
plt.imshow(pred_summ_stats_syn_mean, aspect='auto', interpolation=None)
plt.colorbar()
plt.title('Summary statistics - Predcted empirical')

plt.figure(figsize=(4, 6), dpi=150)
# plt.subplot(121)
# plt.imshow(src_sig_true, aspect='auto', interpolation=None)
# plt.colorbar()
# plt.title('True Source Signals')

plt.imshow(pred_src_sig_mean, aspect='auto', interpolation=None)
plt.colorbar()
plt.title('Predicted Source Signals empirical')
# %%
