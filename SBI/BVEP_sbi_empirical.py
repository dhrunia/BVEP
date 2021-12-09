# %%
import os
import sys
sys.path.append(os.path.join('/', *os.getcwd().split('/')[:-1]))
import time
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
import sbi.utils
import matplotlib.pyplot as plt
import lib.preprocess.feature_extract as fx
import torch
import numpy as np
from BVEP_Simulator import VEP2Dmodel
import lib.utils.transform as trnsfrm


from sbi import inference
# from sbi.inference.base import infer


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
emp_data = fx.prepare_slp_data('../datasets/retro/id004_bj', 'BJcrise1le161128B-BEX_0002.json',
                               'BJcrise1le161128B-BEX_0002.raw.fif', parcellation='dk', hpf=10, lpf=0.05)
strct_con = emp_data['SC']
gain_mat = emp_data['gain']
# ds_freq = emp_data['slp'].shape[0]//300
# slp_emp = emp_data['slp'][::ds_freq, :].T

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
obs_summ_stats_emp = fx.comp_summ_stats(
    slp=torch.from_numpy(slp_emp.copy()), nbins=10)

plt.figure(figsize=(3, 6), dpi=150)
plt.imshow(obs_summ_stats_emp.reshape(
    slp_emp.shape[0], 11), aspect='auto', interpolation=None)
plt.colorbar()
plt.title('Summary statistics')
# %%


def VEP2Dmodel_seeg_simulator_wrapper(params):
    # alpha_star = params[-2]
    # beta_star = params[-1]
    # alpha = trnsfrm.inv_bound_norm(alpha_star, lb = 0, ub = 5.0)
    # beta = trnsfrm.inv_bound_norm(beta_star, lb = -5.0, ub = 5.0)
    params = np.asarray(params)

    # fixed parameters
    sigma = 10e-3
    dt = 0.1
    nt = 300
    constants = np.array([sigma, dt, nt])

    src_sig = VEP2Dmodel(params, constants, strct_con)
    # slp = alpha * np.log(np.matmul(gain_mat, np.exp(src_sig))) + beta
    slp = np.log(np.matmul(gain_mat, np.exp(src_sig)))
    summ_stats = torch.as_tensor(fx.comp_summ_stats(slp, 10))

    return summ_stats


def VEP2Dmodel_seeg_simulator_wrapper_np(params):
    # alpha_star = params[-2]
    # beta_star = params[-1]
    # alpha = trnsfrm.inv_bound_norm(alpha_star, lb = 0, ub = 5.0)
    # beta = trnsfrm.inv_bound_norm(beta_star, lb = -5.0, ub = 5.0)
    params = np.asarray(params)

    # fixed parameters
    sigma = 10e-3
    dt = 0.1
    nt = 300
    constants = np.array([sigma, dt, nt])

    src_sig = VEP2Dmodel(params, constants, strct_con)
    slp = np.log(np.matmul(gain_mat, np.exp(src_sig)))
    # slp = alpha * np.log(np.matmul(gain_mat, np.exp(src_sig))) + beta
    summ_stats = torch.as_tensor(
        fx.comp_summ_stats(torch.from_numpy(slp.copy()), 10))

    return summ_stats


# %%
ns, nn = gain_mat.shape
hz_val = -3.65
pz_val = -2.4
ez_val = -1.6

ez_idx = np.array([6, 34],  dtype=np.int32)
pz_wplng_idx = np.array([5, 11], dtype=np.int32)
pz_kplng_idx = np.array([27], dtype=np.int32)
pz_idx = np.append(pz_kplng_idx, pz_wplng_idx)

eta_true = np.ones(nn)*hz_val
eta_true[ez_idx] = ez_val
eta_true[pz_idx] = pz_val
eta_true_star = trnsfrm.bound_norm(eta_true, lb=-5.0, ub=0.0)

K_true = np.array([1])
K_true_star = trnsfrm.bound_norm(K_true, lb=0.0, ub=10.0)

x_init_true = -2.5*np.ones(nn)
x_init_true_star = trnsfrm.bound_norm(x_init_true, lb=-5.0, ub=-1.5)
z_init_true = 3*np.ones(nn)
z_init_true_star = trnsfrm.bound_norm(z_init_true, lb=4.0, ub=6.0)
y_init_true_star = np.concatenate((x_init_true_star, z_init_true_star))

tau_true = np.array([80.0])
tau_true_star = trnsfrm.bound_norm(tau_true, lb=10.0, ub=100.0)
# alpha_true = np.array([1.0])
# alpha_true_star = trnsfrm.bound_norm(alpha_true, lb=0.0, ub=5.0)
# beta_true = np.array([0.0])
# beta_true_star = trnsfrm.bound_norm(beta_true, lb=-5.0, ub=5.0)

params_true = np.concatenate((eta_true_star, y_init_true_star,
                             K_true_star, tau_true_star))

obs_summ_stats_syn = VEP2Dmodel_seeg_simulator_wrapper(params_true)

# %%
plt.figure(figsize=(3, 6), dpi=150)
plt.imshow(obs_summ_stats_syn.reshape(
    ns, 11), aspect='auto', interpolation=None)
plt.colorbar()
plt.title('Summary statistics - Synthetic observations')
# %%
nn = strct_con.shape[0]
# prior_min_eta = -5.0 * np.ones(nn)
# prior_min_xinit = -5.0 * np.ones(nn)
# prior_min_zinit = 3.0 * np.ones(nn)
# prior_min_K = 0.0 * np.ones(1)
# prior_min_tau = 10.0 * np.ones(1)
# prior_min_alpha = 0.0 * np.ones(1)
# prior_min_beta = -5 * np.ones(1)

# prior_max_eta = -1.0 * np.ones(nn)
# prior_max_xinit = -2.0 * np.ones(nn)
# prior_max_zinit = 6.0 * np.ones(nn)
# prior_max_K = 2.0 * np.ones(1)
# prior_max_tau = 100.0 * np.ones(1)
# prior_max_alpha = 5.0 * np.ones(1)
# prior_max_beta = 5.0 * np.ones(1)

# prior_min = np.concatenate((prior_min_eta, prior_min_xinit,
#                             prior_min_zinit, prior_min_K, prior_min_tau,
#                             prior_min_alpha, prior_min_beta))
# prior_max = np.concatenate((prior_max_eta, prior_max_xinit,
#                             prior_max_zinit, prior_max_K, prior_max_tau,
#                             prior_max_alpha, prior_max_beta))
prior_min = np.zeros(3*nn+2)
prior_max = np.ones(3*nn+2)
prior = sbi.utils.torchutils.BoxUniform(
    low=torch.as_tensor(prior_min), high=torch.as_tensor(prior_max))

# %%
simulator, prior = prepare_for_sbi(VEP2Dmodel_seeg_simulator_wrapper, prior)
inference = SNPE(prior)
posteriors = []
proposal = prior
# %%
n_rounds = 10

start_time = time.time()
for _ in range(n_rounds):
    theta, x = simulate_for_sbi(simulator, proposal, 
                                num_simulations=1000, show_progress_bar=True)
    density_estimator = inference.append_simulations(theta, x, proposal=proposal).train()
    # posterior = inference.build_posterior(density_estimator)
    posterior = inference.build_posterior(density_estimator, sample_with='mcmc',
    mcmc_method='nuts')
    # posterior.leakage_correction(obs_summ_stats_syn, num_rejection_samples=1000)
    posteriors.append(posterior)
    proposal = posterior.set_default_x(obs_summ_stats_syn)
# posterior = infer(VEP2Dmodel_seeg_simulator_wrapper, prior,
#                   method='SNPE',  num_simulations=100000, num_workers=4)

print("-"*60)
print("--- %s seconds ---" % (time.time() - start_time))
# %%
num_samples = 200
posterior_samples = posterior.sample(
    (num_samples,), obs_summ_stats_syn, sample_with='mcmc', mcmc_method='nuts').numpy()
# %%
eta_star_pstr_samples = posterior_samples[:, 0:nn]
xinit_star_pstr_samples = posterior_samples[:, nn:2*nn]
zinit_star_pstr_samples = posterior_samples[:, 2*nn:3*nn]
K_star_pstr_samples = posterior_samples[:, 3*nn]
tau_star_pstr_samples = posterior_samples[:, 3*nn + 1]
alpha_star_pstr_samples = posterior_samples[:, 3*nn + 2]
beta_star_pstr_samples = posterior_samples[:, 3*nn + 3]

eta_pstr_samples = trnsfrm.inv_bound_norm(eta_star_pstr_samples, lb = -5.0, ub = 0.0)
tau_pstr_samples = trnsfrm.inv_bound_norm(tau_star_pstr_samples, lb = 10.0, ub = 100.0)
K_pstr_samples = trnsfrm.inv_bound_norm(K_star_pstr_samples, lb = 0.0, ub = 10.0)
xinit_pstr_samples = trnsfrm.inv_bound_norm(xinit_star_pstr_samples, lb = -5.0, ub = -1.5)
zinit_pstr_samples = trnsfrm.inv_bound_norm(zinit_star_pstr_samples, lb = 4.0, ub = 6.0)
y_init_pstr_samples = np.concatenate((xinit_pstr_samples, zinit_pstr_samples), axis=0)
alpha_pstr_samples = trnsfrm.inv_bound_norm(alpha_star_pstr_samples, lb = 0, ub = 5.0)
beta_pstr_samples = trnsfrm.inv_bound_norm(beta_star_pstr_samples, lb = -5.0, ub = 5.0)

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

plt.subplot(143)
plt.violinplot(alpha_pstr_samples)
plt.scatter(1, params_true[3*nn+2], c='red')
plt.ylabel(r'$\alpha$')
plt.title("Amplitude")

plt.subplot(144)
plt.violinplot(beta_pstr_samples)
plt.scatter(1, params_true[3*nn+3], c='red')
plt.ylabel(r'$\beta$')
plt.title("Offset")


# %%
pstr_samples_mean = posterior_samples.mean(axis=0)
pred_summ_stats_syn = VEP2Dmodel_seeg_simulator_wrapper_np(posterior_samples[0,:])

# %%
plt.figure(figsize=(3, 6), dpi=150)
plt.imshow(pred_summ_stats_syn.reshape(
    ns, 11), aspect='auto', interpolation=None)
plt.colorbar()
plt.title('Summary statistics - Predcted synthetic')
# %%
