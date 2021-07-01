#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 16:02:40 2020

@author: jlee
"""


import time
start_time = time.time()

import numpy as np
import pandas as pd
import glob, os, copy
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

from build_obj import build_obs
from build_obj import build_model
from build_obj import build_sps
from build_obj import build_noise

from prospect.fitting import lnprobfn
from prospect.fitting import fit_model
from prospect.io import write_results as writer
from prospect.io import read_results as reader
from prospect.plotting.corner import quantile

from astropy.cosmology import FlatLambdaCDM


run_params = {}
try_id = "Solar3"

if (glob.glob("./"+str(try_id)) == []):
    os.system("rm -rfv ./"+str(try_id))
    os.system("mkdir ./"+str(try_id))

# emcee    
nwalkers, niter, ndiscard = 64, 5000, 3000

# dynesty
nlive, dlogz = [2000, 1000], 0.05


# ----- Observation data ----- #
c = 2.99792e+10  # cm/s

# Filter names
galex = ["galex_FUV", "galex_NUV"]
hst_acs = ["acs_wfc_f435w", "acs_wfc_f606w", "acs_wfc_f814w"]
hst_wfc3_ir = ["wfc3_ir_f110w", "wfc3_ir_f140w"]
spitzer = ["spitzer_irac_ch1", "spitzer_irac_ch2"]

filternames = galex + hst_acs + hst_wfc3_ir + spitzer

# Object information
m_AB, e_m_AB = np.loadtxt("phot_results.txt").T
e_m_AB = np.maximum(e_m_AB, 0.1)


## Cosmological parameters
redshift = 0.3527
cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)
ldist = cosmo.luminosity_distance(redshift).value  # Mpc


# ----- Building the obs dictionary ----- #
obs  = build_obs(filternames, m_AB, ldist,
                 [True]*len(filternames),
                 e_mags=e_m_AB, snr=20.0)


# ----- Building the model dictionary ----- #
fx_metal = 0.0
model = build_model(object_redshift=redshift, luminosity_distance=ldist,
                    fixed_metallicity=fx_metal, fixed_dust2=2.39,
                    add_duste=True, add_neb=True)


# ----- Building the sps dictionary ----- #
sps = build_sps(zcontinuous=1)


# ----- View the model ----- #
a = 1.0 + model.params.get("zred") # cosmological redshifting

# spectroscopic wavelengths
if obs["wavelength"] is None:
    wspec = sps.wavelengths
    wspec *= a
else:
    wspec = obs["wavelength"]
wspec *= 1.0e-4    # Angstrom to micron
wphot = obs["phot_wave"]*1.0e-4    # micro meter [10**(-6) meter]

xmin, xmax = np.min(wphot)*0.3, np.max(wphot)/0.3
ftemp = 1.0e+4*c/np.linspace(xmin,xmax,10000)
fphot, fspec = 1.0e+4*c/wphot, 1.0e+4*c/wspec


# ----- Function for plotting figure ----- #
def plot_sed(spec_data, phot_data, theta, out):

    # Establish bounds
    temp = np.interp(np.linspace(xmin,xmax,10000), wspec, spec_data)
    ymin, ymax = (ftemp*temp).min()*3631.*1.0e-23*0.2, (ftemp*temp).max()*3631.*1.0e-23/0.2

    # Figure setting
    fig = plt.figure(1, figsize=(18,10))
    gs = GridSpec(1, 1, left=0.11, bottom=0.13, right=0.81, top=0.95)
    ax = fig.add_subplot(gs[0,0])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.tick_params(axis="both", labelsize=25.0, pad=8.0)
    ax.set_xticks([1.0e-1, 5.0e-1, 1.0e+0, 5.0e+0, 1.0e+1])
    ax.set_xticklabels(["0.1", "0.5", "1", "5", "10"])
    ax.set_xlabel(r"Wavelength ${\rm [\mu m]}$ (Observer-frame)", fontsize=25.0, labelpad=10.0)
    ax.set_ylabel(r"$\nu F_{\nu}~{\rm [erg~s^{-1}~cm^{-2}]}$", fontsize=25.0, labelpad=10.0)
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.tick_params(width=2.0, length=12.0)
    ax.tick_params(width=2.0,length=7.5,which="minor")
    for axis in ["top","bottom","left","right"]:
        ax.spines[axis].set_linewidth(2.0)

    # Plotting model + data
    ax.plot(wspec, fspec*spec_data*3631.*1.0e-23,
            label="Model spectrum", lw=1.5, color="navy", alpha=0.5)
    ax.errorbar(wphot, fphot*phot_data*3631.*1.0e-23,
                label="Model photometry", 
                marker="s", markersize=10, alpha=0.8, ls="", lw=3,
                markerfacecolor="none", markeredgecolor="blue", 
                markeredgewidth=3)
    ax.errorbar(wphot, fphot*obs["maggies"]*3631.*1.0e-23, 
                yerr=fphot*obs["maggies_unc"]*3631.*1.0e-23, 
                label="Observed photometry",
                marker="o", markersize=10, alpha=0.8, ls="", lw=3,
                ecolor="tomato", markerfacecolor="none", markeredgecolor="tomato", 
                markeredgewidth=3)    

    # Plotting filters
    for f in obs["filters"]:
        w, t = f.wavelength.copy(), f.transmission.copy()
        t = t / t.max()
        t = 10**(0.2*(np.log10(ymax/ymin)))*t * ymin
        ax.plot(1.0e-4*w, t, lw=3, color="gray", alpha=0.7)

    # Figure texts
    ax.text(1.01, 0.95, "Solutions", fontsize=20.0, fontweight="bold", color="black",
            ha="left", va="top", transform=ax.transAxes)
    ax.text(1.02, 0.75, r"${\rm log} (M_{{\rm star}}/M_{\odot})=%.3f$" \
            %(np.log10(theta[0])), fontsize=20.0, color="blue",
            ha="left", va="top", transform=ax.transAxes)
    ax.text(1.02, 0.70, r"$[Z/Z_{\odot}]=%.2f$ (fixed)" \
            %(fx_metal), fontsize=20.0, color="black",
            ha="left", va="top", transform=ax.transAxes)
    ax.text(1.02, 0.65, r"$\^{\tau}_{2}=%.3f$" \
            %(theta[1]), fontsize=20.0, color="blue",
            ha="left", va="top", transform=ax.transAxes)
    ax.text(1.02, 0.60, r"$t_{\rm age}=%.2f$ Gyr" \
            %(theta[2]), fontsize=20.0, color="blue",
            ha="left", va="top", transform=ax.transAxes)
    ax.text(1.02, 0.55, r"$\tau=%.2f$ Gyr" \
            %(theta[3]), fontsize=20.0, color="blue",
            ha="left", va="top", transform=ax.transAxes)
    chisq = np.sum(((obs["maggies"]-phot_data)/obs["maggies_unc"])**2.0)
    dof = np.sum(obs['phot_mask']) - len(model.theta)
    rchisq = chisq / dof
    ax.text(1.02, 0.30, r"$\chi^{2}=%.3f$" \
            %(chisq), fontsize=20.0, color="red",
            ha="left", va="top", transform=ax.transAxes)
    ax.text(1.02, 0.25, r"$\chi_{v}^{2}=%.3f$" \
            %(rchisq), fontsize=20.0, color="red",
            ha="left", va="top", transform=ax.transAxes)
    ax.legend(loc="best", fontsize=20)

    plt.savefig(str(try_id)+"/"+out+".pdf", dpi=300)
    plt.savefig(str(try_id)+"/"+out+".png", dpi=300)
    plt.close()


np.random.seed(0)


# ##############################################################
# # ----- Solving #1: Minimization (Levenberg-Marquardt) ----- #
# ##############################################################
# run_params["dynesty"] = False
# run_params["emcee"] = False
# run_params["optimize"] = True
# run_params["min_method"] = "lm"
# run_params["nmin"] = 2

# output = fit_model(obs, model, sps, lnprobfn=lnprobfn, **run_params)
# print("Done optmization in {0:.2f}s".format(output["optimization"][1]))

# (results, topt) = output["optimization"]
# ind_best = np.argmin([r.cost for r in results])
# res = results[ind_best]
# theta_lm = res.x.copy()

# # Levenberg-Marquardt uncertainty
# J = res.jac.copy()
# cov0 = np.linalg.inv(J.T.dot(J))
# dof = np.sum(obs['phot_mask']) - len(model.theta)
# chi2 = 2*res.cost.copy()
# s_sq = chi2/dof
# pcov = cov0*s_sq
# e_theta_lm = np.sqrt(np.diag(pcov))
# min_error = np.array([1.0e+6, 0.05, 0.5, 0.5])
# for i in np.arange(len(e_theta_lm)):
#     e_theta_lm[i] = np.maximum(e_theta_lm[i], min_error[i])

# pspec, pphot, pfrac = model.sed(theta_lm, obs=obs, sps=sps)


# # ----- Figure 1 setting ----- #
# plot_sed(pspec, pphot, theta_lm, "sed1_lm")


# ####################################################
# # ----- Solving #2: Ensemble sampling (MCMC) ----- #
# ####################################################
# run_params["dynesty"] = False
# run_params["emcee"] = True
# run_params["nwalkers"] = nwalkers
# run_params["niter"] = niter
# run_params["nburn"] = []
# run_params["optimize"] = False
# run_params["min_method"] = "lm"
# run_params["nmin"] = 2
# ndiscard = ndiscard

# output = fit_model(obs, model, sps, lnprobfn=lnprobfn, **run_params)
# print('done emcee in {0:.2f}s'.format(output["sampling"][1]))


# # ----- Writing & reading output ----- #
# hfile = str(try_id)+"/output_mcmc.h5"
# os.system("rm -rfv "+hfile)
# writer.write_hdf5(hfile, run_params, model, obs,
#                   output["sampling"][0], output["optimization"][0],
#                   tsample=output["sampling"][1],
#                   toptimize=output["optimization"][1])
# res_mcmc, obs_mcmc, mod_mcmc = reader.results_from(hfile, dangerous=False)

# # ----- Finding maximum posterior ----- #
# samp_, lnp_ = res_mcmc["chain"][:, ndiscard:, :], res_mcmc["lnprobability"][:, ndiscard:]
# imax = np.argmax(lnp_)
# i, j = np.unravel_index(imax, lnp_.shape)
# theta_max = samp_[i, j, :].copy()

# # ----- Finding median values ----- #
# weights = res_mcmc.get("weights", None)
# if weights is not None:
#     weights = weights[ndiscard:]
# post_pcts = quantile(samp_.T, q=[0.50-0.3413, 0.50, 0.50+0.3413], weights=weights)
# theta_med = post_pcts[:,1]
# eu_theta_med = post_pcts[:,2] - post_pcts[:,1]
# el_theta_med = post_pcts[:,1] - post_pcts[:,0]

# #print("Optimization value: {}".format(theta_lm))
# print("")
# print("Max lnP value (emcee): {}".format(theta_max))
# print("Median value (emcee): {}".format(theta_med))

# theta_emcee = [theta_max, theta_med, eu_theta_med, el_theta_med]


# # ----- Trace & corner plot ----- #
# plt.rcParams.update({'font.size': 25})

# chosen = np.random.choice(res_mcmc["run_params"]["nwalkers"], size=10, replace=False)
# tracefig = reader.traceplot(res_mcmc, figsize=(20,10), chains=chosen, start=ndiscard)
# tracefig.savefig(str(try_id)+"/trace_emcee.png", dpi=300)
# plt.close()

# cornerfig = reader.subcorner(res_mcmc, start=ndiscard, thin=5, truths=theta_max, 
#                              fig=plt.subplots(len(model.theta), len(model.theta),
#                                               figsize=(25,25))[0])
# cornerfig.savefig(str(try_id)+"/corner_emcee.png", dpi=300)
# plt.close()


# # ----- Figure 2 setting (max lnP) ----- #
# mspec, mphot, mfrac = model.sed(theta_max, obs=obs, sps=sps)
# plot_sed(mspec, mphot, theta_max, "sed2_lnpmax")

# # ----- Figure 2 setting (median) ----- #
# bspec, bphot, bfrac = model.sed(theta_med, obs=obs, sps=sps)
# plot_sed(bspec, bphot, theta_med, "sed2_median")


###########################################
# ----- Solving #3: Nested sampling ----- #
###########################################
run_params["dynesty"] = True
run_params["optmization"] = False
run_params["emcee"] = False
run_params["nested_method"] = "rwalk"
run_params["nlive_init"] = nlive[0]
run_params["nlive_batch"] = nlive[1]
run_params["nested_dlogz_init"] = dlogz
run_params["nested_posterior_thresh"] = dlogz
run_params["nested_maxcall"] = int(1e7)

output = fit_model(obs, model, sps, lnprobfn=lnprobfn, **run_params)
print('done dynesty in {0:.2f}s'.format(output["sampling"][1]))


# ----- Writing & reading output ----- #
hfile = str(try_id)+"/output_nest.h5"
os.system("rm -rfv "+hfile)
writer.write_hdf5(hfile, run_params, model, obs,
                  output["sampling"][0], output["optimization"][0],
                  tsample=output["sampling"][1],
                  toptimize=output["optimization"][1])
res_nest, obs_nest, mod_nest = reader.results_from(hfile, dangerous=False)

# ----- Finding maximum posterior ----- #
samp_, lnp_ = res_nest["chain"][ndiscard:, :], res_nest["lnprobability"][ndiscard:]
imax = np.argmax(lnp_)
theta_max = samp_[imax, :].copy()

# ----- Finding median values ----- #
weights = res_nest.get("weights", None)
if weights is not None:
    weights = weights[ndiscard:]
post_pcts = quantile(samp_.T, q=[0.50-0.3413, 0.50, 0.50+0.3413], weights=weights)
theta_med = post_pcts[:,1]
eu_theta_med = post_pcts[:,2] - post_pcts[:,1]
el_theta_med = post_pcts[:,1] - post_pcts[:,0]

#print("Optimization value: {}".format(theta_lm))
print("")
print("Max lnP value (dynesty): {}".format(theta_max))
print("Median value (dynesty): {}".format(theta_med))

theta_dynesty = [theta_max, theta_med, eu_theta_med, el_theta_med]


# ----- Trace & corner plot ----- #
ndiscard = res_nest['niter'][0] // 4
plt.rcParams.update({'font.size': 25})
tracefig = reader.traceplot(res_nest, start=ndiscard, figsize=(20,10))
tracefig.savefig(str(try_id)+"/trace_dynesty.png", dpi=300)
plt.close()

cornerfig = reader.subcorner(res_nest, start=ndiscard, thin=5, 
                             fig=plt.subplots(len(model.theta), len(model.theta),
                                              figsize=(25,25))[0])
cornerfig.savefig(str(try_id)+"/corner_dynesty.png", dpi=300)
plt.close()


# ----- Figure 3 setting (max lnP) ----- #
mspec, mphot, mfrac = model.sed(theta_max, obs=obs, sps=sps)
plot_sed(mspec, mphot, theta_max, "sed3_lnpmax")

# ----- Figure 3 setting (median) ----- #
bspec, bphot, bfrac = model.sed(theta_med, obs=obs, sps=sps)
plot_sed(bspec, bphot, theta_med, "sed3_median")


# Printing the running time
print("--- {0:.2f} seconds ---".format(time.time()-start_time))
