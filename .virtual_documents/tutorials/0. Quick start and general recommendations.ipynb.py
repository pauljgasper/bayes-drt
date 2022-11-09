import numpy as np
import pandas as pd
import os
import sys
import time
import matplotlib.pyplot as plt
get_ipython().run_line_magic("matplotlib", " notebook")
import glob

from bayes_drt.inversion import Inverter
import bayes_drt.file_load as fl
import bayes_drt.plotting as bp

get_ipython().run_line_magic("load_ext", " autoreload")
get_ipython().run_line_magic("autoreload", " 2")

"""
NOTE: the first time you import bayes_drt.inversion, several models will be compiled, which will take a significant
amount of time (typically ~20 minutes). Once compiled, the model files will be stored with the package,
such that this step will only be necessary the first time you import the package.
"""


# Set plot formatting and data directory
datadir = '../data'

tick_size = 9
label_size = 11

plt.rcParams['font.family'] = 'serif'
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams['xtick.labelsize'] = tick_size
plt.rcParams['ytick.labelsize'] = tick_size
plt.rcParams['axes.labelsize'] = label_size
plt.rcParams['legend.fontsize'] = tick_size - 1


"Load data"
# load simulated impedance data with noise
Z_file = os.path.join(datadir,'simulated','Z_RC-ZARC_Macdonald_0.25.csv')
Zdf = pd.read_csv(Z_file)

# load true DRT
g_file = os.path.join(datadir,'simulated','gamma_RC-ZARC.csv')
g_true = pd.read_csv(g_file)

# extract frequency and complex impedance
freq, Z = fl.get_fZ(Zdf)

# Plot the data
axes = bp.plot_eis(Zdf)


"Fit the data"
# By default, the Inverter class is configured to fit the DRT (rather than the DDT)
# Create separate Inverter instances for HMC and MAP fits
# Set the basis frequencies equal to the measurement frequencies 
# (not necessary in general, but yields faster results here - see Tutorial 1 for more info on basis_freq)
inv_hmc = Inverter(basis_freq=freq)
inv_map = Inverter(basis_freq=freq)

# Perform HMC fit
start = time.time()
inv_hmc.fit(freq, Z, mode='sample')
elapsed = time.time() - start
print('HMC fit time {:.1f} s'.format(elapsed))

# Perform MAP fit
start = time.time()
inv_map.fit(freq, Z, mode='optimize')  # initialize from ridge solution
elapsed = time.time() - start
print('MAP fit time {:.1f} s'.format(elapsed))


"Visualize DRT and impedance fit"
# plot impedance fit and recovered DRT
fig,axes = plt.subplots(1, 2, figsize=(8, 3.5))

# plot fits of impedance data
inv_hmc.plot_fit(axes=axes[0], plot_type='nyquist', color='k', label='HMC fit', data_label='Data')
inv_map.plot_fit(axes=axes[0], plot_type='nyquist', color='r', label='MAP fit', plot_data=False)

# plot true DRT
p = axes[1].plot(g_true['tau'],g_true['gamma'],label='True',ls='--')
# add Dirac delta function for RC element
axes[1].plot([np.exp(-2),np.exp(-2)],[0,10],ls='--',c=p[0].get_color(),lw=1)

# Plot recovered DRT at given tau values
tau_plot = g_true['tau'].values
inv_hmc.plot_distribution(ax=axes[1], tau_plot=tau_plot, color='k', label='HMC mean', ci_label='HMC 95% CI')
inv_map.plot_distribution(ax=axes[1], tau_plot=tau_plot, color='r', label='MAP')

axes[1].set_ylim(0,3.5)
axes[1].legend()


fig.tight_layout()


"Visualize the recovered error structure"
# For visual clarity, only MAP results are shown.
# HMC results can be obtained in the same way
fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharex=True)

# plot residuals and estimated error structure
inv_map.plot_residuals(axes=axes)

# plot true error structure in miliohms
p = axes[0].plot(freq, 3*Zdf['sigma_re'] * 1000, ls='--')
axes[0].plot(freq, -3*Zdf['sigma_re'] * 1000, ls='--', c=p[0].get_color())
axes[1].plot(freq, 3*Zdf['sigma_im'] * 1000, ls='--')
axes[1].plot(freq, -3*Zdf['sigma_im'] * 1000, ls='--', c=p[0].get_color(), label='True $\pm 3\sigma$')

axes[1].legend()

fig.tight_layout()


inv_hmc.plot_full_results()


"Get the polarization resistance, ohmic resistance, and inductance"
# Results are shown for HMC only, but can be obtained for the MAP fit in the same way.
# The only difference is that the MAP fit does not provide percentile prediction.
print('R_inf: {:.4f} ohms'.format(inv_hmc.R_inf))
print('Inductance: {:.4e} H'.format(inv_hmc.inductance))
print('Polarization resistance: {:.4f} ohms'.format(inv_hmc.predict_Rp()))
print('Rp 2.5 percentile: {:.4f} ohms'.format(inv_hmc.predict_Rp(percentile=2.5)))
print('Rp 97.5 percentile: {:.4f} ohms'.format(inv_hmc.predict_Rp(percentile=97.5)))


# Only fit peaks that have a prominence of >= 5% of the estimated polarization resistance
inv_map.fit_peaks(prom_rthresh=0.05)

# plot the peak fit
fig, axes = plt.subplots(1, 2, figsize=(8, 3))
inv_map.plot_peak_fit(ax=axes[0])  # Plot the overall peak fit
inv_map.plot_peak_fit(ax=axes[1], plot_individual_peaks=True)  # Plot the individual peaks

fig.tight_layout()


# Extract the peak parameters
inv_map.extract_peak_info()


# Perform peak fit from approximate time constants
tau0_guess = [0.1, 10]  # specify time constant guesses - these can be rough
inv_map.fit_peaks_constrained(tau0_guess)

# plot the peak fit
fig, axes = plt.subplots(1, 2, figsize=(8, 3))
inv_map.plot_peak_fit(ax=axes[0])  # Plot the overall peak fit
inv_map.plot_peak_fit(ax=axes[1], plot_individual_peaks=True)  # Plot the individual peaks

fig.tight_layout()


# Extract the peak parameters
inv_map.extract_peak_info()


"Load data"
# load simulated impedance data with noise
Z_file = os.path.join(datadir,'simulated','Z_BimodalTP-DDT_Orazem_0.25.csv')
Zdf = pd.read_csv(Z_file)

# load true DDT
g_file = os.path.join(datadir,'simulated','gamma_BimodalTP-DDT.csv')
g_true = pd.read_csv(g_file)

# extract frequency and complex impedance
freq, Z = fl.get_fZ(Zdf)

# plot the data
bp.plot_eis(Zdf)


"Fit the data"
# Define the distribution to be recovered (transmissive planar DDT) in the Inverter initialization
# Use a slightly expanded basis frequency range to fully capture the tail of the low-frequency peak
inv_hmc2 = Inverter(distributions={'DDT':{'kernel':'DDT','dist_type':'parallel','bc':'transmissive',
                                         'symmetry':'planar','basis_freq':np.logspace(6,-3,91)}
                                 }
                  )
inv_map2 = Inverter(distributions={'DDT':{'kernel':'DDT','dist_type':'parallel','bc':'transmissive',
                                         'symmetry':'planar','basis_freq':np.logspace(6,-3,91)}
                                 }
                  )

# # Perform HMC fit
start = time.time()
inv_hmc2.fit(freq, Z, mode='sample')
elapsed = time.time() - start
print('HMC fit time {:.2f}'.format(elapsed))

# Perform MAP fit
start = time.time()
inv_map2.fit(freq, Z)
elapsed = time.time() - start
print('MAP fit time {:.2f}'.format(elapsed))


"Visualize DDT and impedance fit"
# plot impedance fit and recovered DRT
fig,axes = plt.subplots(1, 2, figsize=(8, 3.5))

# plot fits of impedance data
inv_hmc2.plot_fit(axes=axes[0], plot_type='nyquist', color='k', label='HMC fit', data_label='Data')
inv_map2.plot_fit(axes=axes[0], plot_type='nyquist', color='r', label='MAP fit', plot_data=False)

# plot true DRT
p = axes[1].plot(g_true['tau'], g_true['gamma']*1000, label='True', ls='--')

# Plot recovered DRT at given tau values
tau_plot = g_true['tau'].values
inv_hmc2.plot_distribution(ax=axes[1], tau_plot=tau_plot, color='k', label='HMC mean', ci_label='HMC 95% CI')
inv_map2.plot_distribution(ax=axes[1], tau_plot=tau_plot, color='r', label='MAP')

axes[1].legend()

fig.tight_layout()


"Visualize the recovered error structure"
# For visual clarity, only MAP results are shown.
# HMC results can be obtained in the same way
fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharex=True)

# plot residuals and estimated error structure
inv_map2.plot_residuals(axes=axes)

# plot true error structure in miliohms
p = axes[0].plot(freq, 3*Zdf['sigma_re'] * 1000, ls='--')
axes[0].plot(freq, -3*Zdf['sigma_re'] * 1000, ls='--', c=p[0].get_color())
axes[1].plot(freq, 3*Zdf['sigma_im'] * 1000, ls='--')
axes[1].plot(freq, -3*Zdf['sigma_im'] * 1000, ls='--', c=p[0].get_color(), label='True $\pm 3\sigma$')

axes[1].legend()

fig.tight_layout()


"Get the polarization resistance, ohmic resistance, and inductance"
# Results are shown for HMC only, but can be obtained for the MAP fit in the same way.
# The only difference is that the MAP fit does not provide percentile prediction.
print('R_inf: {:.4f} ohms'.format(inv_hmc2.R_inf))
print('Inductance: {:.4e} H'.format(inv_hmc2.inductance))
print('Polarization resistance: {:.4f} ohms'.format(inv_hmc2.predict_Rp()))
print('Rp 2.5 percentile: {:.4f} ohms'.format(inv_hmc2.predict_Rp(percentile=2.5)))
print('Rp 97.5 percentile: {:.4f} ohms'.format(inv_hmc2.predict_Rp(percentile=97.5)))



