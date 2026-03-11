from params_dicotron import *
from struphy import PlottingData, PostProcessor

import os
import cunumpy as xp
from matplotlib import pyplot as plt


# ------------------
# Post process simulation data
# ------------------
sim_path = os.path.join(os.getcwd(), "simdata")

pp = PostProcessor(path_out=sim_path)
pp.process()

pdata = PlottingData(path_out=sim_path)
pdata.load()

# path to save plots
save_path = os.path.join(os.getcwd(), "images")


# ------------------
# Define transformation functions
# ------------------

# Transform logical space to cartesian then polar coordinate
domain_params = domain.params
a1 = domain_params["a1"]
a2 = domain_params["a2"]
Lz = domain_params["Lz"]
poc = domain_params["poc"]

def logical_to_cylindrical(eta1,eta2,eta3):
    r = (a1 + (a2-a1)*eta1)
    theta = (2*xp.pi*eta2/poc)
    z = Lz*eta3
    
    return r, theta, z

r, theta, z = logical_to_cylindrical(*pdata.grids_log)


# ------------------
# Check simulation domain
# ------------------

domain.show(save_dir=os.path.join(save_path,"domain.png"))


# ------------------
# Check Initial electrical potential
# (Dirichlet boundary conditions)
#
# phi(t,r_min,theta) = phi(t,r_max,theta) = 0
# ------------------

# create figure to show boundary conditions
fig, ax = plt.subplots(ncols = 2, figsize = (12,6))
fig.suptitle(rf"$\phi_0$ at $\eta_3$ = {pdata.grids_log[2][0]}")

# determine boundary condition in logical coordinate
init_phi = pdata.spline_values.em_fields.phi_log.data[0.0]

eta1 = pdata.grids_log[0]
eta2 = pdata.grids_log[1]
Eta1, Eta2 = xp.meshgrid(eta1,eta2)

# show boundary condition in logical space
pcm = ax[0].pcolormesh(Eta1,Eta2,init_phi[0][:,:,0], cmap="Purples")
fig.colorbar(pcm, ax = ax[0])
ax[0].set_xlabel(r"$\eta_1$")
ax[0].set_ylabel(r"$\eta_2$")
ax[0].set_title("Logical space")

# determine boundary condiiton in cylindrical coordinate

# create heatmap from cylindrical coordinate
R, Theta = xp.meshgrid(r,theta)

pcm = ax[1].pcolormesh(R, Theta, init_phi[0][:,:,0], cmap = "Purples")
fig.colorbar(pcm, ax = ax[1])

ax[1].set_xlabel(r"$r$")
ax[1].set_ylabel(r"$\theta$");
ax[1].set_title("Cylindrical coordinate")

plt.tight_layout()
plt.savefig(os.path.join(save_path, "initialElectricalPotential.png"))


# ------------------
# Check initial mass density distribution
# ------------------

bin1 = pdata.f.kinetic_ions.e1_e2_density.grid_e1
bin2 = pdata.f.kinetic_ions.e1_e2_density.grid_e2

color_mapped = pdata.f.kinetic_ions.e1_e2_density.delta_f_binned[0].T

fig, ax = plt.subplots(ncols = 2, figsize = (12,6))

# logical space
pcm = ax[0].pcolor(bin1,bin2,color_mapped)
fig.colorbar(pcm, ax=ax[0])
ax[0].set_xlabel(r"$\eta_1$")
ax[0].set_ylabel(r"$\eta_2$")
ax[0].set_title("logical space");

# cylindrical coordinate
r, theta, _ = logical_to_cylindrical(bin1, bin2, 0)

pcm = ax[1].pcolor(r,theta,color_mapped)
ax[1].set_xlabel(r"$r$")
ax[1].set_ylabel(r"$\theta$")
ax[1].set_title("cylindrical coordinate")

fig.colorbar(pcm, ax=ax[1])

ax[1].axvline(r_minus, ls = "--", color = "red", label = r"$r_-$")
ax[1].axvline(r_plus, ls = "--", color = "red", label = r"$r_+$")
ax[1].legend(loc = "upper right")

fig.suptitle(r"Initial $\Delta f$ density distribution")

plt.tight_layout()
plt.savefig(os.path.join(save_path, "initialDensityDistribution.png"))