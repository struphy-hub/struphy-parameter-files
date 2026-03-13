from params_dicotron import *
from struphy import PlottingData, PostProcessor

import os
import cunumpy as xp
from matplotlib import pyplot as plt
import h5py


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
init_phi = xp.array(pdata.spline_values.em_fields.phi_log.data[0.0]).T

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
plt.close()


# ------------------
# Check initial mass density distribution
# ------------------

def plot_rho_dist(quantity:str = "f_binned"):
    bin1 = pdata.f.kinetic_ions.e1_e2_density.grid_e1
    bin2 = pdata.f.kinetic_ions.e1_e2_density.grid_e2

    color_mapped = getattr(pdata.f.kinetic_ions.e1_e2_density, quantity)[0].T

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

    fig.suptitle(f"Initial {quantity} density distribution")

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"initialDensityDistribution_{quantity}.png"))
    plt.close()

plot_rho_dist(quantity="f_binned")
plot_rho_dist(quantity="delta_f_binned")


# ------------------
# Determine energy growth rate
# ------------------

# get scalar data (post processing not needed for scalar data)
pa_data = os.path.join(env.path_out, "data")
with h5py.File(os.path.join(pa_data, "data_proc0.hdf5"), "r") as f:
    time = f["time"]["value"][()]
    en_phi = f["scalar"]["en_phi"][()]
    en_particles = f["scalar"]["en_particles"][()]

fig, ax = plt.subplots(1, figsize = (18, 12))

# plot
ax.plot(time, en_phi, label=r"$\phi$")
ax.plot(time, en_particles, label = "particles")

ax.set_yscale('log')
ax.legend()

ax.set_title(f"{time_opts.dt=}, {time_opts.split_algo=}, {grid.Nel=}, {derham_opts.p=}, {loading_params.ppc=}")
ax.set_xlabel("time")
ax.set_ylabel("Energy [a.u.]")

plt.tight_layout()
plt.savefig(os.path.join(save_path, "growth_rate.png"))
plt.close()


# ------------------
# Show evolution of mass density distribution
# ------------------

nrows = 5
ncols = 4
ntime = len(pdata.f.kinetic_ions.e1_e2_density.f_binned) 
time_indices = [int( i/(nrows*ncols-1) * (ntime - 1) ) for i in range(nrows*ncols)]

def plot_phaseSpace(bin_name, quantity, xs, ys, x_label = "r", y_label = r"$\theta$"):

    fig, axs = plt.subplots(nrows = nrows, ncols = ncols, figsize = (14,10), sharex=True, sharey=True)
    for i in range(nrows):
        for j in range(ncols):
            ax_maxwellian = axs[i][j]
            time_idx = time_indices[j + i*ncols]

            #maxwellian distribution plot
            color_mapped = getattr(
                getattr(pdata.f.kinetic_ions, bin_name), quantity
                )[time_idx].T
            pcm = ax_maxwellian.pcolor(xs, ys, color_mapped)

            ax_maxwellian.set_xlabel(x_label)
            ax_maxwellian.set_ylabel(y_label)
            ax_maxwellian.set_title(f"{quantity} at t = {pdata.t_grid[time_idx]:4.2e}")
            fig.colorbar(pcm, ax = ax_maxwellian)

            if "e1_e2" in bin_name:
                ax_maxwellian.axvline(r_minus, ls = "--", color = "red")
                ax_maxwellian.axvline(r_plus, ls = "--", color = "red")
            
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{bin_name}_{quantity}_phaseSpace"))
    plt.close()

plot_phaseSpace(bin_name="e1_e2_density", quantity="f_binned", xs=r, ys=theta)


# ------------------
# Show evolution of electric potential
# ------------------
nrows = 5
ncols = 4
ntime = len(pdata.f.kinetic_ions.e1_e2_density.f_binned) 
time_indices = [int( i/(nrows*ncols-1) * (ntime - 1) ) for i in range(nrows*ncols)]

r, theta, _ = logical_to_cylindrical(*pdata.grids_log)
R, Theta = xp.meshgrid(r, theta)
time_keys = pdata.t_grid

fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14,10), sharex=True, sharey=True)

for i in range(nrows):
    for j in range(ncols):
        ax_maxwellian = axs[i][j]
        time_idx = time_indices[j + i*ncols]

        phi = xp.array(
            pdata.spline_values.em_fields.phi_log.data[time_keys[time_idx]][0][:,:,0]
        ).T

        pcm = ax_maxwellian.pcolormesh(R, Theta, phi, cmap="Purples")

        ax_maxwellian.set_xlabel("r")
        ax_maxwellian.set_ylabel(r"$\theta$")
        ax_maxwellian.set_title(f"Electrical potential at t = {pdata.t_grid[time_idx]:4.2e}")

        fig.colorbar(pcm, ax=ax_maxwellian)

plt.tight_layout()
plt.savefig(os.path.join(save_path, "potentialEvolution"))
plt.close()


# ------------------
# Check initial magnetic field
# ------------------

t0 = pdata.t_grid[0]
B_field = pdata.spline_values.em_fields.b_field_log.data[t0]

nrows, ncols = 3, 3
fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12,12), sharex=True, sharey=True)

for i in range(nrows):

    # retreive B along axis i
    B_axis = B_field[i]
    for j in range(ncols):
        # determine slicing of B with j
        idx = [0,0,0]
        idx[j] = slice(None)

        axs[i,j].plot(pdata.grids_log[j], B_axis[tuple(idx)])

        axs[i,j].set_xlabel(rf"$\eta${j}")
        axs[i,j].set_ylabel(rf"$B${i}")

plt.tight_layout()
plt.savefig(os.path.join(save_path, "initB"))
plt.close()


# ------------------
# Save copy of used parameter file as txt
# ------------------

# read content of parameter file
with open("params_dicotron.py", "r") as py_file:
    content = py_file.read()
    py_file.close()

# write content to text file
with open(os.path.join(save_path, "param.txt"), "w") as txt_file:
    txt_file.write(content)
    txt_file.close()