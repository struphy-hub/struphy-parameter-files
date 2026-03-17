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
pp.process(physical=True)

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
# Determine electrical potentail growth rate
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
os.makedirs(os.path.join(os.getcwd(), save_path, "phaseSpace"), exist_ok=True)

nrows = 2
ncols = 2
ntime = len(pdata.f.kinetic_ions.e1_e2_density.f_binned) 
time_indices = [int( i/(nrows*ncols-1) * (ntime - 1) ) for i in range(nrows*ncols)]

def plot_phaseSpace(bin_name, quantity, xs, ys, x_label = "x", y_label = "y", in_physical = False):

    fig, axs = plt.subplots(nrows = nrows, ncols = ncols, figsize = (14,10), sharex=True, sharey=True)
    for i in range(nrows):
        for j in range(ncols):
            ax_maxwellian = axs[i][j]
            time_idx = time_indices[j + i*ncols]

            #maxwellian distribution plot
            color_mapped = getattr(
                getattr(pdata.f.kinetic_ions, bin_name), quantity
                )[time_idx].T

            if in_physical: color_mapped = color_mapped.T

            pcm = ax_maxwellian.pcolor(xs, ys, color_mapped)

            ax_maxwellian.set_xlabel(x_label)
            ax_maxwellian.set_ylabel(y_label)
            ax_maxwellian.set_title(f"{quantity} at t = {pdata.t_grid[time_idx]:4.2e}")
            fig.colorbar(pcm, ax = ax_maxwellian)
            
    # plt.tight_layout()
    plt.savefig(os.path.join(save_path, "phaseSpace", f"{bin_name}_{quantity}_phaseSpace"))
    plt.close()

# e1_e2_density binplot in physical coordinate
e1_bin = pdata.f.kinetic_ions.e1_e2_density.grid_e1
e2_bin = pdata.f.kinetic_ions.e1_e2_density.grid_e2

phy_bin = domain(e1_bin, e2_bin, 0, squeeze_out=True) # convert eta to physical coordinate
plot_phaseSpace(bin_name="e1_e2_density", quantity="f_binned", xs=phy_bin[0], ys=phy_bin[1], in_physical=True)
plot_phaseSpace(bin_name="e1_e2_density", quantity="delta_f_binned", xs=phy_bin[0], ys=phy_bin[1], in_physical=True)

# density plot in logical and velocity space
xs = [
    pdata.f.kinetic_ions.e1_v1_density.grid_e1,
    pdata.f.kinetic_ions.e1_v2_density.grid_e1,
    pdata.f.kinetic_ions.e2_v1_density.grid_e2,
    pdata.f.kinetic_ions.e2_v2_density.grid_e2,
    pdata.f.kinetic_ions.v1_v2_density.grid_v1,
]

ys = [
    pdata.f.kinetic_ions.e1_v1_density.grid_v1,
    pdata.f.kinetic_ions.e1_v2_density.grid_v2,
    pdata.f.kinetic_ions.e2_v1_density.grid_v1,
    pdata.f.kinetic_ions.e2_v2_density.grid_v2,
    pdata.f.kinetic_ions.v1_v2_density.grid_v2,
]

x_labels = ["e1", "e1", "e2", "e2", "v1"]
y_labels = ["v1", "v2", "v1", "v2", "v2"]

for curr_bin, X, Y, x_label, y_label in zip((*e_v_bin, v_v_bin), xs, ys, x_labels, y_labels):
    # extract slice and binning quantity for data retrieval
    bin_slice, bin_quantity = curr_bin.slice, curr_bin.output_quantity

    # extract grid names from slice
    grid1, grid2 = bin_slice.split("_")

    bin_name = bin_slice + "_" + bin_quantity
    bin1 = getattr(getattr(pdata.f.kinetic_ions, bin_name), f"grid_{grid1}")
    bin2 = getattr(getattr(pdata.f.kinetic_ions, bin_name), f"grid_{grid2}")
    
    # convert bin to meshgrid
    if len(X.shape) == 1 and len(Y.shape) == 1:
        X, Y = xp.meshgrid(X,Y)

    # plot distribution
    plot_phaseSpace(bin_name=bin_name, quantity="f_binned", xs=X, ys=Y, x_label=x_label, y_label=y_label)
    plot_phaseSpace(bin_name=bin_name, quantity="delta_f_binned", xs=X, ys=Y, x_label=x_label, y_label=y_label)


# ------------------
# Current density evolution
# ------------------

current_density_path = os.path.join(save_path, "current_density")
os.makedirs(current_density_path, exist_ok=True)

def current_1D(time_idx:int):
    nrows, ncols = 2, 3
    fig, ax = plt.subplots(nrows = nrows, ncols = ncols, figsize = (9,9),sharey = True, sharex = True)

    for i in range(nrows):
        for j in range(ncols):

            e_bins = getattr(pdata.f.kinetic_ions, f"e{j+1}_current_{i+1}").f_binned[time_idx]
            es = xp.linspace(0,1,e_bins.shape[0])

            ax[i,j].axhline(color = "red", alpha = 0.5)
            ax[i,j].plot(es, e_bins)

        ax[i,0].set_ylim(-0.25,0.25)

    for i in range(nrows): ax[i,0].set_ylabel(fr"$j_{i+1}$")
    for j in range(ncols): ax[1,j].set_xlabel(fr"$\eta_{ {j+1} }$")

    time = pdata.t_grid[time_idx]
    fig.suptitle(f"Current density at time {time:.2f}")

    plt.tight_layout()
    plt.savefig(os.path.join(
        current_density_path,
        f"{time:.2f}".replace(".", "_") + ".png"
    ))
    plt.close()

for t in time_indices:
    current_1D(t)

# ------------------
# Show evolution of electric potential
# ------------------
nrows = 2
ncols = 2
ntime = len(pdata.f.kinetic_ions.e1_e2_density.f_binned) 
time_indices = [int( i/(nrows*ncols-1) * (ntime - 1) ) for i in range(nrows*ncols)]

time_keys = pdata.t_grid

fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14,10), sharex=True, sharey=True)

for i in range(nrows):
    for j in range(ncols):
        ax_maxwellian = axs[i][j]
        time_idx = time_indices[j + i*ncols]

        phi = pdata.spline_values.em_fields.phi_phy.data[time_keys[time_idx]][0][:,:,0]

        pcm = ax_maxwellian.pcolormesh(pdata.grids_phy[0][:,:,0], pdata.grids_phy[1][:,:,0], phi, cmap="Purples")

        ax_maxwellian.set_xlabel("x")
        ax_maxwellian.set_ylabel(r"y")
        ax_maxwellian.set_title(f"Electrical potential at t = {pdata.t_grid[time_idx]:4.2e}")

        fig.colorbar(pcm, ax=ax_maxwellian)

plt.tight_layout()
plt.savefig(os.path.join(save_path, "potentialEvolution"))
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