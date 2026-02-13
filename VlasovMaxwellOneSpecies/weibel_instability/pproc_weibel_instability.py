import params_weibel_instability as damping_params

import os
import cunumpy as xp
import h5py
from feectools.ddm.mpi import mpi as MPI
from matplotlib import pyplot as plt, gridspec
from struphy import main
from struphy.physics.physics import Units

# post process raw data
path = os.path.join(os.getcwd(), "sim_data")
main.pproc(path=path)

# get sim data
simdata = main.load_data(path=path)

# get parameters
dt = damping_params.time_opts.dt
algo = damping_params.time_opts.split_algo
Nel = damping_params.grid.Nel
p = damping_params.derham_opts.p

env = damping_params.env

ppc = damping_params.loading_params.ppc

#get units
units = Units(damping_params.base_units)
model = damping_params.model
model.units = units
A_bulk = model.bulk_species.mass_number
Z_bulk = model.bulk_species.charge_number
model.units.derive_units(
    velocity_scale = model.velocity_scale,
    A_bulk = A_bulk,
    Z_bulk = Z_bulk
)
unit_t = model.units.t

### Show initial EM-field ###

def plot_EM_state(time_step: int, n_dim = 3):
    eta1 = simdata.grids_log[0]

    electric_field = simdata.spline_values["em_fields"]["e_field_log"][time_step]
    magnetic_field = simdata.spline_values["em_fields"]["b_field_log"][time_step]
    
    fig, axs = plt.subplots(nrows = 2, ncols = 3, figsize = (8,6), sharex = True, sharey = True)

    for i in range(n_dim):
        axs[0,i].plot(eta1, electric_field[i][:,0,0])
        axs[0,i].set_title(fr"$E_{i+1}$")

        axs[1,i].plot(eta1, magnetic_field[i][:,0,0])
        axs[1,i].set_title(fr"$B_{i+1}$")
    
    axs[0,0].set_ylabel(r"Electric field value")
    axs[1,0].set_ylabel(r"Magnetic field value")
    axs[1,0].set_xlabel(r"$\eta_1$")
    axs[1,1].set_xlabel(r"$\eta_2$")
    axs[1,2].set_xlabel(r"$\eta_3$")

    fig.suptitle(f"EM-field at time step: {time_step}")

    plt.show()

plot_EM_state(0)

### Progression of energy in EM-field along different directions ###
phy_grid = simdata.grids_phy[0].shape

Nt = simdata.t_grid
unit_volume = xp.prod([1/(phy_grid[i] - 1) for i in range(len(phy_grid))])

def field_energy(field) -> float:
    """
    Calculate totoal energy of field in space
    """

    energy_square = xp.sum(field ** 2)

    return energy_square * unit_volume / 2

# function to extract field energy along each axis at all time
extract_field_energy_axes = lambda field: [
    xp.array([
        field_energy(simdata.spline_values["em_fields"][field][t][i]) for t in Nt
        ])
    for i in range(3)
]

E_energy = extract_field_energy_axes("e_field_log")
B_energy = extract_field_energy_axes("b_field_log")

# plotting
fig, ax = plt.subplots(1, figsize = (18,12), sharex=True)

# Electric field
ax.plot(simdata.t_grid, E_energy[0], label=fr"$\frac{{\|E_{{{1}}}\|^2}}{{2}}$")
ax.plot(simdata.t_grid, E_energy[1], label=fr"$\frac{{\|E_{{{2}}}\|^2}}{{2}}$")
ax.plot(simdata.t_grid, B_energy[2], label=fr"$\frac{{\|B_{{{3}}}\|^2}}{{2}}$")

ax.set_ylabel("Energy [a.u.]")

ax.legend()
ax.set_yscale("log")

plt.show()

### Binning distribution progression ###      
e1_bins = simdata.f["kinetic_ions"]["e1_v1_density"]["grid_e1"]
v1_bins = simdata.f["kinetic_ions"]["e1_v1_density"]["grid_v1"]  
nrows = 3
ncols = 4
ntime = len(simdata.f["kinetic_ions"]["e1_v1_density"]["f_binned"]) 
time_indices = [int( i/(nrows*ncols-1) * (ntime - 1) ) for i in range(nrows*ncols)]

fig, axs = plt.subplots(nrows = nrows, ncols = ncols, figsize = (14,10), sharex=True, sharey=True)
for i in range(nrows):
    for j in range(ncols):
        ax_maxwellian = axs[i][j]
        time_idx = time_indices[j + i*ncols]

        #maxwellian distribution plot
        color_mapped = simdata.f["kinetic_ions"]["e1_v1_density"]["f_binned"][time_idx].T
        pcm = ax_maxwellian.pcolor(e1_bins,v1_bins, color_mapped)

        ax_maxwellian.set_xlabel(r"$\eta_1$")
        ax_maxwellian.set_ylabel(r"$v_x$")
        ax_maxwellian.set_title(fr"full-$f$ at t = {simdata.t_grid[time_idx]:4.2e} s")
        fig.colorbar(pcm, ax = ax_maxwellian)
        
plt.tight_layout()
plt.show()
