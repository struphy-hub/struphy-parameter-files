import params_weibel_instability as damping_params

import os
import cunumpy as xp
import h5py
from feectools.ddm.mpi import mpi as MPI
from matplotlib import pyplot as plt
from struphy import main
from struphy.io.options import Units

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

### Progression of total Energy in EM field ###

# get scalar data (post processing not needed for scalar data)
if MPI.COMM_WORLD.Get_rank() == 0:
    pa_data = os.path.join(path, "data")
    with h5py.File(os.path.join(pa_data, "data_proc0.hdf5"), "r") as f:
        time = f["time"]["value"][()]*unit_t
        E = f["scalar"]["en_E"][()]
        B = f["scalar"]["en_B"][()]

    # plot
    fig, ax = plt.subplots(1, figsize = (18,12))

    ax.plot(time, E, label=r"$E^2/2$")
    ax.plot(time, B, label=r"$B^2/2$")

    ax.set_title(f"{dt=}, {algo=}, {Nel=}, {p=}, {ppc=}")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("energy [a.u.]")

    # ax.set_xlim(0,0.75 * 1e-5)
    ax.set_yscale("log")

    ax.legend()

    plt.show()      

### Progression of energy in EM-field along different directions ###
phy_grid = simdata.grids_phy[0].shape
print(phy_grid)

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
fig, ax = plt.subplots(2, figsize = (18,12), sharex=True)

# Electric field
for i in range(3):
    ax[0].plot(simdata.t_grid*unit_t, E_energy[i], label=fr"$\frac{{\|E_{{{i+1}}}\|^2}}{{2}}$")

ax[0].set_ylabel("electric energy $E^2/2$ [a.u.]")

ax[0].legend()
ax[0].set_yscale("log")

# Magnetic field
for i in range(3):
    ax[1].plot(simdata.t_grid*unit_t, B_energy[i], label=fr"$\frac{{\|B_{{{i+1}}}\|^2}}{{2}}$")

ax[1].set_xlabel("time [s]")
ax[1].set_ylabel("magnetic energy $B^2/2$ [a.u.]")

ax[1].set_xlim(0, 0.75 * 1e-5)

ax[1].legend()
ax[1].set_yscale("log")

plt.show()

# Compare self-implemented energy along axis to API's result

def total_field_energy(y1, y2, y3):
    return y1 + y2 + y3

def plt_comparison(x, y_API, y1, y2, y3, ax):
    y_cal = total_field_energy(y1,y2,y3)

    ax.plot(x, y_API, label = "API")
    ax.plot(x, y_cal, label = "cal")
    ax.set_yscale("log")
    ax.legend()

fig, ax = plt.subplots(2, figsize = (18,12), sharex = True)

plt_comparison(time, E, *E_energy, ax = ax[0])
ax[0].set_ylabel("electric energy $E^2/2$ [a.u.]")
ax[0].set_ylim(0,1e6)

plt_comparison(time, B, *B_energy, ax = ax[1])
ax[1].set_xlabel("time [s]")
ax[1].set_ylabel("magnetic energy $B^2/2$ [a.u.]")

ax[1].set_xlim(0, 0.75 * 1e-5)
ax[1].set_ylim(0,1e6)
ax[1].grid()

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
        ax_maxwellian.set_title(fr"full-$f$ at t = {simdata.t_grid[time_idx]*unit_t:4.2e} s")
        fig.colorbar(pcm, ax = ax_maxwellian)
        
plt.tight_layout()
plt.show()