import params_weibel_instability as damping_params

import os
import cunumpy as xp
import h5py
from feectools.ddm.mpi import mpi as MPI
from matplotlib import pyplot as plt
from struphy import main
from struphy.io.options import Units

# # post process raw data
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

# ### Initial EM-field ###
# eta_1 = simdata.grids_log[0]

# init_electric_field_1 = simdata.spline_values["em_fields"]["e_field_log"][0][0] # Initial electric field along eta1-axis solved from Poisson's equation
# init_magnetic_field_3 = simdata.spline_values["em_fields"]["b_field_log"][0][2] # Initial magnetic field along z-axis as defined by perturbation

# fig, axs = plt.subplots(nrows = 2, ncols = 1, figsize = (8,6), sharex = True, sharey = True)

# axs[0].plot(simdata.grids_log[0], init_electric_field_1[:,0,0])
# axs[0].set_ylabel(r"$E_1$")

# axs[1].plot(simdata.grids_log[0], init_magnetic_field_3[:,0,0])
# axs[1].set_ylabel(r"$B_3$")
# axs[1].set_xlabel(r"$\eta_1$")

# fig.suptitle(r"Initial EM-field strength along $\eta_1$")
# # plt.show()

### Progression of energy in EM-field along different directions ###
Nel = tuple(el - 1 for el in simdata.grids_phy[0].shape)

Nt = simdata.t_grid
unit_volume = 1 / xp.prod(Nel)

def field_energy(field) -> float:
    """
    Calculate totoal energy of field in space
    """

    energy_square = xp.sum(field ** 2)

    return energy_square * unit_volume / 2



E_1_energy = tuple(field_energy(simdata.spline_values["em_fields"]["e_field_log"][t][0]) for t in Nt)
E_2_energy = tuple(field_energy(simdata.spline_values["em_fields"]["e_field_log"][t][1]) for t in Nt)
# E_3_energy = tuple(field_energy(simdata.spline_values["em_fields"]["e_field_log"][t][2]) for t in Nt)

# B_1_energy = tuple(field_energy(simdata.spline_values["em_fields"]["b_field_log"][t][0]) for t in Nt)
# B_2_energy = tuple(field_energy(simdata.spline_values["em_fields"]["b_field_log"][t][1]) for t in Nt)
B_3_energy = tuple(field_energy(simdata.spline_values["em_fields"]["b_field_log"][t][2]) for t in Nt)

fig, ax = plt.subplots(1, figsize = (8,6))

ax.plot(simdata.t_grid, E_1_energy, label = r"$\frac{||E_1||^2}{2}$")
ax.plot(simdata.t_grid, E_2_energy, label = r"$\frac{||E_2||^2}{2}$")
# ax.plot(simdata.t_grid, E_3_energy, label = r"$\frac{||E_3||^2}{2}$")

# ax.plot(simdata.t_grid, B_1_energy, label = r"$\frac{||B_1||^2}{2}$")
# ax.plot(simdata.t_grid, B_2_energy, label = r"$\frac{||B_2||^2}{2}$")
ax.plot(simdata.t_grid, B_3_energy, label = r"$\frac{||B_3||^2}{2}$")

ax.set_xlabel("Time")
ax.set_ylabel("Energy")

ax.legend()

ax.set_yscale("log")

plt.show()

# get scalar data (post processing not needed for scalar data)
if MPI.COMM_WORLD.Get_rank() == 0:
    pa_data = os.path.join(env.path_out, "data")
    with h5py.File(os.path.join(pa_data, "data_proc0.hdf5"), "r") as f:
        time = f["time"]["value"][()]*unit_t
        E = f["scalar"]["en_E"][()]
        B = f["scalar"]["en_B"][()]

    # plot
    plt.figure(figsize=(18, 12))
    plt.plot(time, E/2, label="E")
    plt.plot(time, B, label = "B")
    plt.legend()
    plt.title(f"{dt=}, {algo=}, {Nel=}, {p=}, {ppc=}")

    plt.yscale("log")
    plt.xlabel("time [s]")
    plt.ylabel("electric energy $E^2/2$ [a.u.]")

    plt.show()      

# ### Binning distribution progression ###      
# e1_bins = simdata.f["kinetic_ions"]["e1_v1"]["grid_e1"]
# v1_bins = simdata.f["kinetic_ions"]["e1_v1"]["grid_v1"]  
# nrows = 3
# ncols = 4
# ntime = len(simdata.f["kinetic_ions"]["e1_v1"]["f_binned"]) 
# time_indices = [int( i/(nrows*ncols-1) * (ntime - 1) ) for i in range(nrows*ncols)]

# fig, axs = plt.subplots(nrows = nrows, ncols = ncols, figsize = (14,10), sharex=True, sharey=True)
# for i in range(nrows):
#     for j in range(ncols):
#         ax_maxwellian = axs[i][j]
#         time_idx = time_indices[j + i*ncols]

#         #maxwellian distribution plot
#         color_mapped = simdata.f["kinetic_ions"]["e1_v1"]["f_binned"][time_idx].T
#         pcm = ax_maxwellian.pcolor(e1_bins,v1_bins, color_mapped)

#         ax_maxwellian.set_xlabel(r"$\eta_1$")
#         ax_maxwellian.set_ylabel(r"$v_x$")
#         ax_maxwellian.set_title(fr"full-$f$ at t = {simdata.t_grid[time_idx]*unit_t:4.2e} s")
#         fig.colorbar(pcm, ax = ax_maxwellian)
        
# plt.tight_layout()
# plt.show()