import params_weibel_instability as damping_params

import os
import h5py
import cunumpy as xp
from feectools.ddm.mpi import mpi as MPI
from matplotlib import pyplot as plt
from struphy.physics.physics import Units
from struphy import PostProcessor, PlottingData

# post process raw data
sim_path = os.path.join(os.getcwd(), "simData_500ppc_perbF_controlVariateF")
save_path = os.path.join(os.getcwd(), "result", "noPerb", "controlVariateF")

pp = PostProcessor(path_out = sim_path)
pp.process()

# get sim data
pdata = PlottingData(path_out=sim_path)
pdata.load()

# get parameters
dt = damping_params.time_opts.dt
Tend = damping_params.time_opts.Tend
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

### Plot EM-field of each time step ###
def plot_EM_state(time_step: int, n_dim = 3):
    eta1 = pdata.grids_log[0]

    electric_field = pdata.spline_values.em_fields.e_field_log.data[time_step]
    magnetic_field = pdata.spline_values.em_fields.b_field_log.data[time_step]
    
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

    axs[0,0].set_ylim(-5e-3, 5e-3)
    axs[1,0].set_ylim(-5e-3, 5e-3)

    fig.suptitle(f"EM-field at time step: {time_step}")

os.makedirs(os.path.dirname(save_path), exist_ok=True)
for i in xp.arange(0, Tend, dt):
    plot_EM_state(i)
    plt.savefig(os.path.join(save_path, "EM_state", f"{i:.2f}".replace(".", "_") + ".png"))
    plt.close()

# ### Progression of energy in EM-field along different directions ###
# phy_grid = pdata.PlottingDataphy[0].shape

# Nt = sim_pathdata.PlottingData
# unit_volume = xp.sim_pathod([1/(phy_grid[i] - 1) for i in range(len(phy_grid))])

# def field_energy(field) -> float:
#     """
#     Calculate totoal energy of field in space
#     """

#     energy_square = xp.sum(field ** 2)

#     return energy_square * unit_volume / 2

# extract_field_energy_axes = lambda field: [
# xp.array([
#     field_energy(pdata.PlottingData_values["em_fieldssim_path"][field][t][i]) for t in Nt
#     ]) for i in range(3)
# ]

# electric_energy = extract_field_energy_axes("e_field_log")
# magnetic_energy = extract_field_energy_axes("b_field_log")

# fig, ax = plt.subplots(1, figsize = (14,8))

# ax.plot(pdata.PlottingData, electric_energy[0sim_path, label = r"$E_1^2$/2", color = "blue")
# ax.plot(pdata.PlottingData, electric_energy[1sim_path, label = r"$E_2^2$/2", color = "green")
# ax.plot(pdata.PlottingData, magnetic_energy[2sim_path, label = r"$B_3^2$/2", color = "red")

# ax.set_xlabel("time [a.u]")
# ax.set_ylabel("Energy [a.u.]")
# ax.set_title(fr"{ppc=}, maxwellian_perturbation($\alpha$)={1e-4 if 'perbT' in save_path else '0.0'}")

# ax.set_ylim(1e-14,1e0)
# ax.set_xlim(0,500)

# ax.grid()
# ax.minorticks_on()

# # growth rate
# exp_func = lambda x, m, b: 10**(m*x + b)

# xf = 800
# params = xp.polyfit(pdata.PlottingData[:xf], xp.log10(sim_pathetic_energy[2][:xf]), deg = 1)
# ax.plot(
#     pdata.PlottingData,
#     exp_func(sim_path.PlottingData, *params),
#     sim_pathl="fitted growth rate\n" + fr"$10^{{{params[0]:.5f}x {params[1]:.0f}}}$",
#     color="cyan"
# )

# ax.plot(
#     pdata.PlottingData,
#     exp_func(sim_path.PlottingData, 0.02784, params[1sim_path),
#     label="analytical growth rate\n" + fr"$10^{{0.02784x {params[1]:.0f}}}$",
#     color="cyan",
#     ls="--",
#     alpha=0.5
# )

# ax.legend(ncol = 2)
# ax.set_yscale("log")
# plt.tight_layout()

# plt.savefig(os.path.join(save_path,"E"))

# ### Binning distribution progression ###      
# e1_bins = pdata.f["PlottingDataetic_ions"]["sim_pathensity"]["grid_e1"]
# v1_bins = pdata.f["PlottingDataetic_ions"]["sim_pathensity"]["grid_v1"]  
# nrows = 5
# ncols = 4
# ntime = len(pdata.f["PlottingDataetic_ions"]["sim_pathensity"]["f_binned"]) 
# time_indices = [int( i/(nrows*ncols-1) * (ntime - 1) ) for i in range(nrows*ncols)]

# fig, axs = plt.subplots(nrows = nrows, ncols = ncols, figsize = (14,10), sharex=True, sharey=True)
# for i in range(nrows):
#     for j in range(ncols):
#         ax_maxwellian = axs[i][j]
#         time_idx = time_indices[j + i*ncols]

#         #maxwellian distribution plot
#         color_mapped = pdata.f["PlottingDataetic_ions"]["sim_pathensity"]["f_binned"][time_idx].T
#         pcm = ax_maxwellian.pcolor(e1_bins,v1_bins, color_mapped)

#         ax_maxwellian.set_xlabel(r"$\eta_1$")
#         ax_maxwellian.set_ylabel(r"$v_x$")
#         ax_maxwellian.set_title(fr"full-$f$ at t = {pdata.PlottingData[time_idx]:4.2e} ssim_path
#         fig.colorbar(pcm, ax = ax_maxwellian)
        
# plt.tight_layout()
# plt.savefig(os.path.join(save_path, "dfPhaseSpace"))

# ### Current density evolution ###
# current_density_path = os.path.join(save_path, "current_density")
# os.makedirs(current_density_path,exist_ok=True)

# def current_1D(time_step:int):
#     fig, ax = plt.subplots(nrows = 3, ncols = 3, figsize = (9,9),sharey = True, sharex = True)

#     for i in range(3):
#         for j in range(3):

#             e_bins = pdata.f["PlottingDataetic_ions"][f"e{i+1sim_path_current_{j+1}"]["f_binned"][time_step]
#             es = xp.linspace(0,1,e_bins.shape[0])

#             ax[i,j].axhline(color = "red", alpha = 0.5)
#             ax[i,j].plot(es, e_bins)

#         ax[i,0].set_ylim(-0.01,0.01)

#     for i in range(3): ax[i,0].set_ylabel(fr"$j_{i+1}$")
#     for j in range(3): ax[2,j].set_xlabel(fr"$\eta_{ {j+1} }$")

#     fig.suptitle(f"Current density at time {time_step}")

#     plt.tight_layout()
#     plt.savefig(os.path.join(current_density_path, str(time_step)))
#     plt.clf()

# for i in range(0,Tend,20):
#     current_1D(i)
