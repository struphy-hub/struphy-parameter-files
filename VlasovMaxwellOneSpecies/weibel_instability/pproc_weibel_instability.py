import params_weibel_instability as params

import os
import h5py
import cunumpy as xp
from feectools.ddm.mpi import mpi as MPI
from matplotlib import pyplot as plt
from struphy.physics.physics import Units
from struphy import PostProcessor, PlottingData

# post process raw data
sim_name = params.sim_folder
sim_path = os.path.join(os.getcwd(), sim_name)
save_path = os.path.join(os.getcwd(), "result", "noPerb", "controlVariate"+sim_name[-1])

pp = PostProcessor(path_out = sim_path)
pp.process()

# get sim data
pdata = PlottingData(path_out=sim_path)
pdata.load()

# get parameters
dt = params.time_opts.dt
Tend = params.time_opts.Tend
algo = params.time_opts.split_algo
Nel = params.grid.Nel
p = params.derham_opts.p

env = params.env
ppc = params.loading_params.ppc

#get units
units = Units(params.base_units)
model = params.model
model.units = units
A_bulk = model.bulk_species.mass_number
Z_bulk = model.bulk_species.charge_number
model.units.derive_units(
    velocity_scale = model.velocity_scale,
    A_bulk = A_bulk,
    Z_bulk = Z_bulk
)
unit_t = model.units.t

control_variate = params.weights_params.control_variate
split_algo = params.time_opts.split_algo

### Gauss law violation and progression of EM-field energy along different direction###

if MPI.COMM_WORLD.Get_rank() == 0:
    # read data for gauss law violation
    pa_data = os.path.join(sim_name, "data")
    with h5py.File(os.path.join(pa_data, "data_proc0.hdf5"), "r") as f:
        time = f["time"]["value"][()]
        gauss_error = f["scalar"]["gauss_error"][()]


    # Progression of energy in EM-field along different directions
    phy_grid = pdata.grids_phy[0].shape

    Nt = pdata.t_grid
    unit_volume = xp.prod([1/(phy_grid[i] - 1) for i in range(len(phy_grid))])

    def field_energy(field) -> float:
        """
        Calculate totoal energy of field in space
        """

        energy_square = xp.sum(field ** 2)

        return energy_square * unit_volume / 2

    extract_field_energy_axes = lambda field: [
        xp.array([
            field_energy(getattr(pdata.spline_values.em_fields, field).data[t][i]) for t in Nt
        ]) for i in range(3)
    ]

    electric_energy = extract_field_energy_axes("e_field_log")
    magnetic_energy = extract_field_energy_axes("b_field_log")

    # plot
    fig, axs = plt.subplots(nrows = 2, ncols = 1, figsize = (10,6), sharex = True)

    # plot of energy in EM-fields
    axs[0].plot(pdata.t_grid, electric_energy[0], label = r"|$E_1|^2$/2", color = "blue")
    axs[0].plot(pdata.t_grid, electric_energy[1], label = r"|$E_2|^2$/2", color = "green")
    axs[0].plot(pdata.t_grid, magnetic_energy[2], label = r"|$B_3|^2$/2", color = "red")

    # determine magnetic field growth rate
    exp_func = lambda x, m, b: 10**(m*x + b)

    xf = xp.abs(pdata.t_grid - 200).argmin()  + 1 # index of time 200 [a.u.] (observed end of growth rate)

    params = xp.polyfit(pdata.t_grid[:xf], xp.log10(magnetic_energy[2][:xf]), deg = 1)
    axs[0].plot(
        pdata.t_grid,
        exp_func(pdata.t_grid, *params),
        label="fitted growth rate\n" + fr"$10^{{{params[0]:.5f}x {params[1]:.0f}}}$",
        color="cyan"
    )

    axs[0].plot(
        pdata.t_grid,
        exp_func(pdata.t_grid, 0.02784, params[1]),
        label="analytical growth rate\n" + fr"$10^{{0.02784x {params[1]:.0f}}}$",
        color="cyan",
        ls="--",
        alpha=0.5
    )
    axs[0].legend(ncol = 2)

    axs[0].set_title(f"Field energy: {split_algo=}, {ppc=}, {control_variate=}")
    axs[0].set_title("Energy in EM field")
    axs[0].set_ylabel("Energy [a.u.]")
    axs[0].set_xlabel("time")
    axs[0].set_ylim(1e-14,1e0)
    axs[0].set_xlim(0,500)

    axs[0].grid()
    axs[0].minorticks_on()

    # plot of gauss violation
    axs[1].plot(time, gauss_error, color = "black")
    axs[1].set_title("Gauss law violation")
    axs[1].set_ylim(1e-9,1e-2)
    axs[1].set_ylabel("residual [a.u.]")

    for ax in axs: ax.grid(); ax.set_yscale("log"); ax.legend(loc = "lower right")

    fig.suptitle(f"VlasovMaxwellOneSpecies simulation:\n {control_variate=}, {ppc=}, {algo=}")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path,"E"))

    ### Binning distribution evolution ###      
    nrows = 5
    ncols = 4
    ntime = len(pdata.f.kinetic_ions.e1_v1_density.f_binned) 
    time_indices = [int( i/(nrows*ncols-1) * (ntime - 1) ) for i in range(nrows*ncols)]

    def plot_phaseSpace(bin, bin_name):
        bins = bin_name.split("_")[:-1]
        grid_1, grid_2, *_ = ["grid_" + s for s in bins]
        bins_1 = getattr(getattr(pdata.f.kinetic_ions, bin_name), grid_1)
        bins_2 = getattr(getattr(pdata.f.kinetic_ions, bin_name), grid_2)

        fig, axs = plt.subplots(nrows = nrows, ncols = ncols, figsize = (14,10), sharex=True, sharey=True)
        for i in range(nrows):
            for j in range(ncols):
                ax_maxwellian = axs[i][j]
                time_idx = time_indices[j + i*ncols]

                #maxwellian distribution plot
                color_mapped = getattr(
                    getattr(pdata.f.kinetic_ions, bin_name), bin
                    )[time_idx].T
                pcm = ax_maxwellian.pcolor(bins_1, bins_2, color_mapped)

                ax_maxwellian.set_xlabel(bins[0])
                ax_maxwellian.set_ylabel(bins[1])
                ax_maxwellian.set_title(f"{bin} at t = {pdata.t_grid[time_idx]:4.2e}")
                fig.colorbar(pcm, ax = ax_maxwellian)
                
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"{bin_name}_{bin}_phaseSpace"))

    plot_phaseSpace("f_binned", bin_name="e1_v1_density")
    plot_phaseSpace("delta_f_binned", bin_name="e1_v1_density")
    plot_phaseSpace("f_binned",bin_name="v1_v2_density")
    plot_phaseSpace("delta_f_binned",bin_name="v1_v2_density")


    ### Plot EM-field of each time step ###
    def plot_EM_state(time_step: int, n_dim = 3):    
        time_step = pdata.t_grid[abs(pdata.t_grid - time_step).argmin()]
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

        fig.suptitle(f"EM-field at time step: {time_step:.2f}, {ppc=},{control_variate=}")

        return time_step

        os.makedirs(os.path.join(save_path, "EM_state"), exist_ok=True)
        for t in xp.arange(0, Tend, 5):
            time_step = plot_EM_state(t)
            plt.savefig(os.path.join(save_path, "EM_state", f"{time_step:.2f}.png"))
            plt.close()

    ### Current density evolution ###
    current_density_path = os.path.join(save_path, "current_density")
    os.makedirs(current_density_path,exist_ok=True)

    def current_1D(time:int):
        time_step = abs(pdata.t_grid - time).argmin()
        fig, ax = plt.subplots(nrows = 3, ncols = 3, figsize = (9,9),sharey = True, sharex = True)

        for i in range(3):
            for j in range(3):

                e_bins = getattr(pdata.f.kinetic_ions, f"e{i+1}_current_{j+1}").f_binned[time_step]
                es = xp.linspace(0,1,e_bins.shape[0])

                ax[i,j].axhline(color = "red", alpha = 0.5)
                ax[i,j].plot(es, e_bins)

            ax[i,0].set_ylim(-0.01,0.01)

        for i in range(3): ax[i,0].set_ylabel(fr"$j_{i+1}$")
        for j in range(3): ax[2,j].set_xlabel(fr"$\eta_{ {j+1} }$")

        fig.suptitle(f"Current density at time {time:.2f}")

        plt.tight_layout()
        plt.savefig(os.path.join(
            current_density_path,
            f"{time:.2f}".replace(".", "_") + ".png"
        ))
        plt.close()

    for t in xp.arange(0, Tend, 5):
        current_1D(t)