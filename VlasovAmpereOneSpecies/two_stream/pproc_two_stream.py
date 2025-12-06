import params_two_stream as damping_params

import os
import cunumpy as xp
import h5py
from psydac.ddm.mpi import mpi as MPI
from matplotlib import pyplot as plt
from struphy import main

### Electric field progression ###
# get parameters
dt = damping_params.time_opts.dt
algo = damping_params.time_opts.split_algo
Nel = damping_params.grid.Nel
p = damping_params.derham_opts.p

env = damping_params.env
ppc = damping_params.loading_params.ppc

# get scalar data (post processing not needed for scalar data)
if MPI.COMM_WORLD.Get_rank() == 0:
    pa_data = os.path.join(env.path_out, "data")
    with h5py.File(os.path.join(pa_data, "data_proc0.hdf5"), "r") as f:
        time = f["time"]["value"][()]
        E = f["scalar"]["en_E"][()]
    logE = xp.log10(E)

    # find where time derivative of E is zero
    dEdt = (xp.roll(logE, -1) - xp.roll(logE, 1))[1:-1] / (2.0 * dt)
    zeros = dEdt * xp.roll(dEdt, -1) < 0.0
    maxima_inds = xp.logical_and(zeros, dEdt > 0.0)
    maxima = logE[1:-1][maxima_inds]
    t_maxima = time[1:-1][maxima_inds]

    # plot
    plt.figure(figsize=(18, 12))
    plt.plot(time, logE, label="numerical")
    plt.legend()
    plt.title(f"{dt=}, {algo=}, {Nel=}, {p=}, {ppc=}")
    plt.xlabel("time [m/c]")
    plt.ylabel("log(E)")
    # plt.plot(t_maxima, maxima, "o-r", markersize=10)

    # plt.savefig("test_weak_Landau")
    plt.show()
      
### Binning distribution progression ###        
# post process raw data
path = os.path.join(os.getcwd(), "sim_data")
main.pproc(path=path)

# get sim data
simdata = main.load_data(path=path)

# plot in e1-v1
e1_bins = simdata.f["kinetic_ions"]["e1_v1"]["grid_e1"]
v1_bins = simdata.f["kinetic_ions"]["e1_v1"]["grid_v1"]

nrows = 4
ntime = len(simdata.f["kinetic_ions"]["e1_v1"]["f_binned"]) 
time_indices = [int( i/(nrows-1) * (ntime - 1) ) for i in range(nrows)]
time_title = ["initial"] + [f"{str(i)}/{str(nrows-1)} th partition" for i in range(1, nrows-1)] + ["final"]

fig, axs = plt.subplots(nrows = nrows, ncols = 2, figsize = (14,10), sharex=True, sharey=True)
for index in range(nrows):
    ax_maxwellian, ax_perturbation = axs[index][0], axs[index][1]
    time_index = time_indices[index]

    #maxwellian distribution plot
    color_mapped = simdata.f["kinetic_ions"]["e1_v1"]["f_binned"][time_index].T
    pcm = ax_maxwellian.pcolor(e1_bins,v1_bins, color_mapped)

    ax_maxwellian.set_xlabel("$\eta_1$")
    ax_maxwellian.set_ylabel("$v_x$")
    ax_maxwellian.set_title(time_title[index] + " Maxwellian")
    fig.colorbar(pcm, ax = ax_maxwellian)

    #perturbation plot
    color_mapped = simdata.f["kinetic_ions"]["e1_v1"]["delta_f_binned"][time_index].T
    pcm = ax_perturbation.pcolor(e1_bins, v1_bins, color_mapped)

    ax_perturbation.set_xlabel("$\eta_1$")
    ax_perturbation.set_ylabel("$v_x$")
    ax_perturbation.set_title(time_title[index] + " perturbation")
    fig.colorbar(pcm, ax = ax_perturbation)

plt.tight_layout()
plt.show()