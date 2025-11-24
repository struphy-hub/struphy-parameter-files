import params_weak_Landau_damping as damping_params

import os
import cunumpy as xp
import h5py
from psydac.ddm.mpi import mpi as MPI
from matplotlib import pyplot as plt

do_plot = True

#get parameters
dt = damping_params.time_opts.dt
algo = damping_params.time_opts.split_algo
Nel = damping_params.grid.Nel
p = damping_params.derham_opts.p

env = damping_params.env
ppc = damping_params.loading_params.ppc

# post processing not needed for scalar data

# get scalar data
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
    if do_plot:
        plt.figure(figsize=(18, 12))
        plt.plot(time, logE, label="numerical")
        plt.legend()
        plt.title(f"{dt=}, {algo=}, {Nel=}, {p=}, {ppc=}")
        plt.xlabel("time [m/c]")
        plt.ylabel("log(E)")
        plt.plot(t_maxima[:5], maxima[:5], "r")
        plt.plot(t_maxima[:5], maxima[:5], "or", markersize=10)

        plt.savefig("weak_landau_damping_test")
        # plt.show()