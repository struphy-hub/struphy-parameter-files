import params_bump_on as damping_params

import os
import cunumpy as xp
import h5py
from feectools.ddm.mpi import mpi as MPI
from matplotlib import pyplot as plt
from struphy import main

# post process raw data
path = os.path.join(os.getcwd(), "sim_data")
main.pproc(path=path)

# get sim data
simdata = main.load_data(path=path)

### Initial velocity distribution ###
v1_bins = simdata.f["kinetic_ions"]["v1"]["grid_v1"]
f_v1 = simdata.f["kinetic_ions"]["v1"]["f_binned"]

fig, ax = plt.subplots(1 ,figsize = (14,10))

ax.plot(v1_bins, f_v1[0])
ax.set_xlabel("Velocity v")
ax.set_ylabel("Distribution f(v)")
ax.set_title("Initial velocity distribution")
plt.tight_layout()
plt.show()

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
e1_bins = simdata.f["kinetic_ions"]["e1_v1"]["grid_e1"]
v1_bins = simdata.f["kinetic_ions"]["e1_v1"]["grid_v1"]  
nrows = 3
ncols = 4
ntime = len(simdata.f["kinetic_ions"]["e1_v1"]["f_binned"]) 
time_indices = [int( i/(nrows*ncols-1) * (ntime - 1) ) for i in range(nrows*ncols)]

fig, axs = plt.subplots(nrows = nrows, ncols = ncols, figsize = (14,10), sharex=True, sharey=True)
for i in range(nrows):
    for j in range(ncols):
        ax_maxwellian = axs[i][j]
        time_idx = time_indices[j + i*ncols]

        #maxwellian distribution plot
        color_mapped = simdata.f["kinetic_ions"]["e1_v1"]["f_binned"][time_idx].T
        pcm = ax_maxwellian.pcolor(e1_bins,v1_bins, color_mapped)

        ax_maxwellian.set_xlabel(r"$\eta_1$")
        ax_maxwellian.set_ylabel(r"$v_x$")
        ax_maxwellian.set_title(f"t = {simdata.t_grid[time_idx]}")
        fig.colorbar(pcm, ax = ax_maxwellian)
        
plt.tight_layout()
plt.show()