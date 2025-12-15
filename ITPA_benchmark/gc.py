from struphy.io.options import EnvironmentOptions, BaseUnits, Time
from struphy.geometry import domains
from struphy.fields_background import equils
from struphy.topology import grids
from struphy.io.options import DerhamOptions
from struphy.io.options import FieldsBackground
from struphy.initial import perturbations
from struphy.kinetic_background import maxwellians
from struphy.pic.utilities import (LoadingParameters,
                                   WeightsParameters,
                                   BoundaryParameters,
                                   BinningPlot,
                                   KernelDensityPlot,
                                   )
from struphy import main

# import model, set verbosity
from struphy.models.toy import GuidingCenter

# environment options
env = EnvironmentOptions(
    out_folders='/tokp/work/bkna/tae', 
    sim_folder='gc', 
    save_step=8, max_runtime=1440)

# units
base_units = BaseUnits(n=0.2)

# time stepping
time_opts = Time(dt=0.25, Tend=200., split_algo='Strang')

# geometry
domain = domains.HollowTorus(a1=0.1, a2=1., R0=10., sfl=False, tor_period=6)

# fluid equilibrium (can be used as part of initial conditions)
equil = equils.AdhocTorus(
    a=1.,
    R0=10.,
    B0=3.,
    q_kind=0,
    q0=1.71,
    q1=1.87,
    n1=0.,
    n2=0.,
    na=1.,
    p_kind=1.,
    p0=1.,
    p1=0.95,
    p2=0.05,
    beta=0.0018,
)

# grid
grid = grids.TensorProductGrid(
    Nel=(24,96,16),
    mpi_dims_mask=(True, True, False)
)

# derham options
derham_opts = DerhamOptions(
    p=(3, 3, 3),
    spl_kind=(False, True, True),
    dirichlet_bc=((True, True), (False, False), (False, False)),
    nq_pr=(6, 6, 6),
    nquads=(6, 6, 6),
    polar_ck=-1,
)

# light-weight model instance
model = GuidingCenter()

# species parameters
model.kinetic_ions.set_phys_params(
    charge_number=2.,
)

loading_params = LoadingParameters(
    ppc=100,
    moments=(0., 0., 0.89740839, 0.89740839)
    )
weights_params = WeightsParameters(
    control_variate=True,
)
boundary_params = BoundaryParameters(
    bc=("remove", "periodic", "periodic"),
    bc_refill=("outer", "inner"),
)
model.kinetic_ions.set_markers(loading_params=loading_params,
                                 weights_params=weights_params,
                                 boundary_params=boundary_params,
                                 bufsize=1.,
                                 n_cols_diag=3,
                                 n_cols_aux=5,
                                 )
model.kinetic_ions.set_sorting_boxes(boxes_per_dim=None)
binning_plots = (
    BinningPlot(
    slice="e1",
    n_bins=128,
    ranges=(0., 1.),
    divide_by_jac=True),
    BinningPlot(
    slice="v1",
    n_bins=128,
    ranges=(-2.5, 2.5),
    divide_by_jac=True),
    )
model.kinetic_ions.set_save_data(
    binning_plots=binning_plots,
)

# propagator options
model.propagators.push_bxe.options = model.propagators.push_bxe.Options(
                        algo = 'explicit',)
model.propagators.push_parallel.options = model.propagators.push_parallel.Options(
                        algo = 'explicit',)
# background, perturbations and initial conditions
n_profile = perturbations.ITPA_density(
    n0=0.00720652,
    c=(0.47023, 0.20323, 0.13273, 0.521298)
)
maxwellian_1 = maxwellians.CanonicalMaxwellian2D(
    equil=equil,
    volume_form=True,
    n=(0., n_profile),
)

background = maxwellian_1
model.kinetic_ions.var.add_background(background, n_as_volume_form=False)

if __name__ == "__main__":
    # start run
    verbose = True

    main.run(model,
             params_path=__file__,
             env=env,
             base_units=base_units,
             time_opts=time_opts,
             domain=domain,
             equil=equil,
             grid=grid,
             derham_opts=derham_opts,
             verbose=verbose,
             )