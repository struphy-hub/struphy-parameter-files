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
from struphy.models.hybrid import LinearMHDDriftkineticCC

# environment options
env = EnvironmentOptions(
    out_folders='/tokp/work/bkna/tae', 
    sim_folder='tae_hybrid_nonlinear_sonic_dt0125', 
    save_step=8, 
    max_runtime=2840,
    restart=False)

# units
base_units = BaseUnits(n=0.2)

# time stepping
time_opts = Time(dt=0.125, Tend=2000., split_algo='Strang')

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
model = LinearMHDDriftkineticCC(
    turn_off = (#'PushGuidingCenterBxEstar',
                #'PushGuidingCenterParallel',
                #'ShearAlfvenCurrentCoupling5D',
                #'Magnetosonic',
                #'CurrentCoupling5DDensity',
                #'CurrentCoupling5DGradB',
                #'CurrentCoupling5DCurlb',
                )
)

# species parameters
model.mhd.set_phys_params()
model.energetic_ions.set_phys_params(
    mass_number=2.,
)

loading_params = LoadingParameters(
    ppc=100,
    moments=(0., 0., 0.89740839, 0.89740839),
    seed=1234,
    )
weights_params = WeightsParameters(
    control_variate=True,
)
boundary_params = BoundaryParameters(
    bc=("remove", "periodic", "periodic"),
    bc_refill=("outer", "inner"),
)
model.energetic_ions.set_markers(loading_params=loading_params,
                                 weights_params=weights_params,
                                 boundary_params=boundary_params,
                                 bufsize=1.,
                                 n_cols_diag=3,
                                 n_cols_aux=5,
                                 )
model.energetic_ions.set_sorting_boxes(boxes_per_dim=None)
model.energetic_ions.set_save_data()

# calculate constant
ep_scale = model.energetic_ions.var.species.mass_number / model.mhd.mass_number

# params else
from struphy.pic.accumulation.filter import FilterParameters
from struphy.linear_algebra.solver import SolverParameters, DiscreteGradientSolverParameters
filter_params = FilterParameters(use_filter="hybrid",)
solver_params = SolverParameters()

# propagator options
model.propagators.push_bxe.options = model.propagators.push_bxe.Options(
                        algo = 'explicit',
                        b_tilde = model.em_fields.b_field,)
model.propagators.push_parallel.options = model.propagators.push_parallel.Options(
                        algo = 'explicit',
                        b_tilde = model.em_fields.b_field,)
model.propagators.shearalfen_cc5d.options = model.propagators.shearalfen_cc5d.Options(
                        filter_params=filter_params,
                        ep_scale=ep_scale,
                        energetic_ions = model.energetic_ions.var,
                        solver_params=solver_params,
                        nonlinear=True,)
model.propagators.magnetosonic.options = model.propagators.magnetosonic.Options(
                        b_field=model.em_fields.b_field,)
model.propagators.cc5d_density.options = model.propagators.cc5d_density.Options(
                        filter_params=filter_params,
                        ep_scale=ep_scale,
                        energetic_ions = model.energetic_ions.var,
                        b_tilde = model.em_fields.b_field,)
model.propagators.cc5d_gradb.options = model.propagators.cc5d_gradb.Options(
                        filter_params=filter_params,
                        ep_scale=ep_scale,
                        b_tilde = model.em_fields.b_field,)
model.propagators.cc5d_curlb.options = model.propagators.cc5d_curlb.Options(
                        filter_params=filter_params,
                        ep_scale=ep_scale,
                        b_tilde = model.em_fields.b_field,)

# background, perturbations and initial conditions
model.mhd.velocity.add_background(FieldsBackground(
    type="LogicalConst",
    values=(0., 0., 0.)
))
model.mhd.velocity.add_perturbation(
    perturbations.TorusModesSin(
        given_in_basis="2",
        amps=(0.001,0.001),
        ms=(10,11),
        ns=(-1,-1),
        pfuns=('exp','exp'),
        pfun_params=((0.44444444,0.1),(0.44444444,0.1)),
        comp=0)
)
model.mhd.velocity.add_perturbation(
     perturbations.TorusModesCos(
        given_in_basis="2",
        amps=(0.000015915494309189535,0.000014468631190172303),
        ms=(10,11),
        ns=(-1,-1),
        pfuns=('d_exp','d_exp'),
        pfun_params=((0.44444444,0.1),(0.44444444,0.1)),
        comp=1),
)
n_profile = perturbations.ITPA_density(
    n0=0.00720652,
    c=(0.47023, 0.20323, 0.13273, 0.521298)
)
maxwellian_1 = maxwellians.CanonicalMaxwellian2D(
    vth=(0.89740839, None),
    equil=equil,
    volume_form=True,
    n=(0., n_profile),
)
background = maxwellian_1
model.energetic_ions.var.add_background(
    background=background,
    n_as_volume_form=False)

# optional: exclude variables from saving
# model.energetic_ions.var.save_data = False

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
