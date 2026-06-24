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
from struphy.models.fluid import LinearMHD

# environment options
env = EnvironmentOptions(
    out_folders='/tokp/work/bkna/tae', 
    sim_folder='linearmhd', 
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
model = LinearMHD()

# species parameters
model.mhd.set_phys_params()

# propagator options
model.propagators.shear_alf.options = model.propagators.shear_alf.Options(
    algo='implicit',
    precond='MassMatrixDiagonalPreconditioner',
)
model.propagators.mag_sonic.options = model.propagators.mag_sonic.Options(b_field=model.em_fields.b_field)

# background, perturbations and initial conditions
model.mhd.velocity.add_background(FieldsBackground(
    type="LogicalConst",
    values=(0., 0., 0.)
))
model.mhd.velocity.add_perturbation(
    perturbations.TorusModesSin(
        given_in_basis="2",
        amps=(0.01,0.01),
        ms=(10,11),
        ns=(-1,-1),
        pfuns=('exp','exp'),
        pfun_params=((0.44444444,0.1),(0.44444444,0.1)),
        comp=0)
)
model.mhd.velocity.add_perturbation(
     perturbations.TorusModesCos(
        given_in_basis="2",
        amps=(0.00015915,0.00015915),
        ms=(10,11),
        ns=(-1,-1),
        pfuns=('d_exp','d_exp'),
        pfun_params=((0.44444444,0.1),(0.44444444,0.1)),
        comp=1),
)

# optional: exclude variables from saving
# model.mhd.pressure.save_data = False

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