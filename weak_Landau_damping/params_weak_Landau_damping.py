#TODO: find Z0, em_field: options: solver: maxwell
"""
Parameter file of weak Landau damping implemented using VlasovAmpereOneSpecies model
"""
import os

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
from struphy.models.kinetic import VlasovAmpereOneSpecies

from struphy.linear_algebra.solver import SolverParameters #import class to set solver Parameters
from struphy.ode.utils import ButcherTableau #import class to set ode method

# environment options
output_folders = os.path.join(os.getcwd())
env = EnvironmentOptions(out_folders=output_folders, sim_folder="weak_Landau")

# units
base_units = BaseUnits(x = 1.0, B = 1.0, n = 1.0)

# time stepping
time_opts = Time(dt = 0.05, Tend = 75.0, split_algo = "LieTrotter")

# geometry
domain = domains.Cuboid(
    l1 = 0., r1 = 12.56,
    l2 = 0., r2 = 1.,
    l3 = 0., r3 = 1.
)

# fluid equilibrium (can be used as part of initial conditions)
equil = equils.HomogenSlab()

# grid
grid = grids.TensorProductGrid(Nel = [32,1,1],mpi_dims_mask=[True,True,True])

# derham options
derham_opts = DerhamOptions(
    p = [1,1,1], spl_kind=[True,True,True],dirichlet_bc=None, nquads=[2,2,1],
    nq_pr = [2,2,1], polar_ck = -1
    )

# light-weight model instance
model = VlasovAmpereOneSpecies(with_B0 = False)

# species parameters
model.kinetic_ions.set_phys_params(mass_number= 1, charge_number= 1, epsilon = 0.25, kappa= 1.)

loading_params = LoadingParameters(ppc = 10000,Np=99,
                                   loading = "pseudo_random",seed = None, 
                                   spatial = "uniform",
                                   dir_particles = "path_to_particles"
                                   ,moments=[0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
weights_params = WeightsParameters(control_variate= True)
boundary_params = BoundaryParameters(bc = ("periodic","periodic","periodic"))
model.kinetic_ions.set_markers(loading_params=loading_params,
                               weights_params=weights_params,
                               boundary_params=boundary_params,
                               )
model.kinetic_ions.set_sorting_boxes()

binplot = BinningPlot(slice='e1_v1', n_bins= (128, 128), ranges= ((0.,1.), (-5.,5.)))
model.kinetic_ions.set_save_data(binning_plots=(binplot,),n_markers=3)

# propagator options

model.propagators.push_eta.options = model.propagators.push_eta.Options(ButcherTableau(algo = "rk4")) 
if model.with_B0:
    model.propagators.push_vxb.options = model.propagators.push_vxb.Options()

# NOTE: matches default parameter
model.propagators.coupling_va.options = model.propagators.coupling_va.Options( #XXX: change coupling type?
    solver = "pcg", precond = "MassMatrixPreconditioner", 
    solver_params = SolverParameters(tol = 1.0e-08, maxiter = 3000, info = False, verbose = False)
)
# NOTE: matches default parameter
model.initial_poisson.options = model.initial_poisson.Options(
    solver = "pcg", precond = "MassMatrixPreconditioner",
    solver_params = SolverParameters(tol = 1.0e-08, maxiter = 3000, info = False, verbose = False)
)

# background, perturbations and initial conditions

# XXX: Should perturbation be separated from background?
background = maxwellians.Maxwellian3D(n = (1.,None)) # XXX: equivalent to Maxwellian6D?
model.kinetic_ions.var.add_background(background)

# if .add_initial_condition is not called, the background is the kinetic initial condition
# XXX: Unmentioned initial condition in .yml
perturbation = perturbations.ModesCos(comp = 0, amps = (0.001,), ls = (1,))
init = maxwellians.Maxwellian3D(n=(1.,perturbation))
model.kinetic_ions.var.add_initial_condition(init = init)


# optional: exclude variables from saving
# model.kinetic_ions.var.save_data = False

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