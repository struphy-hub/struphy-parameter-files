"""
Parameter file of strong Landau damping implemented using VlasovAmpereOneSpecies model
(ModeCos perturbation amplitude = 0.5)
"""

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

# environment options
env = EnvironmentOptions(sim_folder="sim_data")

# units
base_units = BaseUnits()

# time stepping
time_opts = Time(dt = 0.05, Tend = 20.0, split_algo = "LieTrotter")

# geometry
domain = domains.Cuboid(r1 = 12.56)

# fluid equilibrium (can be used as part of initial conditions)
equil = None

# grid
grid = grids.TensorProductGrid(Nel = (32,1,1))

# derham options
derham_opts = DerhamOptions(p = (3,1,1))

# light-weight model instance
model = VlasovAmpereOneSpecies()

# species parameters
model.kinetic_ions.set_phys_params(alpha=1.0, epsilon=-1.0)

loading_params = LoadingParameters(ppc = 1000)
weights_params = WeightsParameters(control_variate= True)
boundary_params = BoundaryParameters()
model.kinetic_ions.set_markers(loading_params=loading_params,
                               weights_params=weights_params,
                               boundary_params=boundary_params,
                               bufsize = 0.4,)
model.kinetic_ions.set_sorting_boxes(boxes_per_dim=(16, 1, 1), do_sort=True)

binplot = BinningPlot(slice='e1_v1', n_bins= (128, 128), ranges= ((0.,1.), (-5.,5.)))
model.kinetic_ions.set_save_data(binning_plots=(binplot,))

# propagator options
model.propagators.push_eta.options = model.propagators.push_eta.Options() 
if model.with_B0:
    model.propagators.push_vxb.options = model.propagators.push_vxb.Options()

model.propagators.coupling_va.options = model.propagators.coupling_va.Options()
model.initial_poisson.options = model.initial_poisson.Options(stab_mat="M0")

# background and initial conditions
background = maxwellians.Maxwellian3D(n=(1.0, None))
model.kinetic_ions.var.add_background(background)

# if .add_initial_condition is not called, the background is the kinetic initial condition
perturbation = perturbations.ModesCos(amps = (0.5,), ls = (1,))
init = maxwellians.Maxwellian3D(n = (1.0,perturbation))
model.kinetic_ions.var.add_initial_condition(init)

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