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
env = EnvironmentOptions()

# units
base_units = BaseUnits()

# time stepping
time_opts = Time()

# geometry
domain = domains.Cuboid()

# fluid equilibrium (can be used as part of initial conditions)
equil = equils.HomogenSlab()

# grid
grid = grids.TensorProductGrid()

# derham options
derham_opts = DerhamOptions()

# light-weight model instance
model = VlasovAmpereOneSpecies()

# species parameters
model.kinetic_ions.set_phys_params()

loading_params = LoadingParameters()
weights_params = WeightsParameters()
boundary_params = BoundaryParameters()
model.kinetic_ions.set_markers(loading_params=loading_params,
                               weights_params=weights_params,
                               boundary_params=boundary_params,
                               )
model.kinetic_ions.set_sorting_boxes()

binplot = BinningPlot(slice='e1', n_bins=128, ranges=(0.0, 1.0))
model.kinetic_ions.set_save_data(binning_plots=(binplot,))

# propagator options
model.propagators.push_eta.options = model.propagators.push_eta.Options()
if model.with_B0:
    model.propagators.push_vxb.options = model.propagators.push_vxb.Options()
model.propagators.coupling_va.options = model.propagators.coupling_va.Options()
model.initial_poisson.options = model.initial_poisson.Options()

# background, perturbations and initial conditions
model.em_fields.phi.add_background(FieldsBackground())
model.em_fields.phi.add_perturbation(perturbations.TorusModesCos())
maxwellian_1 = maxwellians.Maxwellian3D(n=(1.0, None))
maxwellian_2 = maxwellians.Maxwellian3D(n=(0.1, None))
background = maxwellian_1 + maxwellian_2
model.kinetic_ions.var.add_background(background)

# if .add_initial_condition is not called, the background is the kinetic initial condition
perturbation = perturbations.TorusModesCos()
maxwellian_1pt = maxwellians.Maxwellian3D(n=(1.0, perturbation))
init = maxwellian_1pt + maxwellian_2
model.kinetic_ions.var.add_initial_condition(init)

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