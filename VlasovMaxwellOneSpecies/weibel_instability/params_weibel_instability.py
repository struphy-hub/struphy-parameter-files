"""
Parameter file of Weibel instability implemented using VlasovMaxwellOnespecies model
(perturbation's wave number: k = 1.25)

Magnetic field along "z"
Electric field along "x-y"
"""
from struphy import EnvironmentOptions, BaseUnits, Time
from struphy import domains
from struphy import equils
from struphy import grids
from struphy import DerhamOptions
from struphy import FieldsBackground
from struphy import perturbations
from struphy import maxwellians
from struphy import (LoadingParameters,
                                   WeightsParameters,
                                   BoundaryParameters,
                                   BinningPlot,
                                   KernelDensityPlot,
                                   )
from struphy import main


# import model, set verbosity
from struphy.models import VlasovMaxwellOneSpecies
import cunumpy as xp

# setup parameters
k = 1.25
B_pert_amp = -1e-4
dens_pert_amp = 0.0

vth1_background_val = 0.02/xp.sqrt(2)
vth2_background_val = vth1_background_val * xp.sqrt(12)

# environment options
env = EnvironmentOptions(sim_folder="simData_500ppc_perbF_controlVariateF", save_step = 5, max_runtime=xp.inf)

# units
base_units = BaseUnits()

# time stepping
time_opts = Time(dt = 0.05, Tend = 500, split_algo = "LieTrotter")

# geometry
domain = domains.Cuboid(r1 = 2*xp.pi/k)

# fluid equilibrium (can be used as part of initial conditions)
equil = None

# grid
grid = grids.TensorProductGrid(Nel = (32,1,1))

# derham options
derham_opts = DerhamOptions(p = (3,1,1))

# light-weight model instance
model = VlasovMaxwellOneSpecies()
# species parameters
model.kinetic_ions.set_phys_params(alpha = 1, epsilon = 1)

loading_params = LoadingParameters(ppc = 500, set_zero_velocity = (False, False, True), moments = (0.0,0.0,0.0,vth1_background_val,vth2_background_val,1.0))
weights_params = WeightsParameters(control_variate = False)
boundary_params = BoundaryParameters()
model.kinetic_ions.set_markers(loading_params=loading_params,
                               weights_params=weights_params,
                               boundary_params=boundary_params,
                               bufsize = 0.4)
model.kinetic_ions.set_sorting_boxes(boxes_per_dim = (16,1,1), do_sort = True)

binplot_dens = BinningPlot(slice="e1_v1", n_bins= (128, 128), ranges= ((0.,1.), (-0.1,0.1))) 

binplot_current = tuple(
    [BinningPlot(slice=f"e{i}", n_bins= 32, ranges= (0.,1.), output_quantity=f"current_{j}") for j in range(1,4) for i in range(1,4)] 
    )

model.kinetic_ions.set_save_data(binning_plots=(binplot_dens, *binplot_current))

# propagator options
model.propagators.maxwell.options = model.propagators.maxwell.Options()
model.propagators.push_eta.options = model.propagators.push_eta.Options()
model.propagators.push_vxb.options = model.propagators.push_vxb.Options(b2_var=model.em_fields.b_field)
model.propagators.coupling_va.options = model.propagators.coupling_va.Options()
model.initial_poisson.options = model.initial_poisson.Options(stab_mat="M0")

# background, perturbations and initial conditions
model.em_fields.b_field.add_perturbation(perturbation = perturbations.ModesCos(amps=(B_pert_amp,), ls = (1,), comp = 2)) # Initial Bz depending on x-axis

maxwellian = maxwellians.Maxwellian3D(
                    vth1=(vth1_background_val, None) , vth2=(vth2_background_val, None)
                )
model.kinetic_ions.var.add_background(maxwellian)

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
