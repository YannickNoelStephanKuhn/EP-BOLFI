"""!@package models.solversetup
This file eases the setup and simulation of PyBaMM battery models.
"""

import pybamm
from copy import deepcopy
# Reset the PyBaMM colour scheme.
import matplotlib.pyplot as plt
plt.style.use("default")


def solver_setup(
    model,
    parameters,
    submesh_types,
    var_pts,
    spatial_methods,
    geometry=None,
    reltol=1e-6,
    abstol=1e-6,
    root_tol=1e-3,
    dt_max=None,
    free_parameters=[],
    verbose=False
):
    """!@brief Processes the model and returns a runnable solver.

    @par model
        A PyBaMM model. Use one of the models in this folder.
    @par parameters
        The parameters that the model requires as a dictionary.
        Please refer to models.standard_parameters for the names
        or adapt one of the examples in parameters.models.
    @par submesh_types
        A dictionary of the meshes to be used. The keys have to
        match the geometry names in the model. Use
        #spectral_mesh_and_method as reference or a shortcut.
    @par var_pts
        A dictionary giving the number of discretization volumes.
        Since the keys have to be special variables determined by
        PyBaMM, use #auto_var_pts as a shortcut.
    @par spatial_methods
        A dictionary of the spatial methods to be used. The keys
        have to match the geometry names in the model. Use
        #spectral_mesh_and_method as reference or a shortcut.
    @par geometry
        The geometry of the model in dictionary form. Usually,
        model.default_geometry is sufficient, which is the default.
    @par reltol
        The relative tolerance that the Casadi solver shall use.
        Default is 1e-6.
    @par abstol
        The absolute tolerance that the Casadi solver shall use.
        Default is 1e-6.
    @par root_tol
        The tolerance for rootfinding that the Casadi solver shall use.
        Default is 1e-3.
    @par dt_max
        The maximum timestep size for the Casadi solver in seconds.
        Default is chosen by PyBaMM.
    @par free_parameters
        A list of parameter names that shall be input later. They may be
        given to the returned lambda function as a dictionary with the
        names as keys and the values of the parameters as values.
        DO NOT USE GEOMETRICAL PARAMETERS, THEY WILL CRASH THE MESH.
        Instead, just use this function with a complete set of
        parameters where the relevant parameters are changed.
    @par verbose The default (False) sets the PyBaMM flag to only
        show warnings. True will show the details of preprocessing
        and the runtime of the solver. This applies globally, so
        don't set this to True if running simulations in parallel.
    @return
        A lambda function that takes a numpy.array of timepoints to
        evaluate and runs the Casadi solver for those. Optionally takes
        a dictionary of parameters as specified by "free_parameters".
    """

    geometry = geometry or model.default_geometry

    parameters = pybamm.ParameterValues(deepcopy(parameters))

    # Levels of the "logging" module used by PyBaMM:
    # _nameToLevel = {
    #     'CRITICAL': CRITICAL,
    #     'FATAL': FATAL,
    #     'ERROR': ERROR,
    #     'WARN': WARNING,
    #     'WARNING': WARNING,
    #     'INFO': INFO,
    #     'DEBUG': DEBUG,
    #     'NOTSET': NOTSET,
    # }
    if verbose:
        # pybamm.set_logging_level("DEBUG")
        pybamm.set_logging_level("INFO")
    else:
        pybamm.set_logging_level("WARNING")  # "NOTSET"

    # Declare the free parameters as inputs.
    parameters = dict(deepcopy(parameters))
    parameters.update({
        name: "[input]" for name in free_parameters
    })
    parameters = pybamm.ParameterValues(parameters)

    # Load the parameter values and process them into model and
    # geometry.
    parameters.process_model(model)
    parameters.process_geometry(geometry)

    # Set the mesh on which the model is solved.
    mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

    # Discretise the model.
    disc = pybamm.Discretisation(mesh, spatial_methods)
    disc.process_model(model)

    # Initialize the solver.
    solver = pybamm.CasadiSolver(rtol=float(reltol), atol=float(abstol),
                                 root_tol=float(root_tol), dt_max=dt_max)

    return lambda t_eval, inputs={}: solver.solve(
        model, t_eval, inputs=inputs
    )


def simulation_setup(
    model,
    input,
    parameters,
    submesh_types,
    var_pts,
    spatial_methods,
    geometry=None,
    reltol=1e-6,
    abstol=1e-6,
    root_tol=1e-3,
    dt_max=None,
    free_parameters=[],
    verbose=False
):
    """!@brief Processes the model and returns a runnable solver.

    In contrast to solversetup.solver_setup, this allows for a more
    verbose description of the operating conditions and should be
    preferred.

    @par model
        A PyBaMM model. Use one of the models in this folder.
    @par input
        A list of strings which describe the operating conditions. These
        exactly match the PyBaMM usage for pybamm.Experiment. Examples:
        "Hold at 4 V for 60 s", "Discharge at 1 A for 30 s",
        "Rest for 1800 s", "Charge at 1 C for 30 s".
    @par parameters
        The parameters that the model requires as a dictionary.
        Please refer to models.standard_parameters for the names
        or adapt one of the examples in parameters.models.
    @par submesh_types
        A dictionary of the meshes to be used. The keys have to
        match the geometry names in the model. Use
        #spectral_mesh_and_method as reference or a shortcut.
    @par var_pts
        A dictionary giving the number of discretization volumes.
        Since the keys have to be special variables determined by
        PyBaMM, use #auto_var_pts as a shortcut.
    @par spatial_methods
        A dictionary of the spatial methods to be used. The keys
        have to match the geometry names in the model. Use
        #spectral_mesh_and_method as reference or a shortcut.
    @par geometry
        The geometry of the model in dictionary form. Usually,
        model.default_geometry is sufficient, which is the default.
    @par reltol The relative tolerance that the Casadi solver should
        use. Default is 1e-6.
    @par abstol The absolute tolerance that the Casadi solver should
        use. Default is 1e-6.
    @par root_tol The tolerance for rootfinding that the Casadi solver
        should use. Default is 1e-3.
    @par dt_max The maximum timestep size for the Casadi solver in
        seconds. Default is chosen by PyBaMM.
    @par free_parameters
        A list of parameter names that shall be input later. They may be
        given to the solve() function of the returned Simulation object
        as a dictionary to the keyword parameter "inputs" with the
        names as keys and the values of the parameters as values.
        DO NOT USE GEOMETRICAL PARAMETERS, THEY WILL CRASH THE MESH.
        Instead, just use this function with a complete set of
        parameters where the relevant parameters are changed.
    @par verbose The default (False) sets the PyBaMM flag to only
        show warnings. True will show the details of preprocessing
        and the runtime of the solver. This applies globally, so
        don't set this to True if running simulations in parallel.
    @return
        A pybamm.Simulation that runs the simulation when its solve()
        method is called. Please use solve(check_model=False) as it
        won't work properly with redundant model checks in place.
    """

    # Levels of the "logging" module used by PyBaMM:
    # _nameToLevel = {
    #     'CRITICAL': CRITICAL,
    #     'FATAL': FATAL,
    #     'ERROR': ERROR,
    #     'WARN': WARNING,
    #     'WARNING': WARNING,
    #     'INFO': INFO,
    #     'DEBUG': DEBUG,
    #     'NOTSET': NOTSET,
    # }
    if verbose:
        # pybamm.set_logging_level("DEBUG")
        pybamm.set_logging_level("INFO")
    else:
        pybamm.set_logging_level("WARNING")  # "NOTSET"

    # Declare the free parameters as inputs.
    parameters = dict(deepcopy(parameters))
    parameters.update({
        name: "[input]" for name in free_parameters
    })

    experiment = pybamm.Experiment(input)
    # Use the PyBaMM 0.3.0 compatibility setup routine for experiments.
    # experiment.use_simulation_setup_type = 'old'

    return pybamm.Simulation(
        model,
        experiment,
        model.default_geometry,
        pybamm.ParameterValues(parameters),
        submesh_types,
        var_pts,
        spatial_methods,
        pybamm.CasadiSolver(
            rtol=float(reltol),
            atol=float(abstol),
            root_tol=float(root_tol),
            dt_max=dt_max
        )
    )


def auto_var_pts(x_n, x_s, x_p, r_n, r_p, y=1, z=1):
    """!@brief Utility function for setting the discretization density.

    @par x_n
        The number of voxels for the electrolyte in the anode.
    @par x_s
        The number of voxels for the electrolyte in the separator.
    @par x_p
        The number of voxels for the electrolyte in the cathode.
    @par r_n
        The number of voxels for each anode representative particle.
    @par r_p
        The number of voxels for each cathode representative particle.
    @par y
        Used by PyBaMM for spatially resolved current collectors.
        Don't change the default (1) unless the model supports it.
    @par z
        Used by PyBaMM for spatially resolved current collectors.
        Don't change the default (1) unless the model supports it.
    @return
        A discretization dictionary that can be used with PyBaMM models.
    """

    var = pybamm.standard_spatial_vars
    return {
        var.x_n: x_n,
        var.x_s: x_s,
        var.x_p: x_p,
        var.r_n: r_n,
        var.r_p: r_p,
        var.y: y,
        var.z: z,
    }


def spectral_mesh_pts_and_method(
    order_s_n,
    order_s_p,
    order_e,
    volumes_e_n=1,
    volumes_e_s=1,
    volumes_e_p=1,
    halfcell=False
):
    """!@brief Utility function for default mesh and spatial methods.

    Only returns Spectral Volume mesh and spatial methods.
    @par order_s_n
        The order of the anode particles Spectral Volumes.
    @par order_s_p
        The order of the anode particles Spectral Volumes.
    @par order_e
        The order of the anode, separator and cathode electrolyte
        Spectral Volumes. These have to be the same, since the
        corresponding meshes get concatenated.
    @par volumes_e_n
        The # of Spectral Volumes to use for the anode electrolyte.
        This is useful to have different resolutions for each part.
    @par volumes_e_s
        The # of Spectral Volumes to use for the separator electrolyte.
    @par volumes_e_p
        The # of Spectral Volumes to use for the cathode electrolyte.
    @par halfcell
        Default is False, which sets up the mesh and spatial methods for
        a full-cell setup. Set it to True for a half-cell setup.
    @return
        A (submesh_types, spatial_methods) tuple for PyBaMM usage.
    """
    submesh_types = {
        "separator": pybamm.MeshGenerator(
            pybamm.SpectralVolume1DSubMesh,
            {"order": order_e}
        ),
        "positive electrode": pybamm.MeshGenerator(
            pybamm.SpectralVolume1DSubMesh,
            {"order": order_e}
        ),
        "positive particle": pybamm.MeshGenerator(
            pybamm.SpectralVolume1DSubMesh,
            {"order": order_s_p}
        ),
        "current collector": pybamm.SubMesh0D,
    }
    if not halfcell:
        submesh_types["negative electrode"] = pybamm.MeshGenerator(
            pybamm.SpectralVolume1DSubMesh,
            {"order": order_e}
        )
        submesh_types["negative particle"] = pybamm.MeshGenerator(
            pybamm.SpectralVolume1DSubMesh,
            {"order": order_s_n}
        )
    # The potentially superfluous anode pts don't trip up the models.
    # Also, the Spectral Volume method is most effective if the order
    # is increased instead of the amount of Spectral Volumes. That's
    # why the particle domains can be left at one Spectral Volume each.
    var_pts = auto_var_pts(volumes_e_n, volumes_e_s, volumes_e_p, 1, 1)
    spatial_methods = {
        "separator": pybamm.SpectralVolume(order=order_e),
        "positive electrode": pybamm.SpectralVolume(order=order_e),
        "positive particle": pybamm.SpectralVolume(order=order_s_p),
        "current collector": pybamm.ZeroDimensionalSpatialMethod()
    }
    if not halfcell:
        spatial_methods["negative electrode"] = pybamm.SpectralVolume(
            order=order_e
        )
        spatial_methods["negative particle"] = pybamm.SpectralVolume(
            order=order_s_n
        )
    return (submesh_types, var_pts, spatial_methods)
