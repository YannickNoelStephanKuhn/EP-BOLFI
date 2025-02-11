"""
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
    verbose=False,
    logging_file=None,
):
    """
    Processes the model and returns a runnable solver.

    :param model:
        A PyBaMM model. Use one of the models in this folder.
    :param parameters:
        The parameters that the model requires as a dictionary.
        Please refer to models.standard_parameters for the names
        or adapt one of the examples in parameters.models.
    :param submesh_types:
        A dictionary of the meshes to be used. The keys have to
        match the geometry names in the model. Use
        #spectral_mesh_and_method as reference or a shortcut.
    :param var_pts:
        A dictionary giving the number of discretization volumes.
        Since the keys have to be special variables determined by
        PyBaMM, use #auto_var_pts as a shortcut.
    :param spatial_methods:
        A dictionary of the spatial methods to be used. The keys
        have to match the geometry names in the model. Use
        #spectral_mesh_and_method as reference or a shortcut.
    :param geometry:
        The geometry of the model in dictionary form. Usually,
        model.default_geometry is sufficient, which is the default.
    :param reltol:
        The relative tolerance that the Casadi solver shall use.
        Default is 1e-6.
    :param abstol:
        The absolute tolerance that the Casadi solver shall use.
        Default is 1e-6.
    :param root_tol:
        The tolerance for rootfinding that the Casadi solver shall use.
        Default is 1e-3.
    :param dt_max:
        The maximum timestep size for the Casadi solver in seconds.
        Default is chosen by PyBaMM.
    :param free_parameters:
        A list of parameter names that shall be input later. They may be
        given to the returned lambda function as a dictionary with the
        names as keys and the values of the parameters as values.
        DO NOT USE GEOMETRICAL PARAMETERS, THEY WILL CRASH THE MESH.
        Instead, just use this function with a complete set of
        parameters where the relevant parameters are changed.
    :param verbose:
        The default (False) sets the PyBaMM flag to only
        show warnings. True will show the details of preprocessing
        and the runtime of the solver. This applies globally, so
        don't set this to True if running simulations in parallel.
    :param logging_file:
        Optional name of a file to store the logs in.
    :returns:
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
    # The logging callback copies the current level of pybamm.logger.
    if logging_file:
        callback = pybamm.callbacks.LoggingCallback(logging_file)

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
    extra_options = {
        "disable_internal_warnings": True, "newton_scheme": "tfqmr"
    }
    solver = pybamm.CasadiSolver(
        rtol=float(reltol),
        atol=float(abstol),
        root_tol=float(root_tol),
        dt_max=dt_max,
        mode="safe",
        max_step_decrease_count=10,
        return_solution_if_failed_early=True,
        extra_options_setup=extra_options
    )

    if logging_file:
        return lambda t_eval, inputs={}, callbacks=callback: solver.solve(
            model, t_eval, inputs=inputs, callbacks=callbacks,
        )
    else:
        return lambda t_eval, inputs={}: solver.solve(
            model, t_eval, inputs=inputs,
        )


def simulation_setup(
    model,
    operation_input,
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
    verbose=False,
    logging_file=None,
):
    """
    Processes the model and returns a runnable solver.

    In contrast to ``solversetup.solver_setup``, this allows for a more
    verbose description of the operating conditions and should be
    preferred.

    :param model:
        A PyBaMM model. Use one of the models in this folder.
    :param operation_input:
        A list of strings which describe the operating conditions. These
        exactly match the PyBaMM usage for pybamm.Experiment. Examples:
        "Hold at 4 V for 60 s", "Discharge at 1 A for 30 s",
        "Rest for 1800 s", "Charge at 1 C for 30 s".
    :param parameters:
        The parameters that the model requires as a dictionary.
        Please refer to models.standard_parameters for the names
        or adapt one of the examples in parameters.models.
    :param submesh_types:
        A dictionary of the meshes to be used. The keys have to
        match the geometry names in the model. Use
        #spectral_mesh_and_method as reference or a shortcut.
    :param var_pts:
        A dictionary giving the number of discretization volumes.
        Since the keys have to be special variables determined by
        PyBaMM, use #auto_var_pts as a shortcut.
    :param spatial_methods:
        A dictionary of the spatial methods to be used. The keys
        have to match the geometry names in the model. Use
        #spectral_mesh_and_method as reference or a shortcut.
    :param geometry:
        The geometry of the model in dictionary form. Usually,
        model.default_geometry is sufficient, which is the default.
    :param reltol:
        The relative tolerance that the Casadi solver should use.
        Default is 1e-6.
    :param abstol:
        The absolute tolerance that the Casadi solver should use.
        Default is 1e-6.
    :param root_tol:
        The tolerance for rootfinding that the Casadi solver should use.
        Default is 1e-3.
    :param dt_max:
        The maximum timestep size for the Casadi solver in seconds.
        Default is chosen by PyBaMM.
    :param free_parameters:
        A list of parameter names that shall be input later. They may be
        given to the solve() function of the returned Simulation object
        as a dictionary to the keyword parameter "inputs" with the
        names as keys and the values of the parameters as values.
        DO NOT USE GEOMETRICAL PARAMETERS, THEY WILL CRASH THE MESH.
        Instead, just use this function with a complete set of
        parameters where the relevant parameters are changed.
    :param verbose:
        The default (False) sets the PyBaMM flag to only
        show warnings. True will show the details of preprocessing
        and the runtime of the solver. This applies globally, so
        don't set this to True if running simulations in parallel.
    :param logging_file:
        Optional name of a file to store the logs in.
    :returns:
        A 2-tuple of a pybamm.Simulation.solve call that runs the
        simulation when called, and the proper callback for the logging
        file if specified (else None).
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
    # The logging callback copies the current level of pybamm.logger.
    if logging_file:
        callback = pybamm.callbacks.LoggingCallback(logging_file)
        if verbose:
            callback.logger.setLevel("INFO")
        else:
            callback.logger.setLevel("WARNING")

    # Declare the free parameters as inputs.
    parameters = dict(deepcopy(parameters))
    parameters.update({
        name: "[input]" for name in free_parameters
    })

    experiment = pybamm.Experiment(operation_input)
    # Use the PyBaMM 0.3.0 compatibility setup routine for experiments.
    # experiment.use_simulation_setup_type = 'old'

    extra_options = {
        "disable_internal_warnings": True, "newton_scheme": "tfqmr"
    }
    simulator = pybamm.Simulation(
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
            dt_max=dt_max,
            return_solution_if_failed_early=True,
            extra_options_setup=extra_options,
        )
    )

    if logging_file:
        return simulator.solve, callback
    else:
        return simulator.solve, None


def auto_var_pts(x_n, x_s, x_p, r_n, r_p, y=1, z=1, R_n=1, R_p=1):
    """
    Utility function for setting the discretization density.

    :param x_n:
        The number of voxels for the electrolyte in the anode.
    :param x_s:
        The number of voxels for the electrolyte in the separator.
    :param x_p:
        The number of voxels for the electrolyte in the cathode.
    :param r_n:
        The number of voxels for each anode representative particle.
    :param r_p:
        The number of voxels for each cathode representative particle.
    :param y:
        Used by PyBaMM for spatially resolved current collectors.
        Don't change the default (1) unless the model supports it.
    :param z:
        Used by PyBaMM for spatially resolved current collectors.
        Don't change the default (1) unless the model supports it.
    :param R_n:
        Used by PyBaMM for particle size distributions.
        Don't change the default (1) unless the model supports it.
    :param R_p:
        Used by PyBaMM for particle size distributions.
        Don't change the default (1) unless the model supports it.
    :returns:
        A discretization dictionary that can be used with PyBaMM models.
    """

    var = pybamm.standard_spatial_vars
    return {
        var.x_n: x_n,
        var.x_s: x_s,
        var.x_p: x_p,
        var.r_n: r_n,
        var.r_p: r_p,
        var.r_n_prim: r_n,
        var.r_p_prim: r_p,
        var.r_n_sec: r_n,
        var.r_p_sec: r_p,
        var.y: y,
        var.z: z,
        var.R_n: R_n,
        var.R_p: R_p,
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
    """
    Utility function for default mesh and spatial methods.

    Only returns Spectral Volume mesh and spatial methods.

    :param order_s_n:
        The order of the anode particles Spectral Volumes.
    :param order_s_p:
        The order of the anode particles Spectral Volumes.
    :param order_e:
        The order of the anode, separator and cathode electrolyte
        Spectral Volumes. These have to be the same, since the
        corresponding meshes get concatenated.
    :param volumes_e_n:
        The # of Spectral Volumes to use for the anode electrolyte.
        This is useful to have different resolutions for each part.
    :param volumes_e_s:
        The # of Spectral Volumes to use for the separator electrolyte.
    :param volumes_e_p:
        The # of Spectral Volumes to use for the cathode electrolyte.
    :param halfcell:
        Default is False, which sets up the mesh and spatial methods for
        a full-cell setup. Set it to True for a half-cell setup.
    :returns:
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
    if halfcell:
        submesh_types["negative electrode"] = pybamm.MeshGenerator(
            pybamm.Uniform1DSubMesh,
        )
        submesh_types["negative particle"] = pybamm.MeshGenerator(
            pybamm.Uniform1DSubMesh,
        )
    else:
        submesh_types["negative electrode"] = pybamm.MeshGenerator(
            pybamm.SpectralVolume1DSubMesh,
            {"order": order_e}
        )
        submesh_types["negative particle"] = pybamm.MeshGenerator(
            pybamm.SpectralVolume1DSubMesh,
            {"order": order_s_n}
        )
        submesh_types["negative primary particle"] = pybamm.MeshGenerator(
            pybamm.SpectralVolume1DSubMesh,
            {"order": order_s_n}
        )
        submesh_types["negative secondary particle"] = pybamm.MeshGenerator(
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
    if halfcell:
        spatial_methods["negative electrode"] = (
            pybamm.FiniteVolume()
        )
        spatial_methods["negative particle"] = (
            pybamm.FiniteVolume()
        )
    else:
        spatial_methods["negative electrode"] = pybamm.SpectralVolume(
            order=order_e
        )
        spatial_methods["negative particle"] = pybamm.SpectralVolume(
            order=order_s_n
        )
        spatial_methods["negative primary particle"] = pybamm.SpectralVolume(
            order=order_s_n
        )
        spatial_methods["negative secondary particle"] = pybamm.SpectralVolume(
            order=order_s_n
        )
    return (submesh_types, var_pts, spatial_methods)
