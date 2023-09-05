import os.path
from typing import Optional, Union

import importlib_resources
import numpy as np

import refnx.reflect
import refl1d.model
import refl1d.probe
import refl1d.experiment


def direct_beam_path(inst_or_path: str = 'OFFSPEC',
                     polarised: bool = False) -> str:
    """Returns the filepath of the correct direct beam file for the instrument
    being used

    Args:
        inst_or_path: Either a local filepath or the name of the instrument the
        experiement will be performed on. Valid options are ['OFFSPEC','SURF',
        'POLREF'] for unpolarised, and ['OFFSPEC', 'POLREF'] for polarised.
        Defaults to 'OFFSPEC'
        polarised: If the experiment is polarised. Defaults to False

    Returns:
        str or None: A string of the hogben internal path of the correct direct
        beam file or the local path
    """

    non_pol_instr = {
        'OFFSPEC': 'OFFSPEC_non_polarised_old.dat',
        'SURF': 'SURF_non_polarised.dat',
        'POLREF': 'POLREF_non_polarised.dat',
        'INTER': 'INTER_non_polarised.dat',
    }

    pol_instr = {
        'OFFSPEC': 'OFFSPEC_polarised_old.dat',
        'POLREF': 'POLREF_polarised.dat',
    }

    # Check if the key isn't in the dictionary and assume
    # a local filepath
    if inst_or_path not in (non_pol_instr or pol_instr):
        if os.path.isfile(inst_or_path):
            return inst_or_path
        else:
            msg = 'Please provide an instrument name or a valid local filepath'
            raise FileNotFoundError(str(msg))

    path = importlib_resources.files('hogben.data.directbeams').joinpath(
        non_pol_instr[inst_or_path]
    )

    if polarised:
        path = importlib_resources.files('hogben.data.directbeams').joinpath(
            pol_instr[inst_or_path]
        )

    return path


def simulate_magnetic(
    sample: Union['refnx.reflect.Stucture', 'refl1d.model.Stack'],
    angle_times: np.ndarray,
    scale: float = 1.0,
    bkg: float = 5e-7,
    dq: float = 2,
    mm: bool = True,
    mp: bool = True,
    pm: bool = True,
    pp: bool = True,
    angle_scale: float = 0.3,
    inst_or_path: str = 'OFFSPEC',
) -> tuple:
    """Simulates an experiment of a given magnetic `sample` measured
       over a number of angles.

    Args:
        sample: sample structure to simulate.
        angle_times: array of angle, number of data points and time to measure
        for each angle to simulate. e.g. [(0.7, 100, 5), (2.0, 100, 20)]
        scale: experimental scale factor.
        bkg: level of instrument background noise.
        dq: instrument resolution.
        pp: whether to simulate "plus plus" spin state.
        pm: whether to simulate "plus minus" spin state.
        mp: whether to simulate "minus plus" spin state.
        mm: whether to simulate "minus minus" spin state.
        angle_scale: angle to use when scaling directbeam flux.
        inst_or_path: dictionary entry of direct beam stored in hogben or
        filepath of directbeam file to simulate with..

    Returns:
        tuple: model and simulated data for the given `sample`.

    """
    models, datasets = [], []

    direct_beam = direct_beam_path(inst_or_path, polarised=False)

    if not direct_beam:  # Local filepath provided so direct_beam_path==None
        direct_beam = inst_or_path

    # Simulate the "minus minus" spin state if requested.
    if mm:
        model, data = simulate(sample, angle_times, scale, bkg, dq,
                               direct_beam, angle_scale, 0)
        models.append(model)
        datasets.append(data)

    # Simulate the "minus plus" spin state if requested.
    if mp:
        model, data = simulate(sample, angle_times, scale, bkg, dq,
                               direct_beam, angle_scale, 1)
        models.append(model)
        datasets.append(data)

    # Simulate the "plus minus" spin state if requested.
    if pm:
        model, data = simulate(sample, angle_times, scale, bkg, dq,
                               direct_beam, angle_scale, 2)
        models.append(model)
        datasets.append(data)

    # Simulate the "plus plus" spin state if requested.
    if pp:
        model, data = simulate(sample, angle_times, scale, bkg, dq,
                               direct_beam, angle_scale, 3)
        models.append(model)
        datasets.append(data)

    return models, datasets


def simulate(sample: Union['refnx.reflect.Stucture', 'refl1d.model.Stack'],
             angle_times: np.ndarray,
             scale: float = 1.0,
             bkg: float = 5e-6,
             dq: float = 2,
             inst_or_path: str = 'OFFSPEC',
             angle_scale: float = 0.3,
             spin_state: Optional[str] = None) -> tuple:
    """Simulates an experiment of a `sample` measured over a number of angles.

    Args:
        sample: sample to simulate an experiment for.
        angle_times: array of angle, number of data points and time to measure
        for each angle to simulate. e.g. [(0.7, 100, 5), (2.0, 100, 20)]
        scale: experimental scale factor.
        bkg: level of instrument background noise.
        dq: instrument resolution in percentage dQ/Q.
        inst_or_path: dictionary entry of direct beam stored in hogben or
        filepath of directbeam file to simulate with.
        angle_scale: angle to use when scaling directbeam flux.
        spin_state: spin state to simulate if given a magnetic sample.

    Returns:
        tuple: model and simulated data for the given `sample`.
    """

    # If there are no angles to measure, return no model and empty data.
    if not angle_times:
        return None, np.zeros((0, 4))

    direct_beam = direct_beam_path(inst_or_path, polarised=False)

    # Iterate over each angle to simulate.
    q, r, dr, counts = [], [], [], []
    total_points = 0
    for angle, points, time in angle_times:
        # Simulate the experiment.
        total_points += points
        simulated = _run_experiment(sample, angle, points, time,
                                    scale, bkg, dq,
                                    direct_beam, angle_scale, spin_state)

        # Combine the data for the angle with the data from previous angles.
        q.append(simulated[0])
        r.append(simulated[1])
        dr.append(simulated[2])
        counts.append(simulated[3])

    # Create a matrix with all of the simulated data.
    data = np.zeros((total_points, 4))
    data[:, 0] = np.concatenate(q)
    data[:, 1] = np.concatenate(r)
    data[:, 2] = np.concatenate(dr)
    data[:, 3] = np.concatenate(counts)

    data = data[(data != 0).all(1)]  # Remove points of zero reflectivity.
    data = data[data[:, 0].argsort()]  # Sort by Q.

    # If there is no data after removing zeros, return no model.
    if len(data) == 0:
        return None, np.zeros((0, 4))

    # If a refnx sample was given, create a refnx ReflectModel.
    if isinstance(sample, refnx.reflect.Structure):
        model = refnx.reflect.ReflectModel(sample, scale=scale, bkg=bkg, dq=dq)

    # If a Refl1D sample was given, create a Refl1D Experiment.
    elif isinstance(sample, refl1d.model.Stack):
        # Create the experiment object.
        q, r, dr = data[:, 0], data[:, 1], data[:, 2]
        model = refl1d_experiment(sample, q, scale, bkg, dq, spin_state)

        # Record the data.
        model.probe.dq = dq
        if sample.ismagnetic:
            model.probe.xs[spin_state].R = r
            model.probe.xs[spin_state].dR = dr
            model.probe.spin_state = spin_state
        else:
            model.probe.R = r
            model.probe.dR = dr

    # Otherwise, the sample must be invalid.
    else:
        raise RuntimeError('invalid sample given')

    return model, data


def _run_experiment(
    sample: Union['refnx.reflect.Stucture', 'refl1d.model.Stack'],
    angle: float,
    points: int,
    time: float,
    scale: float,
    bkg: float,
    dq: float,
    directbeam_path: str,
    angle_scale: float,
    spin_state: int,
) -> tuple:
    """Simulates a single angle measurement of a given `sample`.

    Args:
        sample: sample to simulate.
        angle: angle to simulate.
        points: number of points to use for simulated data.
        time: counting time for simulation.
        scale: experimental scale factor.
        bkg: level of instrument background noise.
        dq: instrument resolution.
        directbeam_path: path to directbeam file to simulate with.
        angle_scale: angle to use when scaling directbeam flux.
        spin_state: spin state to simulate if given a magnetic sample.

    Returns:
        tuple: simulated Q, R, dR data and incident neutron counts.

    """
    # Load the directbeam file.
    direct_beam = np.loadtxt(directbeam_path, delimiter=',')

    wavelengths = direct_beam[:, 0]  # 1st column is wavelength, 2nd is flux.

    # Adjust flux by measurement angle.
    direct_flux = direct_beam[:, 1] * pow(angle / angle_scale, 2)

    # Calculate Q values.
    q = 4 * np.pi * np.sin(np.radians(angle)) / wavelengths

    # Bin Q's' in equally geometrically-spaced bins using flux as weighting.
    q_bin_edges = np.geomspace(min(q), max(q), points + 1)
    flux_binned, _ = np.histogram(q, q_bin_edges, weights=direct_flux)

    # Get the bin centres.
    q_binned = np.asarray(
        [(q_bin_edges[i] + q_bin_edges[i + 1]) / 2 for i in range(points)]
    )

    # Calculate the model reflectivity at each Q point.
    if isinstance(sample, refnx.reflect.Structure):
        # Create a refnx ReflectModel if the sample was defined in refnx.
        model = refnx.reflect.ReflectModel(sample, scale=scale, bkg=bkg, dq=dq)
        r_model = model(q_binned)

    elif isinstance(sample, refl1d.model.Stack):
        # Create a Refl1D experiment if the sample was defined in Refl1D.
        experiment = refl1d_experiment(sample, q_binned, scale,
                                       bkg, dq, spin_state)
        r_model = reflectivity(q_binned, experiment)

    # Otherwise the given sample must be invalid.
    else:
        raise RuntimeError('invalid sample given')

    # Calculate the number of incident neutrons for each bin.
    counts_incident = flux_binned * time

    # Get the measured reflected count for each bin.
    # r_model accounts for background.
    counts_reflected = np.random.poisson(r_model * counts_incident).astype(
        float)

    # Convert from count space to reflectivity space.
    # Point has zero reflectivity if there is no flux.
    r_noisy = np.divide(counts_reflected, counts_incident,
                        out=np.zeros_like(counts_reflected),
                        where=counts_incident != 0)

    r_error = np.divide(np.sqrt(counts_reflected), counts_incident,
                        out=np.zeros_like(counts_reflected),
                        where=counts_incident != 0)

    return q_binned, r_noisy, r_error, counts_incident


def reflectivity(q: np.ndarray, model: Union['refnx.reflect.ReflectModel',
                                             'refl1d.experiment.Experiment']
                 ) -> np.ndarray:
    """Calculates the reflectance of a `model` at given `q` points.

    Args:
        q: Q points to calculate reflectance at.
        model: model structure to calculate reflectivity from.

    Returns:
        numpy.ndarray: reflectivity for each Q point.

    """
    # If there are no data points, return an empty list.
    if len(q) == 0:
        return []

    # Calculate the reflectance in either refnx or Refl1D.
    if isinstance(model, refnx.reflect.ReflectModel):
        return model(q)

    if isinstance(model, refl1d.experiment.Experiment):
        # If magnetic, use the correct spin state.
        if model.sample.ismagnetic:
            # Get the probe for the spin state to simulate.
            probe = model.probe.xs[model.probe.spin_state]
            scale, bkg, dq = probe.intensity, probe.background, probe.dq

            experiment = refl1d_experiment(model.sample, q,
                                           scale, bkg, dq,
                                           model.probe.spin_state)

            return experiment.reflectivity()[model.probe.spin_state][1]

        # Otherwise, the sample is not magnetic.
        else:
            scale = model.probe.intensity
            bkg = model.probe.background
            dq = model.probe.dq

            experiment = refl1d_experiment(model.sample, q, scale, bkg, dq)
            return experiment.reflectivity()[1]


def refl1d_experiment(sample: 'refl1d.model.stack',
                      q_array: np.ndarray,
                      scale: float,
                      bkg: float,
                      dq: float,
                      spin_state: Optional[int] = None,
                      ) -> refl1d.experiment.Experiment:
    """Creates a Refl1D experiment for a given `sample` and `q_array`.

    Args:
        sample: sample to create an experiment for.
        q_array: Q points to use in the experiment.
        scale: experimental scale factor.
        bkg: level of instrument background noise.
        dq: instrument resolution.
        spin_state: spin state to simulate if given a magnetic sample.

    Returns:
        refl1d.experiment.Experiment: experiment for the given `sample`.

    """
    # Transform the resolution from refnx to Refl1D format.
    refl1d_dq = dq / (100 * np.sqrt(8 * np.log(2)))

    # Calculate the dq array and use it to define a QProbe.
    dq_array = q_array * refl1d_dq
    probe = refl1d.probe.QProbe(q_array, dq_array,
                                intensity=scale, background=bkg)
    probe.dq = dq

    # Adjust probe calculation for constant dQ/Q resolution.
    argmin, argmax = np.argmin(q_array), np.argmax(q_array)
    probe.calc_Qo = np.linspace(
        q_array[argmin] - 3.5 * dq_array[argmin],
        q_array[argmax] + 3.5 * dq_array[argmax],
        21 * len(q_array),
    )

    # If the sample is magnetic, create a polarised QProbe.
    if sample.ismagnetic:
        probes = [None] * 4
        probes[spin_state] = probe
        probe = refl1d.probe.PolarizedQProbe(xs=probes, name='')
        probe.spin_state = spin_state

    return refl1d.experiment.Experiment(probe=probe, sample=sample)
