import os.path
from typing import Optional, Union

import importlib_resources
import numpy as np

import refnx.reflect
import refl1d.model
import refl1d.probe
import refl1d.experiment


class SimulateReflectivity:
    """
    A class for simulating experimental reflectivity data from a refnx or
    refl1D sample structure. It takes a single model, but can simulate a list
    of experimental conditions, e.g. different angles for different times.

    Attributes:
        sample: A refnx Structure or a refl1d Stack
        angle_times: a list of tuples of experimental conditions to simulate,
                    in the order (angle, # of points, time)
        scale: a scale-factor for the dataset (e.g. reflectivity of the
                critical edge), defaults to 1.0
        bkg: background of the measurement, defaults to 1e-6
        dq: resolution of the measurement (assuming constant dq/q as a
            percentage), defaults to 2.0
        inst_or_path: either the name of an instrument already in HOGBEN, or
                      the path to a direct beam file, defaults to 'OFFSPEC'
        angle_scale: the angle at which the direct beam was taken (so that it
                     can be scaled appropriately), defaults to 0.3
        spin_states: a list of the spin states to be simulated if the measurement
                   is polarised, so that the correct direct beam file is taken,
                   defaults to None [mm, mp, pm, pp]
    """

    non_pol_instr_dict = {'OFFSPEC': 'OFFSPEC_non_polarised_old.dat',
                          'SURF': 'SURF_non_polarised.dat',
                          'POLREF': 'POLREF_non_polarised.dat',
                          'INTER': 'INTER_non_polarised.dat'}

    pol_instr_dict = {'OFFSPEC': 'OFFSPEC_polarised_old.dat',
                      'POLREF': 'POLREF_polarised.dat'}

    def __init__(self,
                 sample: Union['refnx.reflect.Stucture', 'refl1d.model.Stack'],
                 angle_times: list[tuple],
                 scale: float = 1.0,
                 bkg: float = 1e-6,
                 dq: float = 2.0,
                 inst_or_path: str = 'OFFSPEC',
                 angle_scale: float = 0.3,
                 spin_states: list = None):

        self.sample = sample
        self.angle_times = angle_times
        self.scale = scale
        self.bkg = bkg
        self.dq = dq
        self.inst_or_path = inst_or_path
        self.spin_states = spin_states
        self.direct_beam_file = self.direct_beam_path()
        self.angle_scale = angle_scale
        self.model = None


    def direct_beam_path(self) -> str:
        """Returns the filepath of the correct direct beam file for the
        instrument being used

        Returns:
            A string of the hogben internal path of the correct
            direct beam file or the local file path
        """

        # Check if the key isn't in the dictionary and check if it is a
        # a local filepath instead
        if self.inst_or_path not in (
                self.non_pol_instr_dict or self.pol_instr_dict):
            if os.path.isfile(self.inst_or_path):
                return self.inst_or_path
            else:
                msg = "Please provide an instrument name or a local filepath"
                raise FileNotFoundError(str(msg))

        if self.spin_states is True:
            path = importlib_resources.files(
                   'hogben.data.directbeams').joinpath(
                   self.pol_instr_dict[self.inst_or_path])

            return path

        path = importlib_resources.files('hogben.data.directbeams').joinpath(
               self.non_pol_instr_dict[self.inst_or_path])

        return path

    def simulate(self, spin_state: Optional[int] = None) -> tuple:
        """Simulates a measurement of self.sample taken at the angles and
        for the durations specified in self.angle_times on the instrument
        specified in self.direct_beam_path

        Args:
            spin_state: optional, integer spin state to simulate if the
            sample is magnetic, refers to [mm, mp, pm, pp]. None if non-
            magnetic

        Returns:
            tuple: model and simulated data for the given `sample`
        """

        # Iterate over each angle to simulate.
        q, r, dr, counts, model = [], [], [], [], None
        total_points = 0
        for angle, points, time in self.angle_times:
            # Simulate the experiment.
            total_points += points
            simulated = self._run_experiment(angle, points, time, spin_state)

            # Combine the data for the angle with the data from previous angles
            q.append(simulated[0])
            r.append(simulated[1])
            dr.append(simulated[2])
            counts.append(simulated[3])
            model = simulated[4]

        # Create a matrix with all the simulated data
        data = np.zeros((total_points, 4))
        for i, element in enumerate([q, r, dr, counts]):
            data[:, i] = np.concatenate(element)

        data = data[data[:, 0].argsort()]  # Sort by Q.

        # If there is no data, return None for the model.
        if len(data) == 0:
            return None, np.zeros((0, 4))

        # If a refnx sample was given, create a refnx ReflectModel.
        if isinstance(self.sample, refnx.reflect.Structure):
            return model, data

        # If a Refl1D sample was given, create a Refl1D Experiment.
        elif isinstance(self.sample, refl1d.model.Stack):
            # Record the data.
            model.probe.dq = self.dq
            if self.sample.ismagnetic:
                model.probe.xs[spin_state].R = r
                model.probe.xs[spin_state].dR = dr
                model.probe.spin_state = spin_state
            else:
                model.probe.R = r
                model.probe.dR = dr

            return model, data

        else:
            msg = 'Sample given is not a valid refnx or refl1D model'
            raise RuntimeError(msg)


    def simulate_magnetic(self, mm: bool = True, mp: bool = True,
                          pm: bool = True, pp: bool = True) -> tuple:
        """Simulates an experiment of a given magnetic `sample` measured
           over a number of angles.
        Args:
            pp: whether to simulate "plus plus" spin state
            pm: whether to simulate "plus minus" spin state
            mp: whether to simulate "minus plus" spin state
            mm: whether to simulate "minus minus" spin state

        Returns:
            tuple: model and simulated data for the given `sample`
        """
        models, datasets = [], []
        # Simulate the spin states if requested.
        for i, spin_state in enumerate([mm, mp, pm, pp]):
            if spin_state is True:
                model, data = self.simulate(i)
                models.append(model)
                datasets.append(data)

        return models, datasets

    def refl1d_experiment(self, q_array: np.ndarray, spin_state: Optional[
        int] = None) -> refl1d.experiment.Experiment:
        """Creates a Refl1D experiment for a given `sample` and `q_array`.
        Also calculates the resolution for refl1D to be the same as is defined
        for refnx as it is not by default.

        Args:
            q_array: Q points to use in the experiment.
            spin_state: spin state to simulate if given a magnetic sample.

        Returns:
            refl1d.experiment.Experiment: experiment for the given `sample`.

        """
        # Transform the resolution from refnx to Refl1D format.
        refl1d_dq = self.dq / (100 * np.sqrt(8 * np.log(2)))

        # Calculate the dq array and use it to define a QProbe.
        dq_array = q_array * refl1d_dq
        probe = refl1d.probe.QProbe(q_array, dq_array, intensity=self.scale,
                                    background=self.bkg)
        probe.dq = self.dq

        # Adjust probe calculation for constant dQ/Q resolution.
        argmin, argmax = np.argmin(q_array), np.argmax(q_array)
        probe.calc_Qo = np.linspace(q_array[argmin] - 3.5 * dq_array[argmin],
                                    q_array[argmax] + 3.5 * dq_array[argmax],
                                    21 * len(q_array))

        # If the sample is magnetic, create a polarised QProbe.
        if self.sample.ismagnetic:
            probes = [None] * 4
            probes[spin_state] = probe
            probe = refl1d.probe.PolarizedQProbe(xs=probes, name='')
            probe.spin_state = spin_state

        return refl1d.experiment.Experiment(probe=probe, sample=self.sample)

    def reflectivity(self, q: np.ndarray) -> np.ndarray:
        """Calculates the reflectance of a `model` at given `q` points.

        Args:
            q: Q points to calculate reflectance at.

        Returns:
            numpy.ndarray: reflectivity for each Q point.

        """
        # If there are no data points, return an empty array.
        if len(q) == 0:
            return np.array([])

        # Calculate the reflectance in either refnx or Refl1D.
        if isinstance(self.sample, refnx.reflect.Stucture):
            self.model = refnx.reflect.ReflectModel(self.sample,
                                                    scale=self.scale,
                                                    bkg=self.bkg,
                                                    dq=self.dq)
            return self.model(q)

        if isinstance(self.sample, refl1d.model.Stack):
            # If magnetic, use the correct spin state.
            experiment = self.refl1d_experiment(q,
                                                self.sample.probe.spin_state)

            if self.sample.ismagnetic:
                return experiment.reflectivity()[self.sample.probe.spin_state][1]
            # experiment.reflectivity() returns q, r, or an array if magnetic

            else:
                return experiment.reflectivity()[1]

    def _run_experiment(self, angle: float, points: int, time: float,
                        spin_state: int) -> tuple:
        """Simulates a single angle measurement of a given `sample` on the
        instrument set in self.direct_beam_file

        Args:
            angle: angle to simulate.
            points: number of points to use for simulated data.
            time: counting time for simulation.
            spin_state: spin state to simulate if given a magnetic sample.

        Returns:
            tuple: simulated Q, R, dR data and incident neutron counts.

        """
        direct_beam = np.loadtxt(self.direct_beam_file, delimiter=',')

        wavelengths = direct_beam[:, 0]  # direct_beam = [wavelength, flux]

        # Scale flux by relative measurement angle squared (assuming both slits
        # scale linearly with angle, this should be correct)
        direct_flux = direct_beam[:, 1] * pow(angle / self.angle_scale, 2)

        # Calculate Q values from the incident angle and wavelength
        q = 4 * np.pi * np.sin(np.radians(angle)) / wavelengths

        # Bin Q's' in equally geometrically-spaced bins using flux as weighting
        q_bin_edges = np.geomspace(min(q), max(q), points + 1)
        flux_binned, _ = np.histogram(q, q_bin_edges, weights=direct_flux)

        # Get the bin centres.
        q_binned = np.asarray(
            [(q_bin_edges[i] + q_bin_edges[i + 1]) / 2 for i in range(points)])

        # Calculate the model reflectivity at each Q point.
        if isinstance(self.sample, refnx.reflect.Structure):
            # Create a refnx ReflectModel if the sample was defined in refnx.
            model = refnx.reflect.ReflectModel(self.sample, scale=self.scale,
                                               bkg=self.bkg, dq=self.dq)
            r_model = model(q_binned)

        elif isinstance(self.sample, refl1d.model.Stack):
            # Create a Refl1D experiment if the sample was defined in Refl1D.
            experiment = self.refl1d_experiment(self.sample, q_binned,
                                                spin_state)
            r_model = self.reflectivity(q_binned, experiment)

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

        return q_binned, r_noisy, r_error, counts_incident, r_model
