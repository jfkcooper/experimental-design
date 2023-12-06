"""Tests for the simulation module"""

import pytest

import numpy as np
from refnx.reflect import SLD, ReflectModel

from refl1d.material import SLD as refl1dSLD

from unittest import mock
from hogben.simulate import SimulateReflectivity


@pytest.fixture(scope="module")
def refnx_structure():
    """Defines a structure describing a simple sample."""
    air = SLD(0, name='Air')
    layer1 = SLD(4, name='Layer 1')(thick=100, rough=2)
    layer2 = SLD(8, name='Layer 2')(thick=150, rough=2)
    substrate = SLD(2.047, name='Substrate')(thick=0, rough=2)

    sample_1 = air | layer1 | layer2 | substrate
    return sample_1


@pytest.fixture(scope="module")
def refnx_model(refnx_structure):
    model = ReflectModel(refnx_structure)
    model.bkg = 1e-6
    model.dq = 2
    model.scale = 1.0
    return model


class TestSimulate:
    angle_times = [(0.3, 100, 1000)]  # (Angle, Points, Time)
    instrument = 'OFFSPEC'
    def test_data_streaming(self, refnx_model):
        """Tests that without an input for the datafile, the correct one is picked up"""
        sim = SimulateReflectivity(refnx_model, self.angle_times, self.instrument)
        simulated_datapoints = sim.simulate()
        np.testing.assert_array_less(np.zeros_like(simulated_datapoints), simulated_datapoints)  # counts

        # Check that the default instrument also works
        sim = SimulateReflectivity(refnx_model, self.angle_times)
        simulated_datapoints = sim.simulate()
        np.testing.assert_array_less(np.zeros_like(simulated_datapoints), simulated_datapoints)  # counts

    def test_incident_flux_data(self):
        """
        Tests that the `_incident_flux_data` function correctly returns count data
        when instruments are loaded, and raises errors when given an incorrect path
        """
        # Test that non-polarised instruments work and have counts in
        for instrument in ['OFFSPEC', 'SURF', 'INTER', 'POLREF']:
            sim_no_pol = SimulateReflectivity(None, angle_times=self.angle_times,
                                              inst_or_path=instrument)
            assert len(sim_no_pol._incident_flux_data(polarised=False)) > 1
            assert np.sum(sim_no_pol._incident_flux_data(polarised=False)) > 10

        # Test that polarised instruments work and have counts in
        for instrument in ['OFFSPEC', 'POLREF']:
            sim_pol = SimulateReflectivity(None, angle_times=self.angle_times,
                                              inst_or_path=instrument)
            assert len(sim_pol._incident_flux_data(polarised=True)) > 1
            assert np.sum(sim_pol._incident_flux_data(polarised=True)) > 10

        # Test that a non-existing path raises an error
        with pytest.raises(FileNotFoundError):
            sim_wrong_path = SimulateReflectivity(None, angle_times=self.angle_times,
                                     inst_or_path='no_instrument_exists')
            sim_wrong_path._incident_flux_data()

        # Test that a blank instrument raises an error
        with pytest.raises(FileNotFoundError):
            sim_no_path = SimulateReflectivity(None, angle_times=self.angle_times,
                                                  inst_or_path='')
            sim_no_path._incident_flux_data()

        # Test that a non-polarised instrument can't be used for a polarised simulation
        with pytest.raises(FileNotFoundError):
            sim_not_in_pol = SimulateReflectivity(None, angle_times=self.angle_times,
                                                  inst_or_path='SURF')
            sim_not_in_pol._incident_flux_data(polarised=True)

    def test_refnx_reflectivity_model(self, refnx_model):
        """
        Checks that a refnx model reflectivity generated through
        `hogben.reflectivity` is always greater than zero.
        """
        sim = SimulateReflectivity(refnx_model, self.angle_times, self.instrument)
        ideal_reflectivity = sim.reflectivity(np.linspace(0.001, 0.3, 200))

        np.testing.assert_array_less(np.zeros(len(ideal_reflectivity)),
                                     ideal_reflectivity)

    def test_run_experiment_unpolarised(self, refnx_model):
        """Checks the output of _run_experiment gives the right outputs"""
        sim = SimulateReflectivity(refnx_model, self.angle_times, self.instrument)
        for angle, points, time in self.angle_times:
            q_binned, r_noisy, r_error, counts_incident = sim._run_experiment(angle, points, time)
        assert len(q_binned) == self.angle_times[0][1]

    def test_run_experiment_polarised(self, refnx_model):
        """Checks the output of _run_experiment gives the right outputs"""
        sim = SimulateReflectivity(refnx_model, self.angle_times, self.instrument)
        for angle, points, time in self.angle_times:
            q_binned, r_noisy, r_error, counts_incident = sim._run_experiment(angle, points, time)
        assert len(q_binned) == self.angle_times[0][1]
"""
def test_refnx_simulate_data(self):
    """
    #Checks that simulated reflectivity data points and simulated neutron
    #counts generated through `hogben.simulate` are always greater than
    #zero (given a long count time).
    """
    angle_times = [(0.3, 100, 1000)]
    _, simulated_datapoints = simulate(self.sample_1, angle_times,
                                       self.scale, self.bkg, self.dq,
                                       self.ref)

    np.testing.assert_array_less(np.zeros(len(simulated_datapoints)),
                                 simulated_datapoints[:,1])  # reflectivity
    np.testing.assert_array_less(np.zeros(len(simulated_datapoints)),
                                 simulated_datapoints[:, 3])  # counts

@pytest.mark.parametrize('instrument',
                         ('OFFSPEC',
                          'POLREF',
                          'SURF',
                          'INTER'))
def test_simulation_instruments(self, instrument):
    """
    #Tests that all of the instruments are able to simulate a model and
    #counts data.
    """
    angle_times = [(0.3, 100, 1000)]
    _, simulated_datapoints = simulate(self.sample_1, angle_times,
                                       self.scale, self.bkg, self.dq,
                                       inst_or_path=instrument)
    # reflectivity
    np.testing.assert_array_less(np.zeros(angle_times[0][1]),
                                 simulated_datapoints[:, 1])
    np.all(np.less_equal(np.zeros(angle_times[0][1]),
                                 simulated_datapoints[:, 3]))  # counts

@pytest.mark.parametrize('instrument',
                         ('OFFSPEC',
                         'POLREF'))
def test_simulation_magnetic_instruments(self, instrument):
    """
    #Tests that all of the instruments are able to simulate a model and
    #counts data.
    """
    angle_times = [(0.3, 100, 1000)]
    _, simulated_datapoints = simulate_magnetic(self.sample_1, angle_times,
                                       self.scale, self.bkg, self.dq,
                                       inst_or_path=instrument)

    for i in range(4):
        # reflectivity
        np.testing.assert_array_less(np.zeros(angle_times[0][1]),
                                     simulated_datapoints[i][:, 1])
        # counts
        np.testing.assert_array_less(np.zeros(angle_times[0][1]),
                                     simulated_datapoints[i][:, 3])

"""