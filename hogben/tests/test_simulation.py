import pytest

import numpy as np
from refnx.reflect import SLD

from unittest import mock
from hogben.simulate import Simulation



#@pytest.fixture(scope="module")
def sample_structure():
    """Defines a structure describing a simple sample."""
    air = SLD(0, name='Air')
    layer1 = SLD(4, name='Layer 1')(thick=100, rough=2)
    layer2 = SLD(8, name='Layer 2')(thick=150, rough=2)
    substrate = SLD(2.047, name='Substrate')(thick=0, rough=2)

    sample_1 = air | layer1 | layer2 | substrate
    return sample_1


class TestSimulate:
    angle_times = [(0.3, 100, 1000)]  # (Angle, Points, Time)
    scale = 1
    bkg = 1e-6
    dq = 2
    instrument = 'OFFSPEC'


    def test_data_streaming(self):
        """Tests that without an input for the datafile, the correct one is picked up"""
        sim = Simulation(sample_structure(), self.angle_times, self.scale, self.bkg, self.dq,
                         self.instrument)
        _, simulated_datapoints = sim.simulate()
        np.testing.assert_array_less(np.zeros(len(simulated_datapoints)), simulated_datapoints[:, 3])  # counts

        # Check that the default instrument also works
        sim = Simulation(sample_structure(), self.angle_times, self.scale, self.bkg, self.dq)
        _, simulated_datapoints = sim.simulate()
        np.testing.assert_array_less(np.zeros(len(simulated_datapoints)), simulated_datapoints[:, 3])  # counts


    def test_refnx_simulate_model(self):
        """
        Checks that a model reflectivity from refnx generated through
        hogben.simulate is always greater than zero.
        """
        sim = Simulation(sample_structure(), self.angle_times, self.scale,
                         self.bkg, self.dq, self.instrument)
        model_1, data_1 = sim.simulate()
        q = data_1[:, 0]
        r_model = sim.reflectivity(q)

        np.testing.assert_array_less(np.zeros(len(r_model)), r_model)

def test_refnx_simulate_data(self):
    """
    Checks that simulated reflectivity data points and simulated neutron
    counts generated through hogben.simulate are always greater than
    zero (given a long count time).
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
    Tests that all of the instruments are able to simulate a model and
    counts data.
    """
    angle_times = [(0.3, 100, 1000)]
    _, simulated_datapoints = simulate(self.sample_1, angle_times,
                                       self.scale, self.bkg, self.dq,
                                       inst_or_path=instrument)
    # reflectivity
    np.testing.assert_array_less(np.zeros(angle_times[0][1]),
                                 simulated_datapoints[:, 1])
    np.testing.assert_array_less(np.zeros(angle_times[0][1]),
                                 simulated_datapoints[:, 3])  # counts

@pytest.mark.parametrize('instrument',
                         ('OFFSPEC',
                         'POLREF'))
def test_simulation_magnetic_instruments(self, instrument):
    """
    Tests that all of the instruments are able to simulate a model and
    counts data.
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
