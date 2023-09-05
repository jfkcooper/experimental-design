import pytest

import numpy as np
from refnx.reflect import SLD

from hogben.simulate import simulate, simulate_magnetic, reflectivity


def sample_structure():
    """Defines a structure describing a simple sample."""
    air = SLD(0, name='Air')
    layer1 = SLD(4, name='Layer 1')(thick=100, rough=2)
    layer2 = SLD(8, name='Layer 2')(thick=150, rough=2)
    substrate = SLD(2.047, name='Substrate')(thick=0, rough=2)

    sample_1 = air | layer1 | layer2 | substrate
    return sample_1


class Test_Simulate():
    ref = 'OFFSPEC'
    angle_times = [(0.7, 100, 5), (2.0, 100, 20)]  # (Angle, Points, Time)
    scale = 1
    bkg = 1e-6
    dq = 2
    instrument = 'OFFSPEC'
    sample_1 = sample_structure()
    model_1, data_1 = simulate(sample_1, angle_times, scale, bkg, dq, ref)

    def test_data_streaming(self):
        """Tests that without an input for the datafile, the correct one is
         picked up"""
        angle_times = [(0.3, 100, 1000)]
        _, simulated_datapoints = simulate(
            self.sample_1, angle_times, self.scale, self.bkg, self.dq,
            self.ref)
        np.testing.assert_array_less(np.zeros(len(simulated_datapoints)),
                                     simulated_datapoints[:, 3])  # counts

        _, simulated_datapoints = simulate(self.sample_1, angle_times,
                                           self.scale, self.bkg, self.dq)
        np.testing.assert_array_less(np.zeros(len(simulated_datapoints)),
                                     simulated_datapoints[:, 3])  # counts

    def test_refnx_simulate_model(self):
        """
        Checks that a model reflectivity from refnx generated through
        hogben.simulate is always greater than zero.
        """
        q = self.data_1[:, 0]
        r_model = reflectivity(q, self.model_1)

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
        # reflectivity
        np.testing.assert_array_less(np.zeros(len(simulated_datapoints)),
                                     simulated_datapoints[:, 1])
        # counts
        np.testing.assert_array_less(np.zeros(len(simulated_datapoints)),
                                     simulated_datapoints[:, 3])

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

    @pytest.mark.parametrize('instrument', ('OFFSPEC', 'POLREF'))
    def test_simulation_magnetic_instruments(self, instrument):
        """
        Tests that all of the instruments are able to simulate a model and
        counts data.
        """
        angle_times = [(0.3, 100, 1000)]
        _, simulated_datapoints = simulate_magnetic(
            self.sample_1, angle_times,
            self.scale, self.bkg, self.dq, inst_or_path=instrument)

        for i in range(4):
            # reflectivity
            np.testing.assert_array_less(np.zeros(angle_times[0][1]),
                                         simulated_datapoints[i][:, 1])
            # counts
            np.testing.assert_array_less(np.zeros(angle_times[0][1]),
                                         simulated_datapoints[i][:, 3])
