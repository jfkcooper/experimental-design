import pytest
import numpy as np

from hogben.optimise import Optimiser
from refnx.reflect import SLD as SLD_refnx
from hogben.models.samples import Sample
from hogben.models.bilayers import BilayerDMPC
from unittest.mock import patch

@pytest.fixture
def refnx_sample():
    """Defines a structure describing a simple sample."""
    air = SLD_refnx(0, name='Air')
    layer1 = SLD_refnx(4, name='Layer 1')(thick=100, rough=2)
    layer2 = SLD_refnx(8, name='Layer 2')(thick=150, rough=2)
    substrate = SLD_refnx(2.047, name='Substrate')(thick=0, rough=2)
    structure = air | layer1 | layer2 | substrate
    return Sample(structure)


@patch("hogben.optimise.Optimiser._Optimiser__optimise")
def test_optimise_angle_times_length(mock_optimise, refnx_sample):
    num_angles = 2
    optimiser = Optimiser(refnx_sample)
    mock_optimise.return_value = np.array([0.8847156, 0.88834418,
                                           0.00139696,
                                           0.99860304]), -0.7573710562837207
    angles, splits, _ = optimiser.optimise_angle_times(num_angles,
                                                       angle_bounds=(0.2, 2.3),
                                                       verbose=False)
    assert len(angles) == num_angles and len(splits) == num_angles


@patch("hogben.optimise.Optimiser._Optimiser__optimise")
def test_optimise_optimise_contrasts(mock_optimise):
    optimiser = Optimiser(BilayerDMPC())

    # Get mock values from older run
    mock_optimise.return_value = (
        np.array([-0.56, 2.15, 6.36, 0.17, 0.28, 0.56]), -0.18
    )
    num_contrasts = 3
    angle_times = [(0.7, 100, 10), (2.3, 100, 40)]

    contrasts, splits, _ = optimiser.optimise_contrasts(num_contrasts,
                                                        angle_times,
                                                        workers=-1,
                                                        verbose=False)
    assert len(contrasts) == num_contrasts and len(splits) == num_contrasts

def test_angle_times_func_result(refnx_sample):
    x = [0.3, 1.3, 0.8, 0.2]  # [angle, angle, time, time]
    num_angles = 2
    contrasts = [3, 14, -2]
    points = 100
    total_time = 10000

    optimiser = Optimiser(refnx_sample)
    result = optimiser._angle_times_func(x, num_angles, contrasts, points, total_time)
    expected_result = -1.7721879778537162

    np.testing.assert_allclose(result, expected_result, rtol=1e-08)

