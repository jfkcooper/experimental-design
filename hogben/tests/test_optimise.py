"""Unit tests for the methods in the optimise module"""
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


@patch('hogben.optimise.Optimiser._Optimiser__optimise')
def test_optimise_angle_times_length(mock_optimise, refnx_sample):
    """
    Tests that the optimise_angle_times method outputs the correct amount of
    angles and counting times.
    """
    num_angles = 2
    optimiser = Optimiser(refnx_sample)

    # Mock values retreived from previous run
    mock_optimise.return_value = np.array([0.8847156, 0.88834418,
                                           0.00139696,
                                           0.99860304]), -0.7573710562837207
    angles, splits, _ = optimiser.optimise_angle_times(num_angles,
                                                       angle_bounds=(0.2, 2.3),
                                                       verbose=False)
    assert len(angles) == num_angles and len(splits) == num_angles


@patch('hogben.optimise.Optimiser._Optimiser__optimise')
def test_optimise_contrasts(mock_optimise):
    """
    Tests that the optimise_contrasts method outputs the correct amount of
    contrasts and counting times.
    """
    optimiser = Optimiser(BilayerDMPC())
    num_contrasts = 3
    angle_times = [(0.7, 100, 10), (2.3, 100, 40)]

    # Mock values retreived from previous run
    mock_optimise.return_value = (
        np.array([-0.56, 2.15, 6.36, 0.17, 0.28, 0.56]), -0.18
    )
    contrasts, splits, _ = optimiser.optimise_contrasts(num_contrasts,
                                                        angle_times,
                                                        workers=-1,
                                                        verbose=False)
    assert len(contrasts) == num_contrasts and len(splits) == num_contrasts


@patch('hogben.optimise.Optimiser._Optimiser__optimise')
def test_optimise_underlayers(mock_optimise):
    """
    Tests that the optimise_contrasts method outputs the correct amount of
    contrasts and counting times.
    """
    optimiser = Optimiser(BilayerDMPC())

    num_underlayers = 3
    angle_times = [(0.7, 100, 10), (2.3, 100, 40)]
    contrasts = [-0.56, 6.36]
    thick_bounds = (0, 500)
    sld_bounds = (1, 9)

    # Mock values retreived from previous run
    mock_optimise.return_value = (
        np.array([-0.56, 2.15, 6.36, 0.17, 0.28, 0.56]), -0.18
    )

    contrasts, splits, _ = optimiser.optimise_underlayers(num_underlayers,
                                                          angle_times,
                                                          contrasts,
                                                          thick_bounds,
                                                          sld_bounds,
                                                          verbose=False)
    assert len(contrasts) == num_underlayers and len(splits) == num_underlayers


def test_angle_times_func_result(refnx_sample):
    """Checks that the angle_times_func method gives the correct result"""
    angle_time_split = [0.3, 1.3, 0.8, 0.2]  # [angle, angle, time, time]
    num_angles = 2
    contrasts = [3, 14, -2]
    points = 100
    total_time = 10000

    optimiser = Optimiser(refnx_sample)
    result = optimiser._angle_times_func(angle_time_split, num_angles,
                                         contrasts, points, total_time)

    expected_result = -1.722251
    np.testing.assert_allclose(result, expected_result, rtol=1e-06)


def test_contrasts_func_result():
    """Checks that the _contrasts_func method gives the correct result"""
    contrasts_time = [0.3, 9.3, 0.8, 8.2]  # [SLD, SLD, time, time]
    num_contrasts = 2
    angle_splits = [(0.7, 100, 0.6), (2.3, 100, 0.4)]
    total_time = 100000

    optimiser = Optimiser(BilayerDMPC())
    result = optimiser._contrasts_func(contrasts_time, num_contrasts,
                                       angle_splits, total_time)

    expected_result = -2.55788110
    np.testing.assert_allclose(result, expected_result, rtol=1e-06)


def test_underlayers_func():
    """Checks that the _underlayers_func method gives the correct result"""
    thickness_SLD = [50, 20, -5, 10]  # [thickness, thickness, SLD, SLD]

    bilayer = BilayerDMPC()
    optimiser = Optimiser(bilayer)
    num_underlayers = 2
    contrasts = [-0.56, 6.36]

    angle_times = [(0.7, 100, 10000),
                   (2.3, 100, 10000)]
    result = optimiser._underlayers_func(thickness_SLD, num_underlayers,
                                         angle_times, contrasts)

    expected_result = -1.500100628
    np.testing.assert_allclose(result, expected_result, rtol=1e-06)
