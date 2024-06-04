"""Unit tests for the methods in the visualise module"""

import pytest
from hogben.models.samples import Sample
from hogben.visualise import scan_parameters
from refnx.analysis import Parameter
from refnx.reflect import SLD
from unittest.mock import MagicMock, patch


@pytest.fixture
def refnx_sample():
    """Defines a structure describing a simple sample."""
    air = SLD(0, name='Air')
    layer1_thick = Parameter(100, 'Layer 1 Thickness', (50, 120))
    layer1_sld = Parameter(4, 'Layer 1 SLD', (50, 120))
    layer1_thick.optimize = True
    layer1_sld.optimize = True
    layer1 = SLD(layer1_sld, name='Layer 1')(thick=layer1_thick, rough=2)
    layer2 = SLD(8, name='Layer 2')(thick=150, rough=2)
    substrate = SLD(2.047, name='Substrate')(thick=0, rough=2)
    structure = air | layer1 | layer2 | substrate
    return Sample(structure)


class MockFisher:
    """Mocks the Fisher class, to return a simple minimum eigenvalue of 1.0 """
    def __init__(self, _sample, _angle_times):
        """Initialize the mocked Fisher class and set eigenval to 1"""
        self.min_eigenval = 1.0

    @staticmethod
    def from_sample(sample, angle_times):
        """Create a Mocked Fisher object from sample"""
        return MockFisher(sample, angle_times)


@patch('hogben.visualise.plt.subplots')
@patch('hogben.optimise.Fisher.from_sample', new=MockFisher)
def test_scan_parameters(mock_subplots, refnx_sample):
    """Tests whether one graph is generated for each optimization parameter"""
    refnx_sample._vary_structure()

    # Mock the return value of plt.subplots to get the ax object
    mock_subplots.return_value = (MagicMock(), MagicMock())

    params = refnx_sample.get_param_by_attribute('optimize')
    angle_times = [(0.7, 100, 100000), (2.0, 100, 100000)]

    # Call the method with mocked objects
    scan_parameters(refnx_sample, params, angle_times)
    # Assert the call count of plot method on the ax object
    assert mock_subplots.return_value[1].plot.call_count == len(params)
