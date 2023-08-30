import copy
import os
import tempfile

import numpy as np
import pytest
import matplotlib
import refnx.reflect
import refl1d.experiment
import hogben.models.samples as samples

from hogben.models.samples import Sample
from hogben.simulate import simulate
from hogben.utils import fisher
from refnx.reflect import SLD as SLD_refnx
from refl1d.material import SLD as SLD_refl1d
from unittest.mock import Mock, patch


@pytest.fixture
def refnx_sample():
    """Defines a structure describing a simple sample."""
    air = SLD_refnx(0, name='Air')
    layer1 = SLD_refnx(4, name='Layer 1')(thick=60, rough=8)
    layer2 = SLD_refnx(8, name='Layer 2')(thick=150, rough=2)
    substrate = SLD_refnx(2.047, name='Substrate')(thick=0, rough=2)
    structure = air | layer1 | layer2 | substrate
    return Sample(structure)

@pytest.fixture
def refl1d_sample():
    """Define a bilayer sample, and return the associated refl1d model"""
    air = SLD_refl1d(rho=0, name='Air')
    layer1 = SLD_refl1d(rho=4, name='Layer 1')(thickness=60, interface=8)
    layer2 = SLD_refl1d(rho=8, name='Layer 2')(thickness=150, interface=2)
    substrate = SLD_refl1d(rho=2.047, name='Substrate')(thickness=0,
                                                        interface=2)
    structure = substrate | layer2 | layer1 | air
    return Sample(structure)

def mock_save_plot(fig: matplotlib.figure.Figure,
                   save_path: str,
                   filename: str):
    """
    A mocked version of the hogben.utils.save_plot method, where a lower
    dpi is used when saving a figure

    Args:
        fig: The matplotlib figure to be plotted
        save_path: The path where the figure will be saved
        filename: The file name of the figure without the png extension
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_path = os.path.join(save_path, filename + '.png')
    fig.savefig(file_path, dpi=40)

def compare_sample_structure(refl1d: refl1d.model.Stack,
                             refnx: refnx.reflect.Structure):
    """
    Compare the structural parameters of a refnx and refl1d model, and check
    if the values are equal. This function is as a helper function to
    validate that a converted refl1d/refnx sample still has the same
    structural parameters and values as it had before conversion.
    Args:
        refl1d: The refl1d structure to be compared
        refnx: The refnx structure to be compared

    Returns:
        bool: Boolean describing whether the two models have the same values
    """
    refl1d_params = {}
    refnx_params = {}

    # Check structure in reversed order, so it matches with refl1d
    for component in list(reversed(refnx.structure))[1:]:
        refnx_params[component.name] = {
            "sld": component.sld.real.value,
            "thick": component.thick.value,
            "rough": component.rough.value
        }
    for component in refl1d.structure[1:]:
        refl1d_params[component.name] = {
            "sld": component.material.rho.value,
            "thick": component.thickness.value,
            "rough": component.interface.value
        }
    return refnx_params == refl1d_params

@pytest.mark.parametrize('sample_class', ("refnx_sample",
                                         "refl1d_sample"))
def test_angle_info(sample_class, request):
    """
    Tests whether the angle_info function correctly calculates the Fisher
    information, and outputs the same values as if the functions were called
    manually.
    """

    # Get Fisher information from tested unit
    sample = request.getfixturevalue(sample_class)
    angle_times = [(0.7, 100, 10000), (2.0, 100, 10000)]
    angle_info = sample.angle_info(angle_times)

    # Get Fisher information directly
    model, data = simulate(sample.structure, angle_times)
    qs, counts, models = [data[:, 0]], [data[:, 3]], [model]
    g = fisher(qs, sample.params, counts, models)

    np.testing.assert_allclose(g, angle_info, rtol=1e-08)

@patch('hogben.models.samples.Sample._get_sld_profile')
@patch('hogben.models.samples.save_plot', side_effect=mock_save_plot)
def test_sld_profile_valid_figure(_mock_save_plot,
                                  mock_sld_profile, refnx_sample):
    """
    Tests whether the sld_profile function succesfully outputs a figure
    """
    mock_sld_profile.return_value = ([0, 10, 60, 110, 160, 210],
                                     [4, 9, -2, 9, -2, 9])


    # Use temporary directory, so it doesn't leave any files after testing
    with tempfile.TemporaryDirectory() as temp_dir:
        refnx_sample.sld_profile(temp_dir)
        img_test = os.path.join(temp_dir, refnx_sample.name, 'sld_profile.png')
        assert os.path.isfile(img_test)

@patch('hogben.models.samples.Sample._get_reflectivity_profile')
@patch('hogben.models.samples.save_plot', side_effect=mock_save_plot)
def test_reflectivity_profile_valid_figure(_mock_save_plot,
                                           _mock_reflectivity_profile,
                                           refnx_sample):
    """
    Tests whether the reflectivity_profile function succesfully outputs a
    figure
    """
    _mock_reflectivity_profile.return_value = ([0, 0.05, 0.1, 0.15, 0.2],
                                               [1, 0.9, 0.8, 0.75, 0.8])
    # Use temporary directory, so it doesn't leave any files after testing
    with tempfile.TemporaryDirectory() as temp_dir:
        refnx_sample.reflectivity_profile(temp_dir)
        img_test = os.path.join(temp_dir, refnx_sample.name,
                                'reflectivity_profile.png')
        assert os.path.isfile(img_test)

@patch('hogben.models.samples.save_plot', side_effect=mock_save_plot)
@pytest.mark.parametrize('sample_class', ("refnx_sample",
                                         "refl1d_sample"))
def test_sld_profile_length(_mock_save_plot, sample_class,
                                        request):
    """
    Tests whether _get_sld_profile() succesfully retrieves two arrays with
    equal lengths, representing an SLD profile that can be plotted in a figure
    """
    sample = request.getfixturevalue(sample_class)
    z, slds = sample._get_sld_profile()
    assert len(z) == len(slds)
    assert len(z) > 0 # Make sure arrays are not empty

@pytest.mark.parametrize('sample_class', ("refnx_sample",
                                         "refl1d_sample"))
def test_reflectivity_profile_positive(sample_class,
                                        request):
    """
    Tests whether _get_reflectivity_profile() succesfully obtains reflectivity
    values that are all positively valued
    """
    sample = request.getfixturevalue(sample_class)
    q, r = sample._get_reflectivity_profile(0.005, 0.4, 500, 1, 1e-7, 2)
    assert min(r) > 0

def test_reflectivity_invalid_structure():
    """
    Test whether a RunTimeError is correctly given when an invalid sample
    structure is used in get_reflectivity_profile
    """
    sample = Mock(spec=None)
    with pytest.raises(RuntimeError):
        Sample._get_reflectivity_profile(sample, 0.005, 0.4, 500, 1,
                                            1e-7, 2)

def test_sld_invalid_structure():
    """
    Test whether a RunTimeError is correctly given when an invalid sample
    structure is used in get_sld_profile
    """
    sample = Mock(spec=None)
    with pytest.raises(RuntimeError):
        Sample._get_sld_profile(sample)

def test_vary_structure_invalid_structure():
    """
    Test whether a RunTimeError is correctly given when an invalid sample
    structure is used in _vary_structure
    """
    structure = Mock(spec=None)
    with pytest.raises(RuntimeError):
        Sample._Sample__vary_structure(structure)

@pytest.mark.parametrize('sample_class', ("refnx_sample",
                                         "refl1d_sample"))
def test_reflectivity_profile_length(sample_class,
                                        request):
    """
    Tests whether _get_reflectivity_profile() succesfully retrieves two arrays
    with equal lengths, representing a reflectivity profile that can be
    plotted in a figure.
    """
    sample = request.getfixturevalue(sample_class)
    q, r = sample._get_reflectivity_profile(0.005, 0.4, 500, 1, 1e-7, 2)
    assert len(q) == len(r)
    assert len(q) > 0 # Make sure array is not empty

def test_to_refl1d_instance(refnx_sample):
    """
    Tests whether a conversion from refnx to a refl1d structure correctly
    results in a structure of the refl1d instance.
    """
    refnx_sample.to_refl1d()
    assert isinstance(refnx_sample.structure, refl1d.model.Stack)

def test_to_refnx_instance(refl1d_sample):
    """
    Tests whether a conversion from refl1d to a refnx structure correctly
    results in a structure of the refnx instance.
    """
    refl1d_sample.to_refnx()
    assert isinstance(refl1d_sample.structure, refnx.reflect.Structure)

def test_to_refl1d_values(refnx_sample):
    """
    Tests whether the strucutral parameters are correctly carried over when
    converting from a refnx sample to a refl1d sample.
    """
    refl1d_sample = copy.deepcopy(refnx_sample)
    refl1d_sample.to_refl1d()
    assert compare_sample_structure(refl1d_sample, refnx_sample)

def test_to_refnx_values(refl1d_sample):
    """
    Tests whether the structural parameters are correctly carried over when
    converting from a refl1d sample to a refnx sample.
    """
    refnx_sample = copy.deepcopy(refl1d_sample)
    refnx_sample.to_refnx()
    assert compare_sample_structure(refl1d_sample, refnx_sample)

def test_simple_sample():
    """
    Tests whether simple_sample leads to a valid refnx structure that can be
    converted to refl1d
    """
    simple_sample = samples.simple_sample()
    simple_sample.to_refl1d()

def test_many_param_sample():
    """
    Tests whether many_param_sample leads to a valid refnx structure that
    can be converted to refl1d
    """
    many_param_sample = samples.many_param_sample()
    many_param_sample.to_refl1d()

def test_thin_layer_sample_1():
    """
    Tests whether thin_layer_sample_1 leads to a valid refnx structure that
    can be converted to refl1d
    """
    thin_layer_sample_1 = samples.thin_layer_sample_1()
    thin_layer_sample_1.to_refl1d()

def test_thin_layer_sample_2():
    """
    Tests whether thin_layer_sample_2 leads to a valid refnx structure that
    can be converted to refl1d
    """
    thin_layer_sample_2 = samples.thin_layer_sample_2()
    thin_layer_sample_2.to_refl1d()

def test_similar_sld_sample_1():
    """
    Tests whether similar_sld_sample_1 leads to a valid refnx structure that
    can be converted to refl1d
    """
    similar_sld_sample_1 = samples.similar_sld_sample_1()
    similar_sld_sample_1.to_refl1d()

def test_similar_sld_sample_2():
    """
    Tests whether similar_sld_sample_2 leads to a valid refnx structure that
    can be converted to refl1d
    """
    similar_sld_sample_2 = samples.similar_sld_sample_2()
    similar_sld_sample_2.to_refl1d()

@patch('hogben.models.samples.save_plot', side_effect=mock_save_plot)
def test_main_function(_mock_save_plot):
    """
    Tests whether the main function runs properly and creates a figure for
    all defined model types.
    """
    work_dir = os.getcwd()
    with tempfile.TemporaryDirectory() as temp_dir:
        # Results are saved in parent folder, so need to create a temporary
        # child, as we don't have access to the parent of the temp folder
        child_dir = os.path.join(temp_dir, 'child')
        os.mkdir(child_dir)
        os.chdir(child_dir) # Run script from inside child folder
        samples.run_main()
        os.chdir(temp_dir) # Check the results from temp folder

        for subfolder in os.listdir('results'):
            reflectivity_profile = os.path.join('results', subfolder,
                                                'reflectivity_profile.png')
            sld_profile = os.path.join('results', subfolder, 'sld_profile.png')
            assert os.path.isfile(reflectivity_profile)
            assert os.path.isfile(sld_profile)
    os.chdir(work_dir)
