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
from matplotlib.testing.compare import compare_images
from refnx.reflect import SLD as SLD_refnx
from refl1d.material import SLD as SLD_refl1d
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
    fig.savefig(file_path, dpi=60)

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

@patch('hogben.models.samples.save_plot', side_effect=mock_save_plot)
@pytest.mark.parametrize('sample_class', ("refnx_sample",
                                         "refl1d_sample"))
def test_sld_profile(_mock_save_plot, sample_class, request):
    """
    Tests whether the sld_profile function still correctly outputs a figure of
    the sld_profile as compared to a reference figure
    """
    sample = request.getfixturevalue(sample_class)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Use temporary directory, so it doesn't leave any files after testing
    with tempfile.TemporaryDirectory() as temp_dir:
        sample.sld_profile(temp_dir)
        img_ref = os.path.join(script_dir, 'reference_figures', 'sld_profile.png')
        img_test = os.path.join(temp_dir, sample.name, 'sld_profile.png')
        compare_images(img_ref, img_test, 0.001)
@patch('hogben.models.samples.save_plot', side_effect=mock_save_plot)
@pytest.mark.parametrize('sample_class', ("refnx_sample",
                                         "refl1d_sample"))
def test_reflectivity_profile(_mock_save_plot, sample_class, request):
    """
    Tests whether the reflectivity_profile function still correctly outputs a
    figure of the reflectivity_profile as compared to a reference figure
    """
    sample = request.getfixturevalue(sample_class)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Use temporary directory, so it doesn't leave any files after testing
    with tempfile.TemporaryDirectory() as temp_dir:
        sample.reflectivity_profile(temp_dir)
        img_ref = os.path.join(script_dir, 'reference_figures',
                               'reflectivity_profile.png')
        img_test = os.path.join(temp_dir, sample.name,
                                'reflectivity_profile.png')
        compare_images(img_ref, img_test, 0.001)

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