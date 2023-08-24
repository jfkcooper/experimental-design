import pytest
import os
import copy
import numpy as np
import refnx.reflect
import refl1d.experiment
import tempfile
import hogben.models.samples as samples

from hogben.utils import fisher
from refnx.reflect import SLD as SLD_refnx
from refl1d.material import SLD as SLD_refl1d
from hogben.models.samples import Sample
from hogben.simulate import simulate
from matplotlib.testing.compare import compare_images


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
    # Define sample
    air = SLD_refl1d(rho=0, name='Air')
    layer1 = SLD_refl1d(rho=4, name='Layer 1')(thickness=60, interface=8)
    layer2 = SLD_refl1d(rho=8, name='Layer 2')(thickness=150, interface=2)
    substrate = SLD_refl1d(rho=2.047, name='Substrate')(thickness=0,
                                                        interface=2)
    layer1.thickness.pm(10)
    layer2.thickness.pm(10)
    layer1.interface.pm(1)
    layer2.interface.pm(1)
    structure = substrate | layer2 | layer1 | air
    return Sample(structure)

def compare_sample_structure(refl1d, refnx):
    refl1d_params = {}
    refnx_params = {}
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
def test_angle_info(refnx_sample):
    angle_times = [(0.7, 100, 10000), (2.0, 100, 10000)]
    angle_info = refnx_sample.angle_info(angle_times)

    model, data = simulate(refnx_sample.structure, angle_times)
    qs, counts, models = [data[:, 0]], [data[:, 3]], [model]
    g = fisher(qs, refnx_sample.params, counts, models)
    np.testing.assert_allclose(g, angle_info, rtol=1e-08)

@pytest.mark.parametrize('sample_class', ("refnx_sample",
                                         "refl1d_sample"))
def test_sld_profile(sample_class, request):
    sample = request.getfixturevalue(sample_class)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with tempfile.TemporaryDirectory() as temp_dir:
        sample.sld_profile(temp_dir)
        img_ref = os.path.join(script_dir, 'reference_figures',
                               'sld_profile.png')
        img_test = os.path.join(temp_dir, sample.name, 'sld_profile.png')
        compare_images(img_ref, img_test, 0.001)

@pytest.mark.parametrize('sample_class', ("refnx_sample",
                                         "refl1d_sample"))
def test_reflectivity_profile(sample_class, request):
    sample = request.getfixturevalue(sample_class)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with tempfile.TemporaryDirectory() as temp_dir:
        sample.reflectivity_profile(temp_dir)
        img_ref = os.path.join(script_dir, 'reference_figures',
                               'reflectivity_profile.png')
        img_test = os.path.join(temp_dir, sample.name, 'reflectivity_profile.png')
        compare_images(img_ref, img_test, 0.001)

def test_to_refl1d_instance(refnx_sample):
    refnx_sample.to_refl1d()
    assert isinstance(refnx_sample.structure, refl1d.model.Stack)

def test_to_refnx_instance(refl1d_sample):
    refl1d_sample.to_refnx()
    assert isinstance(refl1d_sample.structure, refnx.reflect.Structure)

def test_to_refl1d_values(refnx_sample):
    refl1d_sample = copy.deepcopy(refnx_sample)
    refl1d_sample.to_refl1d()
    assert compare_sample_structure(refl1d_sample, refnx_sample)

def test_to_refnx_values(refl1d_sample):
    refnx_sample = copy.deepcopy(refl1d_sample)
    refnx_sample.to_refnx()
    assert compare_sample_structure(refl1d_sample, refnx_sample)

def test_simple_sample():
    simple_sample = samples.simple_sample()
    simple_sample.to_refl1d()

def test_many_param_sample():
    many_param_sample = samples.many_param_sample()
    many_param_sample.to_refl1d()

def test_thin_layer_sample_1():
    thin_layer_sample_1 = samples.thin_layer_sample_1()
    thin_layer_sample_1.to_refl1d()

def test_thin_layer_sample_2():
    thin_layer_sample_2 = samples.thin_layer_sample_2()
    thin_layer_sample_2.to_refl1d()

def test_similar_sld_sample_1():
    similar_sld_sample_1 = samples.similar_sld_sample_1()
    similar_sld_sample_1.to_refl1d()

def test_similar_sld_sample_2():
    similar_sld_sample_2 = samples.similar_sld_sample_2()
    similar_sld_sample_2.to_refl1d()