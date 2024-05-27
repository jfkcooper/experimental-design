"""
Contains class and methods related to the Sample class.
"""

import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

import refnx.dataset
import refnx.reflect
import refnx.analysis

from hogben.simulate import SimulateReflectivity
from hogben.utils import Fisher, Sampler, save_plot
from hogben.models.base import BaseSample
from refnx.analysis import Objective

plt.rcParams['figure.figsize'] = (9, 7)
plt.rcParams['figure.dpi'] = 600


class Sample(BaseSample):
    """Wrapper class for a standard refnx reflectometry sample.

    Attributes:
        structure (refnx.reflect.Structure: refnx sample.
        name (str): name of the sample.
        params (list): varying parameters of sample.

    """

    def __init__(self, structure, **settings):
        """
        Initializes a sample given a structure, and sets the sample name and
        parameters

        Args:
            structure: Sample structure defined in the refnx or refl1d model
        """
        if isinstance(structure, refnx.reflect.Structure):
            structure = [structure]
        self.structures = structure
        self.name = structure[0].name
        self.scale = settings.get('scale', 1)
        self.bkg = settings.get('bkg', 5e-6)
        self.dq = settings.get('dq', 2)
        self.polarised = settings.get('polarised', True)

    @property
    def params(self):
        return self.get_varying_parameters()

    def _vary_structure(self, bound_size=0.2):
        """Varies the SLD and thickness of each layer in the sample structures.

        Args:
            structure (refnx.reflect.Structure): structure to vary.
            bound_size (float): size of bounds to place on varying parameters.

        Returns:
            list: varying parameters of sample.

        """
        for structure in self._structures:
            params = []
            # The structure was defined in refnx.
            if isinstance(structure, refnx.reflect.Structure):
                # Vary the SLD and thickness of each coprimponent (layer).
                for component in structure[1:-1]:
                    sld = component.sld.real
                    sld_bounds = (
                        sld.value * (1 - bound_size),
                        sld.value * (1 + bound_size),
                    )
                    sld.setp(vary=True, bounds=sld_bounds)
                    params.append(sld)

                    thick = component.thick
                    thick_bounds = (
                        thick.value * (1 - bound_size),
                        thick.value * (1 + bound_size),
                    )
                    thick.setp(vary=True, bounds=thick_bounds)
                    params.append(thick)

            else:
                raise RuntimeError('invalid structure given')

            return params

    @property
    def models(self):
        return self.get_models()

    def angle_info(self, angle_times, contrasts=None):
        """Calculates the Fisher information matrix for a sample measured
           over a number of angles.

        Args:
            angle_times (list): points and times for each angle to simulate.

        Returns:
            Fisher: Fisher information object

        """
        # Return the Fisher information matrix calculated from simulated data.
        model = refnx.reflect.ReflectModel(self.get_structures()[0])
        data = SimulateReflectivity(model, angle_times).simulate()
        qs, counts, models = [data[0]], [data[3]], [model]
        return Fisher(qs, self.params, counts, models)

    def nested_sampling(self,
                        angle_times: list,
                        save_path: str,
                        filename: str,
                        dynamic: bool = False) -> None:
        """Runs nested sampling on simulated data of the sample.

        Args:
            angle_times (list): points and times for each angle to simulate.
            save_path (str): path to directory to save corner plot to.
            filename (str): file name to use when saving corner plot.
            dynamic (bool): whether to use static or dynamic nested sampling.

        """
        # Simulate data for the sample.
        model = refnx.reflect.ReflectModel(self.get_structures()[0])
        data = SimulateReflectivity(model, angle_times).simulate()

        objective = Objective(model, data)

        # Sample the objective using nested sampling.
        sampler = Sampler(objective)
        fig = sampler.sample(dynamic=dynamic)

        # Save the sampling corner plot.
        save_path = os.path.join(save_path, self.name)
        save_plot(fig, save_path, filename + '_nested_sampling')


def simple_sample():
    """Defines a 2-layer simple sample.

    Returns:
        samples.Sample: structure in format for design optimisation.

    """
    air = refnx.reflect.SLD(0, name='Air')
    layer1 = refnx.reflect.SLD(4, name='Layer 1')(thick=100, rough=2)
    layer2 = refnx.reflect.SLD(8, name='Layer 2')(thick=150, rough=2)
    substrate = refnx.reflect.SLD(2.047, name='Substrate')(thick=0, rough=2)

    structure = air | layer1 | layer2 | substrate
    structure.name = 'simple_sample'
    return Sample(structure)


def many_param_sample():
    """Defines a 5-layer sample with many parameters.

    Returns:
        samples.Sample: structure in format for design optimisation.

    """
    air = refnx.reflect.SLD(0, name='Air')
    layer1 = refnx.reflect.SLD(2.0, name='Layer 1')(thick=50, rough=6)
    layer2 = refnx.reflect.SLD(1.7, name='Layer 2')(thick=15, rough=2)
    layer3 = refnx.reflect.SLD(0.8, name='Layer 3')(thick=60, rough=2)
    layer4 = refnx.reflect.SLD(3.2, name='Layer 4')(thick=40, rough=2)
    layer5 = refnx.reflect.SLD(4.0, name='Layer 5')(thick=18, rough=2)
    substrate = refnx.reflect.SLD(2.047, name='Substrate')(thick=0, rough=2)

    structure = air | layer1 | layer2 | layer3 | layer4 | layer5 | substrate
    structure.name = 'many_param_sample'
    return Sample(structure)


def thin_layer_sample_1():
    """Defines a 2-layer sample with thin layers.

    Returns:
        samples.Sample: structure in format for design optimisation.

    """
    air = refnx.reflect.SLD(0, name='Air')
    layer1 = refnx.reflect.SLD(4, name='Layer 1')(thick=200, rough=2)
    layer2 = refnx.reflect.SLD(6, name='Layer 2')(thick=6, rough=2)
    substrate = refnx.reflect.SLD(2.047, name='Substrate')(thick=0, rough=2)

    structure = air | layer1 | layer2 | substrate
    structure.name = 'thin_layer_sample_1'
    return Sample(structure)


def thin_layer_sample_2():
    """Defines a 3-layer sample with thin layers.

    Returns:
        samples.Sample: structure in format for design optimisation.

    """
    air = refnx.reflect.SLD(0, name='Air')
    layer1 = refnx.reflect.SLD(4, name='Layer 1')(thick=200, rough=2)
    layer2 = refnx.reflect.SLD(5, name='Layer 2')(thick=30, rough=6)
    layer3 = refnx.reflect.SLD(6, name='Layer 3')(thick=6, rough=2)
    substrate = refnx.reflect.SLD(2.047, name='Substrate')(thick=0, rough=2)

    structure = air | layer1 | layer2 | layer3 | substrate
    structure.name = 'thin_layer_sample_2'
    return Sample(structure)


def similar_sld_sample_1():
    """Defines a 2-layer sample with layers of similar SLD.

    Returns:
        samples.Sample: structure in format for design optimisation.

    """
    air = refnx.reflect.SLD(0, name='Air')
    layer1 = refnx.reflect.SLD(0.9, name='Layer 1')(thick=80, rough=2)
    layer2 = refnx.reflect.SLD(1.0, name='Layer 2')(thick=50, rough=6)
    substrate = refnx.reflect.SLD(2.047, name='Substrate')(thick=0, rough=2)

    structure = air | layer1 | layer2 | substrate
    structure.name = 'similar_sld_sample_1'
    return Sample(structure)


def similar_sld_sample_2():
    """Defines a 3-layer sample with layers of similar SLD.

    Returns:
        samples.Sample: structure in format for design optimisation.

    """
    air = refnx.reflect.SLD(0, name='Air')
    layer1 = refnx.reflect.SLD(3.0, name='Layer 1')(thick=50, rough=2)
    layer2 = refnx.reflect.SLD(5.5, name='Layer 2')(thick=30, rough=6)
    layer3 = refnx.reflect.SLD(6.0, name='Layer 3')(thick=35, rough=2)
    substrate = refnx.reflect.SLD(2.047, name='Substrate')(thick=0, rough=2)

    structure = air | layer1 | layer2 | layer3 | substrate
    structure.name = 'similar_sld_sample_2'
    return Sample(structure)


def run_main(save_path: Optional[str] = '../results') -> None:
    """
    Runs the main function of the module, retrieves an SLD and
    reflectivity profile for each defined structure, and saves it in the
    results directory by default.

    Args:
        save_path: The directory where the SLD and reflectivity profiles
        are saved
    """
    # Plot the SLD and reflectivity profiles of all structures in this file.
    for structure in [simple_sample, many_param_sample,
                      thin_layer_sample_1, thin_layer_sample_2,
                      similar_sld_sample_1, similar_sld_sample_2]:

        sample = structure()
        sample.sld_profile(save_path)
        sample.reflectivity_profile(save_path)

        # Close the plots.
        plt.close('all')


if __name__ == '__main__':
    run_main()
