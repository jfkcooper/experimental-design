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
from hogben.utils import fisher, Sampler, save_plot
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

    def __init__(self, structure):
        """
        Initializes a sample given a structure, and sets the sample name and
        parameters

        Args:
            structure: Sample structure defined in the refnx model
        """
        self.structure = structure
        self.name = structure.name
        self.params = Sample.__vary_structure(structure)

    @staticmethod
    def __vary_structure(structure, bound_size=0.2):
        """Varies the SLD and thickness of each layer of a given `structure`.

        Args:
            structure (refnx.reflect.Structure): structure to vary.
            bound_size (float): size of bounds to place on varying parameters.

        Returns:
            list: varying parameters of sample.

        """
        params = []

        # Vary the SLD and thickness of each component (layer).
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

        return params

    def angle_info(self, angle_times, contrasts=None):
        """Calculates the Fisher information matrix for a sample measured
           over a number of angles.

        Args:
            angle_times (list): points and times for each angle to simulate.

        Returns:
            numpy.ndarray: Fisher information matrix.

        """
        # Return the Fisher information matrix calculated from simulated data.
        model = refnx.reflect.ReflectModel(self.structure)
        data = SimulateReflectivity(model, angle_times).simulate()
        qs, counts, models = [data[0]], [data[3]], [model]
        return fisher(qs, self.params, counts, models)

    def sld_profile(self, save_path):
        """Plots the SLD profile of the sample.

        Args:
            save_path (str): path to directory to save SLD profile to.

        """
        # Determine if the structure was defined in refnx.

        z, slds = self._get_sld_profile()

        fig = plt.figure()
        ax = fig.add_subplot(111)

        # Plot the SLD profile.
        ax.plot(z, slds, color='black', label=self.name)

        x_label = '$\mathregular{Distance\ (\AA)}$'
        y_label = '$\mathregular{SLD\ (10^{-6} \AA^{-2})}$'

        ax.set_xlabel(x_label, fontsize=11, weight='bold')
        ax.set_ylabel(y_label, fontsize=11, weight='bold')

        # Save the plot.
        save_path = os.path.join(save_path, self.name)
        save_plot(fig, save_path, 'sld_profile')

    def _get_sld_profile(self):
        """
        Obtains the SLD profile of the sample, in terms of z (depth) vs SLD

        Returns:
            numpy.ndarray: depth
            numpy.ndarray: SLD values
        """
        z, slds = self.structure.sld_profile()

        return z, slds

    def reflectivity_profile(self,
                             save_path: str,
                             q_min: float = 0.005,
                             q_max: float = 0.4,
                             points: int = 500,
                             scale: float = 1,
                             bkg: float = 1e-7,
                             dq: float = 2,
                             ) -> None:
        """Plots the reflectivity profile of the sample.

        Args:
            save_path (str): path to directory to save reflectivity profile to.
            q_min (float): minimum Q value to plot.
            q_max (float): maximum Q value to plot.
            points (int): number of points to plot.
            scale (float): experimental scale factor.
            bkg (float): level of instrument background noise.
            dq (float): instrument resolution.

        """
        # Calculate the model reflectivity.
        q, r = self._get_reflectivity_profile(q_min, q_max, points, scale,
                                              bkg, dq)

        # Plot Q versus model reflectivity.
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(q, r, color='black')

        x_label = '$\mathregular{Q\ (Å^{-1})}$'
        y_label = 'Reflectivity (arb.)'

        ax.set_xlabel(x_label, fontsize=11, weight='bold')
        ax.set_ylabel(y_label, fontsize=11, weight='bold')
        ax.set_yscale('log')

        # Save the plot.
        save_path = os.path.join(save_path, self.name)
        save_plot(fig, save_path, 'reflectivity_profile')

    def _get_reflectivity_profile(self, q_min, q_max, points, scale, bkg, dq):
        """
        Obtains the reflectivity profile of the sample, in terms of q
        vs r

        Returns:
            numpy.ndarray: q values at each reflectivity point
            numpy.ndarray: model reflectivity values
        """
        # Geometriaclly-space Q points over the specified range.
        q = np.geomspace(q_min, q_max, points)

        # Determine if the structure was defined in refnx.
        model = refnx.reflect.ReflectModel(self.structure, scale=scale,
                                           bkg=bkg, dq=dq)
        r = SimulateReflectivity(model).reflectivity(q)
        return q, r

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
        model = refnx.reflect.ReflectModel(self.structure)
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
