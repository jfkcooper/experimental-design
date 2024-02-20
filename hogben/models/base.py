"""Base classes for different sample types """

import os, copy
from abc import ABC, abstractmethod
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

import refnx.dataset
import refnx.reflect
import refnx.analysis
from refnx.reflect import ReflectModel

from hogben.simulate import SimulateReflectivity
from hogben.utils import Fisher, Sampler, save_plot, flatten, sig_fig_round

plt.rcParams['figure.figsize'] = (9, 7)
plt.rcParams['figure.dpi'] = 600


class VariableAngle(ABC):
    """Abstract class representing whether the measurement angle of a sample
       can be varied."""

    @abstractmethod
    def angle_info(self):

        """Calculates the Fisher information matrix for a sample measured
        over a number of angles."""
        pass


class VariableContrast(ABC):
    """Abstract class representing whether the contrast of a sample
       dan be varied."""

    @abstractmethod
    def contrast_info(self):
        """Calculates the Fisher information matrix for a sample with contrasts
           measured over a number of angles."""
        pass


class VariableUnderlayer(ABC):
    """Abstract class representing whether the underlayer(s) of a sample
       can be varied."""

    @abstractmethod
    def underlayer_info(self):
        """Calculates the Fisher information matrix for a sample with
        underlayers, and contrasts measured over a number of angles."""
        pass


class BaseSample(VariableAngle):
    """Abstract class representing a "standard" neutron reflectometry sample
    defined by a series of contiguous layers."""

    @abstractmethod
    def sld_profile(self):
        """Plots the SLD profile of the sample."""
        pass

    @abstractmethod
    def reflectivity_profile(self):
        """Plots the reflectivity profile of the sample."""
        pass

    @abstractmethod
    def nested_sampling(self):
        """Runs nested sampling on measured or simulated data of the sample."""
        pass

    def scan_parameter(self, param, angle_times):
        lb, ub = param.bounds.lb, param.bounds.ub
        old_value = param.value
        param_range = np.linspace(lb, ub, 100)
        eigenvals = []
        fig, ax = plt.subplots()
        for value in param_range:
            param.value = value
            eigenvals.append(Fisher.from_sample(self, angle_times).min_eigenval)
        ax.plot(param_range, eigenvals)
        ax.set_title(param.name)
        ax.set_ylabel("Minimum eigenvalue")
        ax.set_xlabel("Parameter value")

        param.value = old_value


    @property
    def underlayers_indices(self):
        underlayer_list = []
        if not hasattr(self, "base_structures"):
            self.base_structures = self.structures
        for structure in self.base_structures:
            underlayers = []
            for index, layer in enumerate(structure):
                if hasattr(layer, "underlayer") and layer.underlayer:
                    underlayers.append(index)
            underlayer_list.append(underlayers)
        return underlayer_list

    def _remove_underlayers(self):
        delete_index = self.underlayers_indices
        delete_index.reverse()
        for structure, indices in zip(self.structures, delete_index):
            for index in indices:
                del structure[index]
        return self

    def get_optimization_parameters(self):
        """Get list of parameters that are varying
        Current implementation won't win any beauty prices, but works as
        temporary solution. Will clean this a bit...
        """
        #if not hasattr(self, "model"):
            # This is kinda a temp. workaround, for the predefined cases
            # where no model is defined (so it doesn't crash on the lack
            # of self.model.parameters).
         #   return self.params
        params = []
        for model in self.get_models():
            for p in flatten(model.parameters):
                if hasattr(p, "optimize") and p.optimize:
                    params.append(p)
                    continue
                if len(p._deps):
                    params.extend([_p for _p in p.dependencies() if hasattr(_p, "optimize") and _p.optimize])
        return list(set(params))

    def get_varying_parameters(self):
        """Get list of parameters that are varying
        Current implementation won't win any beauty prices, but works as
        temporary solution. Will clean this a bit...
        """
        #if not hasattr(self, "model"):
            # This is kinda a temp. workaround, for the predefined cases
            # where no model is defined (so it doesn't crash on the lack
            # of self.model.parameters).
         #   return self.params
        params = []
        for model in self.get_models():
            for p in flatten(model.parameters):
                if p.vary:
                    params.append(p)
                    continue
                if len(p._deps):
                    params.extend([_p for _p in p.dependencies() if _p.vary])
        return list(set(params))

    def get_models(self):
        # Add code to set background for each structure with D2O
        return [refnx.reflect.ReflectModel(structure,
                                       scale=self.scale,
                                       bkg=self.bkg,
                                       dq=self.dq)
            for structure in self.get_structures()]

    def get_structures(self):
        """
        Get a list of the possible sample structures.
        """
        spin_structures = []
        from hogben.models.samples import MagneticLayerSLD
        if not hasattr(self, "base_structures"):
            self.base_structures = self.structures
        for structure in self.base_structures:
            magnetic = False
            up_structure = structure.copy()
            down_structure = structure.copy()
            for i, layer in enumerate(structure):
                if isinstance(layer, MagneticLayerSLD):
                    magnetic = True
                    up_structure[i] = layer.spin_up
                    down_structure[i] = layer.spin_down
            if magnetic:
                spin_structures.extend([up_structure, down_structure])
            else:
                spin_structures.extend([structure.copy()])
        if magnetic:
            return spin_structures
        return self.base_structures

    def sld_profile(self, save_path=None, single=True):
        """Plots the SLD profile of the sample.

        Args:
            save_path (str): path to directory to save SLD profile to.
            single (bool): whether to plot all profiles on a single plot or
            separate ones.

        """
        # Create a figure and axes based on the 'single' parameter
        fig, ax = plt.subplots() if single else (plt.figure(), None)
        for i, (z, slds) in enumerate(self._get_sld_profile()):
            # Create a new subplot for each profile if not 'single'
            if self.underlayers_indices[0]:
                label = \
                    self.get_structures()[i][self.underlayers_indices[0][0]].name
            else:
                label = f"SLD profile {i}"

            # Create a new subplot for each profile if not 'single'
            if not single:
                ax = fig.add_subplot(len(self.get_structures()), 1, i + 1)
                ax.set_title(f"SLD Profile {label}")
                fig.subplots_adjust(hspace=0.5)
            ax.plot(z, slds, label=label)

            ax.set_xlabel('$\mathregular{Distance\ (\AA)}$', fontsize=11,
                          weight='bold')
            ax.set_ylabel('$\mathregular{SLD\ (10^{-6} \AA^{-2})}$',
                          fontsize=11, weight='bold')
            # Add a legend if 'single'
            if single:
                ax.set_title("SLD Profile")
                ax.legend()

        # Save the plot.
        if save_path:
            save_path = os.path.join(save_path, self.name)
            save_plot(fig, save_path, 'sld_profile')

    def _get_sld_profile(self):
        """
        Obtains the SLD profile of the sample, in terms of z (depth) vs SLD

        Returns:
            numpy.ndarray: depth
            numpy.ndarray: SLD values
        """
        return [structure.sld_profile() for structure in self.get_structures()]

    def reflectivity_profile(self,
                             save_path: str = None,
                             q_min: float = 0.005,
                             q_max: float = 0.4,
                             points: int = 500,
                             scale: float = 1,
                             bkg: float = 1e-7,
                             dq: float = 2,
                             single = True,
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
        fig, ax = plt.subplots() if single else (plt.figure(), None)
        profiles = self._get_reflectivity_profile(q_min, q_max, points, scale,
                                              bkg, dq)
        for i, (q, r) in enumerate(profiles):
            if self.underlayers_indices[0]:
                label = \
                    self.get_structures()[i][self.underlayers_indices[0][0]].name
            else:
                label = f"Reflectivity profile {i}"

            # Create a new subplot for each profile if not 'single'
            if not single:
                ax = fig.add_subplot(len(self.get_structures()), 1, i + 1)
                ax.set_title(f"Reflectivity profile {label}")
                fig.subplots_adjust(hspace=0.5)

            # Plot Q versus model reflectivity.
            ax.plot(q, r, label=label)

            x_label = '$\mathregular{Q\ (Å^{-1})}$'
            y_label = 'Reflectivity (arb.)'

            ax.set_xlabel(x_label, fontsize=11, weight='bold')
            ax.set_ylabel(y_label, fontsize=11, weight='bold')
            ax.set_yscale('log')

            # Add a legend if 'single'
            if single:
                ax.set_title(f"Reflectivity profile")
                ax.legend()


        if save_path:
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
        profiles = []
        for structure in self.get_structures():
            # Geometriaclly-space Q points over the specified range.
            q = np.geomspace(q_min, q_max, points)

            # Determine if the structure was defined in refnx.
            model = refnx.reflect.ReflectModel(structure, scale=scale,
                                               bkg=bkg, dq=dq)
            r = SimulateReflectivity(model).reflectivity(q)
            profiles.append((q, r))
        return profiles

class BaseLipid(BaseSample, VariableContrast, VariableUnderlayer):
    """Abstract class representing the base class for a lipid model."""

    def __init__(self):
        """
        Initialize a BaseLipid object sample, and loads the
        experimentally measured data
        """
        self._create_objectives()  # Load experimentally-measured data.

    @abstractmethod
    def _create_objectives(self):
        """Loads the measured data for the lipid sample."""
        pass

    def angle_info(self, angle_times, contrasts):
        """Calculates the Fisher information matrix for the lipid sample
           measured over a number of angles.

        Args:
            angle_times (list): points and times for each angle to simulate.
            contrasts (list): SLDs of contrasts to simulate.

        Returns:
            numpy.ndarray: Fisher information matrix.

        """
        return self.__conditions_info(angle_times, contrasts, None)

    def contrast_info(self, angle_times, contrasts):
        """Calculates the Fisher information matrix for the lipid sample
           with contrasts measured over a number of angles.

        Args:
            angle_times (list): points and times for each angle to simulate.
            contrasts (list): SLDs of contrasts to simulate.

        Returns:
            numpy.ndarray: Fisher information matrix.

        """
        return self.__conditions_info(angle_times, contrasts, None)

    def underlayer_info(self, angle_times, contrasts, underlayers):
        """Calculates the Fisher information matrix for the lipid sample with
           `underlayers`, and `contrasts` measured over a number of angles.

        Args:
            angle_times (list): points and times for each angle to simulate.
            contrasts (list): SLDs of contrasts to simulate.
            underlayers (list): thickness and SLD of each underlayer to add.

        Returns:
            numpy.ndarray: Fisher information matrix.

        """
        return self.__conditions_info(angle_times, contrasts, underlayers)

    def __conditions_info(self, angle_times, contrasts, underlayers):
        """Calculates the Fisher information object for the lipid sample
           with given conditions.

        Args:
            angle_times (list): points and times for each angle to simulate.
            contrasts (list): SLDs of contrasts to simulate.
            underlayers (list): thickness and SLD of each underlayer to add.

        Returns:
            Fisher: Fisher information matrix object

        """
        # Iterate over each contrast to simulate.
        qs, counts, models = [], [], []

        for contrast in contrasts:
            # Simulate data for the contrast.
            sample = self._using_conditions(contrast, underlayers)
            contrast_point = (contrast + 0.56) / (6.35 + 0.56)
            background_level = (2e-6 * contrast_point
                                + 4e-6 * (1 - contrast_point))
            model = ReflectModel(sample)
            model.bkg = background_level
            model.dq = 2
            data = SimulateReflectivity(model, angle_times).simulate()
            qs.append(data[0])
            counts.append(data[3])
            models.append(model)

        # Exclude certain parameters if underlayers are being used.
        if underlayers is None:
            return Fisher(qs, self.params, counts, models)
        else:
            return Fisher(qs, self.underlayer_params, counts, models)

    @abstractmethod
    def _using_conditions(self):
        """Creates a structure describing the given measurement conditions."""
        pass

    def nested_sampling(self,
                        contrasts: list,
                        angle_times: list,
                        save_path: str,
                        filename: str,
                        underlayers=None,
                        dynamic=False) -> None:
        """Runs nested sampling on simulated data of the lipid sample.

        Args:
            contrasts (list): SLDs of contrasts to simulate.
            angle_times (list): points and times for each angle to simulate.
            save_path (str): path to directory to save corner plot to.
            filename (str): file name to use when saving corner plot.
            underlayers (list): thickness and SLD of each underlayer to add.
            dynamic (bool): whether to use static or dynamic nested sampling.

        """
        # Create objectives for each contrast to sample with.
        objectives = []
        for contrast in contrasts:
            # Simulate an experiment using the given contrast.
            sample = self._using_conditions(contrast, underlayers)
            contrast_point = (contrast + 0.56) / (6.35 + 0.56)
            background_level = 2e-6 * contrast_point + 4e-6 * (
                1 - contrast_point)

            model = ReflectModel(sample)
            model.bkg = background_level
            model.dq = 2
            data = SimulateReflectivity(model, angle_times).simulate()

            dataset = refnx.dataset.ReflectDataset(
                [data[0], data[1], data[2]]
            )
            objectives.append(refnx.analysis.Objective(model, dataset))

        # Combine objectives into a single global objective.
        global_objective = refnx.analysis.GlobalObjective(objectives)

        # Exclude certain parameters if underlayers are being used.
        if underlayers is None:
            global_objective.varying_parameters = lambda: self.params
        else:
            global_objective.varying_parameters = (
                lambda: self.underlayer_params
            )

        # Sample the objective using nested sampling.
        sampler = Sampler(global_objective)
        fig = sampler.sample(dynamic=dynamic)

        # Save the sampling corner plot.
        save_path = os.path.join(save_path, self.name)
        save_plot(fig, save_path, 'nested_sampling_' + filename)
