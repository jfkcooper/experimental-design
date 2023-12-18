"""Base classes for different sample types """

import os
from abc import ABC, abstractmethod
from typing import Optional

import matplotlib.pyplot as plt

import refnx.dataset
import refnx.reflect
import refnx.analysis
from refnx.reflect import ReflectModel

from hogben.simulate import SimulateReflectivity
from hogben.utils import fisher, Sampler, save_plot

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
        """Calculates the Fisher information matrix for the lipid sample
           with given conditions.

        Args:
            angle_times (list): points and times for each angle to simulate.
            contrasts (list): SLDs of contrasts to simulate.
            underlayers (list): thickness and SLD of each underlayer to add.

        Returns:
            numpy.ndarray: Fisher information matrix.

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
            return fisher(qs, self.params, counts, models)
        else:
            return fisher(qs, self.underlayer_params, counts, models)

    @abstractmethod
    def _using_conditions(self):
        """Creates a structure describing the given measurement conditions."""
        pass

    def sld_profile(self,
                    save_path: str,
                    filename: str = 'sld_profile',
                    ylim: Optional[tuple] = None,
                    legend: bool = True) -> None:
        """Plots the SLD profile of the lipid sample.

        Args:
            save_path (str): path to directory to save SLD profile to.
            filename (str): file name to use when saving the SLD profile.
            ylim (tuple): limits to place on the SLD profile y-axis.
            legend (bool): whether to include a legend in the SLD profile.

        """
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # Plot the SLD profile for each measured contrast.
        for structure in self.structures:
            ax.plot(*structure.sld_profile(self.distances))

        x_label = '$\mathregular{Distance\ (\AA)}$'
        y_label = '$\mathregular{SLD\ (10^{-6} \AA^{-2})}$'

        ax.set_xlabel(x_label, fontsize=11, weight='bold')
        ax.set_ylabel(y_label, fontsize=11, weight='bold')

        # Limit the y-axis if specified.
        if ylim:
            ax.set_ylim(*ylim)

        # Add a legend if specified.
        if legend:
            ax.legend(self.labels, loc='upper left')

        # Save the plot.
        save_path = os.path.join(save_path, self.name)
        save_plot(fig, save_path, filename)

    def reflectivity_profile(self,
                             save_path: str,
                             filename: str = 'reflectivity_profile') -> None:
        """Plots the reflectivity profile of the lipid sample.

        Args:
            save_path (str): path to directory to save profile to.
            filename (str): file name to use when saving the profile.

        """
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # Iterate over each measured contrast.
        colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for i, objective in enumerate(self.objectives):
            # Get the measured data and calculate the model reflectivity.
            q, r, dr = objective.data.x, objective.data.y, objective.data.y_err
            r_model = objective.model(q)

            # Offset the data, for clarity.
            offset = 10 ** (-2 * i)
            r *= offset
            dr *= offset
            r_model *= offset

            # Add the offset in the label.
            label = self.labels[i]
            if offset != 1:
                label += ' $\\mathregular{(x10^{-' + str(2 * i) + '})}$'

            # Plot the measured data and the model reflectivity.
            ax.errorbar(q, r, dr,
                        marker='o', ms=3, lw=0, elinewidth=1, capsize=1.5,
                        color=colours[i], label=label)
            ax.plot(q, r_model, color=colours[i], zorder=20)

        x_label = '$\\mathregular{Q\\ (Å^{-1})}$'
        y_label = 'Reflectivity (arb.)'

        ax.set_xlabel(x_label, fontsize=11, weight='bold')
        ax.set_ylabel(y_label, fontsize=11, weight='bold')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim(1e-10, 3)
        ax.legend()

        # Save the plot.
        save_path = os.path.join(save_path, self.name)
        save_plot(fig, save_path, filename)

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
