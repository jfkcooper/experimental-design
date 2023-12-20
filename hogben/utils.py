"""General utility functions to run nested sampling, calculate the Fisher
information, and save plots"""

import os

import numpy as np

from dynesty import NestedSampler, DynamicNestedSampler
from dynesty import plotting as dyplot
from dynesty import utils as dyfunc

import refnx.reflect
import refnx.analysis

from hogben.simulate import SimulateReflectivity


class Sampler:
    """Contains code for running nested sampling on refnx models.

    Attributes:
        objective (refnx.analysis.Objective): objective to sample.
        params (list): varying model parameters.
        ndim (int): number of varying model parameters.
        sampler_static (dynesty.NestedSampler): static nested sampler.
        sampler_dynamic (dynesty.DynamicNestedSampler): dynamic nested sampler.

    """

    def __init__(self, objective):
        """
        Initialise the sample given an objective to the sample

        Args:
            objective: objective to the sample
        """
        self.objective = objective

        # Use log-likelihood and prior transform methods of refnx objective
        self.params = objective.varying_parameters()
        logl = objective.logl
        prior_transform = objective.prior_transform

        self.ndim = len(self.params)
        self.sampler_static = NestedSampler(logl, prior_transform, self.ndim)
        self.sampler_dynamic = DynamicNestedSampler(logl, prior_transform,
                                                    self.ndim)

    def sample(self, verbose=True, dynamic=False):
        """Samples an Objective/FitProblem using nested sampling.

        Args:
            verbose (bool): whether to display sampling progress.
            dynamic (bool): whether to use static or dynamic nested sampling.

        Returns:
            matplotlib.pyplot.Figure or float: corner plot.

        """
        # Run either static or dynamic nested sampling.
        if dynamic:
            # Weighting is entirely on the posterior (0 weight on evidence).
            self.sampler_dynamic.run_nested(print_progress=verbose,
                                            wt_kwargs={'pfrac': 1.0})
            results = self.sampler_dynamic.results

        else:
            self.sampler_static.run_nested(print_progress=verbose)
            results = self.sampler_static.results

        # Calculate the parameter means.
        weights = np.exp(results.logwt - results.logz[-1])
        mean, _ = dyfunc.mean_and_cov(results.samples, weights)

        # Set the parameter values to the estimated means.
        for i, param in enumerate(self.params):
            param.value = mean[i]

        # Return the corner plot
        return self.__corner(results)

    def __corner(self, results):
        """Calculates a corner plot from given nested sampling `results`.

        Args:
            results (dynesty.results.Results): full output of a sampling run.

        Returns:
            matplotlib.pyplot.Figure: nested sampling corner plot.

        """
        # Get the corner plot from dynesty package.
        fig, _ = dyplot.cornerplot(results, color='blue', quantiles=None,
                                   show_titles=True, max_n_ticks=3,
                                   truths=np.zeros(self.ndim),
                                   truth_color='black')

        # Label the axes with parameter labels.
        axes = np.reshape(np.array(fig.get_axes()), (self.ndim, self.ndim))
        for i in range(1, self.ndim):
            for j in range(self.ndim):
                if i == self.ndim - 1:
                    axes[i, j].set_xlabel(self.params[j].name)
                if j == 0:
                    axes[i, j].set_ylabel(self.params[i].name)

        axes[self.ndim - 1, self.ndim - 1].set_xlabel(self.params[-1].name)
        return fig


def fisher(qs: np.ndarray,
           xi: list['refnx.analysis.Parameter'],
           counts: np.ndarray,
           models: list['refnx.reflect.ReflectModel'],
           step: float = 0.005) -> np.ndarray:
    """Calculates the Fisher information matrix for multiple `models`
    containing parameters `xi`. The model describes the experiment,
    including the sample, and is defined using `refnx`. The
    lower and upper bounds of each parameter in the model are transformed
    into a standardized range from 0 to 1, which is used to calculate the
    Fisher information matrix. Each parameter in the Fisher information
    matrix is scaled using an importance parameter. By default,
    the importance parameter is set to 1 for all parameters, and can be set
    by changing the `importance` attribute of the parameter when setting up
    the model. For example the relative importance of the thickness in
    "layer1" can be set to 2 using `layer1.thickness.importance = 2`.

    Args:
        qs: The Q points for each model.
        xi: The varying model parameters.
        counts: incident neutron counts corresponding to each Q value.
        models: models to calculate gradients with.
        step: step size to take when calculating gradient.
    Returns:
        numpy.ndarray: Fisher information matrix for given models and data.

    """
    n = sum(len(q) for q in qs)  # Number of data points.
    m = len(xi)  # Number of parameters.
    J = np.zeros((n, m))

    # There is no information if there is no data.
    if n == 0:
        return np.zeros((m, m))

    # Calculate the gradient of model reflectivity with every model parameter
    # for every model data point.
    for i, parameter in enumerate(xi):
        old = parameter.value

        # Calculate reflectance for each model for first part of gradient.
        x1 = parameter.value = old * (1 - step)
        y1 = np.concatenate([SimulateReflectivity(model).reflectivity(q)
                             for q, model in list(zip(qs, models))])

        # Calculate reflectance for each model for second part of gradient.
        x2 = parameter.value = old * (1 + step)
        y2 = np.concatenate([SimulateReflectivity(model).reflectivity(q)
                             for q, model in list(zip(qs, models))])

        parameter.value = old  # Reset the parameter.

        J[:, i] = (y2 - y1) / (x2 - x1)  # Calculate the gradient.

    # Calculate the reflectance for each model for the given Q values.
    r = np.concatenate([SimulateReflectivity(model).reflectivity(q)
                        for q, model in list(zip(qs, models))])

    # Calculate the Fisher information matrix using equations from the paper.
    M = np.diag(np.concatenate(counts) / r, k=0)
    g = np.dot(np.dot(J.T, M), J)

    # If there are multiple parameters, scale each parameter's
    # information by its bounds and "importance":
    if len(xi) > 1:
        lb = np.array([param.bounds.lb for param in xi])
        ub = np.array([param.bounds.ub for param in xi])

        # Scale each parameter with their specified importance,
        # scale with one if no importance was specified.
        importance_array = []
        for param in xi:
            if hasattr(param, 'importance'):
                importance_array.append(param.importance)
            else:
                importance_array.append(1)
        importance = np.diag(importance_array)
        H = np.diag(1 / (ub - lb))  # Get unit scaling Jacobian.
        g = np.dot(np.dot(H.T, g), H)  # Perform unit scaling.
        g = np.dot(g, importance)  # Perform importance scaling.

        # Return the Fisher information matrix.
    return g


def save_plot(fig, save_path, filename):
    """Saves a figure to a given directory.

    Args:
        fig (matplotlib.pyplot.Figure): figure to save.
        save_path (str): path to directory to save figure to.
        filename (str): name of file to save plot as.

    """
    # Create the directory if not present.
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    file_path = os.path.join(save_path, filename + '.png')
    fig.savefig(file_path, dpi=600)
