import os
from typing import Union

import numpy as np

from dynesty import NestedSampler, DynamicNestedSampler
from dynesty import plotting as dyplot
from dynesty import utils as dyfunc

import refl1d.experiment
import refnx.reflect
import refnx.analysis
import bumps.parameter
import bumps.fitproblem

from hogben.simulate import reflectivity, simulate


class Sampler:
    """Contains code for running nested sampling on refnx and Refl1D models.

    Attributes:
        objective (refnx.analysis.Objective or
                   bumps.fitproblem.FitProblem): objective to sample.
        params (list): varying model parameters.
        ndim (int): number of varying model parameters.
        sampler_static (dynesty.NestedSampler): static nested sampler.
        sampler_dynamic (dynesty.DynamicNestedSampler): dynamic nested sampler.

    """

    def __init__(self, objective):
        self.objective = objective

        # Determine if the objective is from refnx or Refl1D.
        if isinstance(objective, refnx.analysis.BaseObjective):
            # Use log-likelihood and prior transform methods of refnx objective
            self.params = objective.varying_parameters()
            logl = objective.logl
            prior_transform = objective.prior_transform

        elif isinstance(objective, bumps.fitproblem.BaseFitProblem):
            # Use this class' custom log-likelihood and prior transform methods
            self.params = self.objective._parameters
            logl = self.logl_refl1d
            prior_transform = self.prior_transform_refl1d

        # Otherwise the given objective must be invalid.
        else:
            raise RuntimeError('invalid objective/fitproblem given')

        self.ndim = len(self.params)
        self.sampler_static = NestedSampler(logl, prior_transform, self.ndim)
        self.sampler_dynamic = DynamicNestedSampler(logl, prior_transform,
                                                    self.ndim)

    def logl_refl1d(self, x):
        """Calculates the log-likelihood of given parameter values `x`
           for a Refl1D FitProblem.

        Args:
            x (numpy.ndarray): parameter values to calculate likelihood of.

        Returns:
            float: log-likelihood of parameter values `x`.

        """
        self.objective.setp(x)  # Set the parameter values.
        return -self.objective.model_nllf()  # Calculate the log-likelihood.

    def prior_transform_refl1d(self, u):
        """Calculates the prior transform for a Refl1D FitProblem.

        Args:
            u (numpy.ndarray): values in interval [0,1] to be transformed.

        Returns:
            numpy.ndarray: `u` transformed to parameter space of interest.

        """
        return np.asarray([param.bounds.put01(u[i])
                           for i, param in enumerate(self.params)])

    def sample(self, verbose=True, dynamic=False):
        """Samples an Objective/FitPrffor layeroblem using nested sampling.

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


class Fisher():
    """Calculates the Fisher information matrix for multiple `models`
    containing parameters `xi`. The model describes the experiment,
    including the sample, and is defined using `refnx` or `refl1d`. The
    lower and upper bounds of each parameter in the model are transformed
    into a standardized range from 0 to 1, which is used to calculate the
    Fisher information matrix. Each parameter in the Fisher information
    matrix is scaled using an importance parameter. By default,
    the importance parameter is set to 1 for all parameters, and can be set
    by changing the `importance` attribute of the parameter when setting up
    the model. For example the relative importance of the thickness in
    "layer1" can be set to 2 using `layer1.thickness.importance = 2` or
    `layer1.thick.importance = 2` in `refnx` and `refl1d` respectively.

    Attributes:
        qs: The Q points for each model.
        xi: The varying model parameters.
        counts: incident neutron counts corresponding to each Q value.
        models: models to calculate gradients with.
        step: step size to take when calculating gradient.
        fisher_information: The Fisher information matrix
        min_eigenval: The minimum eigenvalue of the Fisher information matrix
    """

    def __init__(self,
                 qs: list[np.ndarray],
                 xi: list[Union['refnx.analysis.Parameter',
                                'bumps.parameter.Parameter']],
                 counts: list[int],
                 models: list[Union['refnx.reflect.ReflectModel',
                                    'refl1d.experiment.Experiment']],
                 step: float = 0.005):
        """Initialize the Fisher matrix class.

        Args:
            qs: The Q points for each model.
            xi: The varying model parameters.
            counts: incident neutron counts corresponding to each Q value.
            models: models to calculate gradients with.
            step: step size to take when calculating gradient.
        """
        self.qs = qs
        self.xi = xi
        self.counts = counts
        self.models = models
        self.step = step

    @property
    def fisher_information(self) -> np.ndarray:
        """Calculate and return the Fisher information matrix.

        Returns:
            numpy.ndarray: The Fisher information matrix.
        """
        return self._calculate_fisher_information()

    @property
    def min_eigenval(self) -> float:
        """Calculate and return the minimum eigenvalue of the Fisher
        information matrix.

        Returns:
            float: The minimum eigenvalue.
        """
        return np.linalg.eigvalsh(self.fisher_information)[0]

    @property
    def n(self) -> int:
        """The total number of datapoints.

        Returns:
            int: total number of datapoints.
        """
        return sum(len(q) for q in self.qs)

    @property
    def m(self) -> int:
        """The total number of parameters.

        Returns:
            int: total number of parameters.
        """
        return len(self.xi)

    @classmethod
    def from_sample(cls,
                    sample,
                    angle_times,
                    contrasts = None,
                    underlayers = None):
        """
        Get Fisher object using a sample.
        Seperate constructor for magnetic simulation maybe? Probably depends
        on new simulate function either way.
        """

        qs, counts, models = [], [], []
        if contrasts is None:
            model, data = simulate(
                sample.structure, angle_times, scale=1, bkg=2e-6, dq=2
            )
            qs.append(data[:, 0])
            counts.append(data[:, 3])
            models.append(model)
        else:
            for contrast in contrasts:
                contrast_point = (contrast + 0.56) / (6.35 + 0.56)
                background_level = (2e-6 * contrast_point
                                    + 4e-6 * (1 - contrast_point)
                                    )
                new_structure = sample._using_conditions(contrast, underlayers)
                model, data = simulate(
                    new_structure, angle_times, scale=1, bkg=background_level,
                    dq=2
                )
                qs.append(data[:, 0])
                counts.append(data[:, 3])
                models.append(model)

        xi = sample.get_params()
        return cls(qs, xi, counts, models)

    def _calculate_fisher_information(self) -> np.ndarray:
        """Calculates the Fisher information matrix using the class attributes.

        Returns:
            numpy.ndarray: The Fisher information matrix.
        """
        if self.n == 0:
            return np.zeros((self.m, self.m))
        J = self._get_gradient_matrix()

        # Calculate the reflectance for each model for the given Q values.
        r = np.concatenate([reflectivity(q, model)
                            for q, model in list(zip(self.qs, self.models))])

        # Calculate the Fisher information matrix using equations from
        # the paper.
        M = np.diag(np.concatenate(self.counts) / r, k=0)
        g = np.dot(np.dot(J.T, M), J)

        # Perform unit scaling if there's at least one parameter
        if len(self.xi) >= 1:
            g = self._scale_units(g)  # Scale by unit bounds
            g = self._scale_importance(g)  # Scale by importance

        return g

    def _scale_units(self, g: np.ndarray) -> np.ndarray:

        """Scale the values of the fisher information matrix for each parameter
        from interval [lb, ub] to the interval [0, 1]

        Args:
            g: The Fisher information matrix.

        Returns:
            numpy.ndarray: The scaled Fisher information matrix.
        """
        lb, ub = self._get_bounds()
        H = np.diag(1 / (ub - lb))  # Get unit scaling Jacobian.
        return np.dot(np.dot(H.T, g), H)  # Perform unit scaling.

    def _scale_importance(self, g: np.ndarray) -> np.ndarray:
        """Scale the Fisher information matrix using importance scaling.

        Args:
            g: The Fisher information matrix.

        Returns:
            numpy.ndarray: The scaled Fisher information matrix.
        """
        importance_array = [param.importance if hasattr(param, 'importance')
                            else 1 for param in self.xi]
        importance = np.diag(importance_array)
        return np.dot(g, importance)

    def _get_gradient_matrix(self) -> np.ndarray:
        """Calculate the gradient matrix.

        Returns:
            numpy.ndarray: The gradient matrix.
        """
        J = np.zeros((self.n, self.m))
        for i, parameter in enumerate(self.xi):
            old = parameter.value

            # Calculate reflectance for each model for first part of gradient.
            x1 = parameter.value = old * (1 - self.step)
            y1 = np.concatenate([reflectivity(q, model)
                                 for q, model in list(zip(self.qs,
                                                          self.models))])

            # Calculate reflectance for each model for second part of gradient.
            x2 = parameter.value = old * (1 + self.step)
            y2 = np.concatenate([reflectivity(q, model)
                                 for q, model in list(zip(self.qs,
                                                          self.models))])

            parameter.value = old  # Reset the parameter.

            J[:, i] = (y2 - y1) / (x2 - x1)  # Calculate the gradient.
        return J

    def _get_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the bounds from the refnx or refl1d model.

        Returns:
            tuple: The lower and upper bounds of the parameters.
        """
        if isinstance(self.xi[0], refnx.analysis.Parameter):
            lb = np.array([param.bounds.lb for param in self.xi])
            ub = np.array([param.bounds.ub for param in self.xi])

        elif isinstance(self.xi[0], bumps.parameter.Parameter):
            lb = np.array([param.bounds.limits[0] for param in self.xi])
            ub = np.array([param.bounds.limits[1] for param in self.xi])
        # Otherwise, the sample must be invalid.
        else:
            raise RuntimeError('Invalid sample given')
        return lb, ub


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
