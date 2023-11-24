"""Module containing the Optimiser class used to optimise a neutron
reflectometry experiment"""

import numpy as np
import copy
from typing import Optional

from scipy.optimize import differential_evolution, NonlinearConstraint

from hogben.models.base import (
    BaseSample,
    VariableAngle,
    VariableContrast,
    VariableUnderlayer,
)
from hogben.utils import Fisher
from hogben.simulate import reflectivity, simulate


class Optimiser:
    """Contains code for optimising a neutron reflectometry experiment.

    Attributes:
        sample (base.BaseSample): sample to optimise an experiment for.

    """

    def __init__(self, sample: BaseSample):
        """
        Initializes Optimiser given a sample

        Args:
            sample: The sample to optimise an experiment for
        """
        self.sample = sample

    def optimise_angle_times(
        self,
        num_angles: int,
        contrasts: Optional[list] = None,
        total_time: float = 1000,
        angle_bounds: tuple = (0.2, 4),
        points: int = 100,
        workers: int = -1,
        verbose: bool = True,
    ) -> tuple:
        """Optimises the measurement angles and associated counting times
           of an experiment, given a fixed time budget.

        Args:
            num_angles (int): number of angles to optimise.
            contrasts (list): contrasts of the experiment, if applicable.
            total_time (float): time budget of the experiment.
            angle_bounds (tuple): interval containing angles to consider.
            points (int): number of data points to use for each angle.
            workers (int): number of CPU cores to use when optimising.
            verbose (bool): whether to display progress or not.

        Returns:
            tuple: optimised angles, counting times and the corresponding
                   optimisation function value.

        """
        # Check that the measurement angle of the sample can be varied.
        assert isinstance(self.sample, VariableAngle)

        # Define bounds on each condition to optimise (angles and time splits).
        bounds = [angle_bounds] * num_angles + [(0, 1)] * num_angles

        # Arguments for the optimisation function.
        args = [num_angles, contrasts, points, total_time]

        # Constrain the counting times to sum to the fixed time budget.
        # Also constrain the angles to be in non-decreasing order.
        def _sum_of_splits(x):
            """
            Sets the constraint for the counting times to the sum of the
            fixed time budget
            """
            return sum(x[num_angles:])

        def _non_decreasing(x):
            """
            Sets the constraint for the angles to be in non-decreasing
            order
            """
            return int(np.all(np.diff(x[:num_angles]) >= 0))

        # Set both constrains as equality constraints
        constraints = [NonlinearConstraint(_sum_of_splits, 1, 1),
                       NonlinearConstraint(_non_decreasing, 1, 1)]

        # Optimise angles and times, and return the results.
        res, val = Optimiser.__optimise(self._angle_times_func, bounds,
                                        constraints, args, workers, verbose)
        return res[:num_angles], res[num_angles:], val

    def optimise_contrasts(
        self,
        num_contrasts: int,
        angle_splits: list,
        total_time: float = 1000,
        contrast_bounds: tuple = (-0.56, 6.36),
        workers: int = -1,
        verbose: bool = True,
    ) -> tuple:
        """Finds the optimal contrasts, given a fixed time budget.

        Args:
            num_contrasts (int): number of contrasts to optimise.
            angle_splits (list): points and proportion of time for each angle.
            total_time (float): time budget for the experiment.
            contrast_bounds (tuple): contrast to consider.
            workers (int): number of CPU cores to use when optimising.
            verbose (bool): whether to display progress or not.

        Returns:
            tuple: optimised contrast SLDs, counting time proportions and the
                   corresponding optimisation function value.

        """
        # Check that the contrast SLD of the sample can be varied.
        assert isinstance(self.sample, VariableContrast)

        # Define the bounds on each condition to optimise
        # (contrast SLDs and time splits).
        bounds = [contrast_bounds] * num_contrasts + [(0, 1)] * num_contrasts

        # Constrain the counting times to sum to the fixed time budget.
        # Also constrain the contrasts to be in non-decreasing order.
        def _sum_of_splits(x):
            """
            Sets the constraint for the counting times to the sum of the
            fixed time budget
            """
            return sum(x[num_contrasts:])

        def _non_decreasing(x):
            """
            Sets the constraint for the contrasts to be in non-decreasing
            order
            """
            return int(np.all(np.diff(x[:num_contrasts]) >= 0))

        # Set both constrains as equality constraints
        constraints = [
            NonlinearConstraint(_sum_of_splits, 1, 1),
            NonlinearConstraint(_non_decreasing, 1, 1),
        ]

        # Arguments for the optimisation function.
        args = [num_contrasts, angle_splits, total_time]

        # Optimise contrasts and counting time splits, and return the results.
        res, val = Optimiser.__optimise(
            self._contrasts_func, bounds, constraints, args, workers, verbose
        )
        return res[:num_contrasts], res[num_contrasts:], val

    def optimise_underlayers(
        self,
        num_underlayers,
        angle_times,
        contrasts,
        thick_bounds=(0, 500),
        sld_bounds=(1, 9),
        workers=-1,
        verbose=True,
    ) -> tuple:
        """Finds the optimal underlayer thicknesses and SLDs of a sample.

        Args:
            num_underlayers (int): number of underlayers to optimise.
            angle_times (list): points and times for each angle to simulate.
            contrasts (list): contrasts to simulate.
            thick_bounds (tuple): underlayer thicknesses to consider.
            sld_bounds (tuple): underlayer SLDs to consider.
            workers (int): number of CPU cores to use when optimising.
            verbose (bool): whether to display progress or not.

        Returns:
            tuple: optimised underlayer thicknesses and SLD, and the
                   corresponding optimisation function value.

        """
        # Check that the underlayers of the sample can be varied.
        assert isinstance(self.sample, VariableUnderlayer)

        # Define bounds on each condition to optimise
        # (underlayer thicknesses and SLDs).
        bounds = [thick_bounds] * num_underlayers \
                 + [sld_bounds] * num_underlayers

        # Arguments for the optimisation function.
        args = [num_underlayers, angle_times, contrasts]

        # Optimise underlayer thicknesses and SLDs, and return the results.
        res, val = Optimiser.__optimise(
            self._underlayers_func, bounds, [], args, workers, verbose
        )
        return res[:num_underlayers], res[num_underlayers:], val

    def optimise_underlayer_experiment(
        self,
        sample,
        conditions,
        workers=-1,
        verbose=True,
    ) -> tuple:
        """Finds the optimal underlayer thicknesses and SLDs of a sample.

        Args:
            num_underlayers (int): number of underlayers to optimise.
            angle_times (list): points and times for each angle to simulate.
            contrasts (list): contrasts to simulate.
            thick_bounds (tuple): underlayer thicknesses to consider.
            sld_bounds (tuple): underlayer SLDs to consider.
            workers (int): number of CPU cores to use when optimising.
            verbose (bool): whether to display progress or not.

        Returns:
            tuple: optimised underlayer thicknesses and SLD, and the
                   corresponding optimisation function value.

        """
        # Check that the underlayers of the sample can be varied.
        #angle_times = experiment.angle_times
        conditions = {} if conditions is None else conditions
        angle_times = conditions.get("angle_times", None)
        thick_bounds = conditions.get("thick_bounds", None)
        sld_bounds = conditions.get("sld_bounds", None)
        num_underlayers = sample.num_underlayers
        bounds = [thick_bounds] * num_underlayers + [sld_bounds] * num_underlayers

        # Arguments for the optimisation function.
        args = [sample, angle_times]

        # Optimise underlayer thicknesses and SLDs, and return the results.
        res, val = Optimiser.__optimise(
            self._underlayers_func2, bounds, [], args, workers, verbose
        )
        return res[:num_underlayers], res[num_underlayers:], val


    def optimise_sld(
        self,
        sample,
        conditions,
        workers=-1,
        verbose=True,
    ) -> tuple:
        """Finds the optimal underlayer thicknesses and SLDs of a sample.

        Args:
            num_underlayers (int): number of underlayers to optimise.
            angle_times (list): points and times for each angle to simulate.
            contrasts (list): contrasts to simulate.
            thick_bounds (tuple): underlayer thicknesses to consider.
            sld_bounds (tuple): underlayer SLDs to consider.
            workers (int): number of CPU cores to use when optimising.
            verbose (bool): whether to display progress or not.

        Returns:
            tuple: optimised underlayer thicknesses and SLD, and the
                   corresponding optimisation function value.

        """
        # Check that the underlayers of the sample can be varied.
        #angle_times = experiment.angle_times
        conditions = {} if conditions is None else conditions
        angle_times = conditions.get("angle_times", None)
        nsld_bounds = conditions.get("nsld_bounds", None)
        msld_bounds = conditions.get("msld_bounds", None)
        num_underlayers = sample.num_underlayers
        bounds = [nsld_bounds] * num_underlayers + [msld_bounds] * \
                 num_underlayers

        def _non_decreasing(x):
            """
            Sets the constraint for the contrasts to be in non-decreasing
            order
            """
            return int(np.all(np.diff(x[:0]) >= 0))

        # Set both constrains as equality constraints
        constraints = [
            NonlinearConstraint(_non_decreasing, 1, 1),
        ]
        # Arguments for the optimisation function.
        args = [sample, angle_times]

        # Optimise underlayer thicknesses and SLDs, and return the results.
        res, val = Optimiser.__optimise(
            self._sld_func, bounds, [], args, workers, verbose
        )
        return res[:num_underlayers], res[num_underlayers:], val

    def _sld_func(self,
                          x: list,
                          sample,
                          angle_times: type) -> float:
        """Defines the function for optimising an experiment's underlayers.

        Args:
            x (list): underlayer thicknesses and SLDs to calculate with.
            num_underlayers (int): number of underlayers being optimised.
            angle_times (type): points and times for each angle.
            contrasts (list): contrasts of the experiment, if applicable.

        Returns:
            float: negative of minimum eigenvalue using given conditions.

        """
        # Extract the underlayer thicknesses and SLDs from the given `x` list.
        sld_n, sld_m = x[0], x[1]
        sld_range = [sld_n + sld_m, sld_n - sld_m]

        qs, counts, models = [], [], []
        for sld in sld_range:
            for layer in sample.structure:
                if hasattr(layer, 'underlayer') and layer.underlayer:
                    layer.sld.real.value = sld
            #sample.structure = sample.underlayer_sld(sld_n)
            model, data = simulate(
                sample.structure, angle_times, scale=sample.scale,
                bkg=sample.bkg,
                dq=sample.dq
            )
            qs.append(data[:, 0])
            counts.append(data[:, 3])
            models.append(model)
        xi = sample.get_varying_parameters()
        fisher = Fisher(qs, xi, counts, models)
        # Return negative of the minimum eigenvalue as algorithm is minimising.
        return -fisher.min_eigenval

    def _underlayers_func2(self,
                          x: list,
                          sample,
                          angle_times: type) -> float:
        """Defines the function for optimising an experiment's underlayers.

        Args:
            x (list): underlayer thicknesses and SLDs to calculate with.
-            angle_times (type): points and times for each angle.
            contrasts (list): contrasts of the experiment, if applicable.

        Returns:
            float: negative of minimum eigenvalue using given conditions.

        """
        num_underlayers = sample.num_underlayers
        # Calculate the Fisher information matrix using the conditions.
        underlayer_conditions = [
            (x[i], x[num_underlayers + i]) for i in range(num_underlayers)
        ]

        i = 0
        for layer in sample.structure:
            if hasattr(layer, 'underlayer') and layer.underlayer:
                layer.thick.value = underlayer_conditions[i][0]
                layer.sld.real.value = underlayer_conditions[i][1]
                i += 1

        fisher = Fisher.from_sample(sample, angle_times)

        # Return negative of the minimum eigenvalue as algorithm is minimising.
        return -(fisher.min_eigenval)

    def _angle_times_func(self,
                          x: list,
                          num_angles: int,
                          contrasts: list,
                          points: int,
                          total_time: float) -> float:
        """Defines the function for optimising an experiment's measurement
           angles and associated counting times.

        Args:
            x (list): angles and time splits to calculate the function with.
            num_angles (int): number of angles being optimised.
            contrasts (list): contrasts of the experiment, if applicable.
            points (int): number of data points to use for each angle.
            total_time (float): total time budget for experiment.

        Returns:
            float: negative of minimum eigenvalue using given conditions, `x`.

        """
        # Extract the angles and counting times from given list, `x`.
        angle_times = [
            (x[i], points, total_time * x[num_angles + i])
            for i in range(num_angles)
        ]
        # Calculate the Fisher information matrix.
        fisher = Fisher.from_sample(self.sample, angle_times, contrasts)
        # Return negative of the minimum eigenvalue as algorithm is minimising.
        return -fisher.min_eigenval

    def _contrasts_func(self,
                        x: list,
                        num_contrasts: int,
                        angle_splits: type,
                        total_time: float) -> float:
        """Defines the function for optimising an experiment's contrasts.

        Args:
            x (list): contrasts to calculate the optimisation function with.
            num_contrasts (int): number of contrasts being optimised.
            angle_splits (type): points and time splits for each angle.
            total_time (float): total time budget for experiment.

        Returns:
            float: negative of minimum eigenvalue using given conditions.

        """
        # Define the initial angle times.
        angle_times = []

        # Iterate over each contrast.
        for i in range(num_contrasts):
            # Calculate proportion of the total counting time for each angle.
            angle_times_new = \
                [(angle, points, total_time * x[num_contrasts + i] * split) for
                 angle, points, split in angle_splits]
            angle_times.append(angle_times_new)

        # Calculate the Fisher Information Matrix for the total information.
        fisher = Fisher.from_sample(self.sample, angle_times[0],
                                    x[:num_contrasts])
        #fisher = self.sample.contrast_info(angle_times, x[:num_contrasts])

        # Return negative of the minimum eigenvalue as algorithm is minimising.
        return -fisher.min_eigenval

    def _underlayers_func(self,
                          x: list,
                          num_underlayers: int,
                          angle_times: type,
                          contrasts: list) -> float:
        """Defines the function for optimising an experiment's underlayers.

        Args:
            x (list): underlayer thicknesses and SLDs to calculate with.
            num_underlayers (int): number of underlayers being optimised.
            angle_times (type): points and times for each angle.
            contrasts (list): contrasts of the experiment, if applicable.

        Returns:
            float: negative of minimum eigenvalue using given conditions.

        """
        # Extract the underlayer thicknesses and SLDs from the given `x` list.
        underlayers = [
            (x[i], x[num_underlayers + i]) for i in range(num_underlayers)
        ]

        # Make sure the angle times is set for each contrast
        angle_times = [angle_times for _ in contrasts]

        # Calculate the Fisher information matrix using the conditions.
        fisher = self.sample.underlayer_info(angle_times, contrasts,
                                             underlayers)

        # Return negative of the minimum eigenvalue as algorithm is minimising.
        return -fisher.min_eigenval



    @staticmethod
    def __optimise(func: callable,
                   bounds: list,
                   constraints: list,
                   args: list,
                   workers: int,
                   verbose: bool) -> tuple:
        """Optimises a given `func` using the differential evolution
           global optimisation algorithm.

        Args:
            func (callable): function to optimise.
            bounds (list): permissible values for the conditions to optimise.
            constraints (list): constraints on conditions to optimise.
            args (list): arguments for optimisation function.
            workers (int): number of CPU cores to use when optimising.
            verbose (bool): whether to display progress or not.

        Returns:
            tuple: optimised experimental conditions and function value.

        """
        # Run differential evolution on the given optimisation function.
        res = differential_evolution(func, bounds, constraints=constraints,
                                     args=args, polish=False, tol=0.001,
                                     updating='deferred', workers=workers,
                                     disp=verbose)

        return res.x, res.fun
