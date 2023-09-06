"""
Functions for visualizing and optimizing angle choice for a given sample.
"""
import os
import time

import numpy as np

from hogben.optimise import Optimiser
from models.samples import simple_sample
from hogben.visualise import angle_choice, angle_choice_with_time


def _angle_results_visualise(save_path):
    """Visualises the initial angle choice for a sample and how the choice of
       next angle changes as the counting time of the initial angle increases.


    Args:
        save_path (str): path to directory to save results to.

    """
    # Choose sample here.
    sample = simple_sample()

    # Choose contrasts here (only bilayers should use this).
    contrasts = []
    # contrasts = [-0.56, 6.36]

    # Number of points and counting time for the initial angle choice.
    points = 100
    time = 100

    # Get the best angle to initially measure.
    angle_range = np.linspace(0.2, 4, 500)
    initial_angle = angle_choice(sample, [], angle_range, points, time,
                                 save_path, 'initial', contrasts)
    # Plot how the choice of angle changes with initial angle counting time.
    angle_range = np.linspace(0.2, 4, 50)
    time_range = np.linspace(0, time * 8, 50)
    angle_choice_with_time(sample, initial_angle, angle_range, time_range,
                           points, time, save_path, contrasts)


def _angle_results_optimise(save_path):
    """Optimises the choice measurement angles and counting times for a sample.

    Args:
        save_path (str): path to directory to save results to.

    """
    # Choose sample here.
    sample = simple_sample()

    # Choose contrasts here (only bilayers should use this).
    contrasts = []
    # contrasts = [-0.56, 6.36]

    # Total time budget.
    total_time = 1000  # A large time improves DE convergence.

    # Interval containing angles to consider.
    angle_bounds = (0.2, 4.0)

    # Create a new .txt file for the results.
    save_path = os.path.join(save_path, sample.name)
    with open(os.path.join(save_path, 'optimised_angles.txt'), 'w') as file:
        optimiser = Optimiser(sample)  # Optimiser for the experiment.

        # Optimise the experiment using 1-4 angles.
        for i, num_angles in enumerate([1, 2, 3, 4]):
            # Display progress.
            print('>>> {0}/{1}'.format(i, 4))

            # Time how long the optimisation takes.
            start = time.time()
            results = optimiser.optimise_angle_times(num_angles, contrasts,
                                                     total_time, angle_bounds,
                                                     verbose=False)
            end = time.time()

            # Convert to percentages.
            angles, splits, val = results
            splits = np.array(splits) * 100

            # Round the optimisation function value to 4 significant figures.
            val = np.format_float_positional(
                val, precision=4, unique=False, fractional=False, trim='k'
            )

            # Save the conditions, objective value and computation time.
            file.write(
                '----------- {} Angles -----------\n'.format(num_angles)
            )
            file.write('Angles: {}\n'.format(list(np.round(angles, 2))))
            file.write('Splits (%): {}\n'.format(list(np.round(splits, 1))))
            file.write('Objective value: {}\n'.format(val))
            file.write(
                'Computation time: {}\n\n'.format(round(end - start, 1))
            )


if __name__ == '__main__':
    save_path = './results'
    _angle_results_visualise(save_path)
    _angle_results_optimise(save_path)
