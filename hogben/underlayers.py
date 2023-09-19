"""
Functions for visualizing and optimizing the choice of underlayer thickness
and SLD for a given bilayer sample.
"""

import os
import time

import numpy as np

from hogben.optimise import Optimiser
from hogben.visualise import underlayer_choice
from models.bilayers import BilayerDMPC


def _underlayer_results_visualise(save_path):
    """Visualises choice of underlayer thickness and SLD for a bilayer sample.

    Args:
        save_path (str): path to directory to save results to.

    """

    # Choose sample here.
    bilayer = BilayerDMPC()

    # Contrasts to simulate.
    contrasts = [[6.36], [-0.56], [-0.56, 6.36]]

    # Number of points and counting times for each angle to simulate.
    angle_times = [(0.7, 100, 10), (2.3, 100, 40)]

    # Underlayer thicknesses and SLDs to consider.
    thickness_range = np.linspace(5, 500, 50)
    sld_range = np.linspace(1, 9, 100)

    # Visualise underlayer choice, assuming no prior measurement,
    # using different contrasts.
    labels = ['D2O', 'H2O', 'D2O_H2O']
    for c, label in zip(contrasts, labels):
        underlayer_choice(bilayer, thickness_range, sld_range, c, angle_times,
                          save_path, label)

    angle_times = [(0.7, 100, 40)]
    underlayers = [(127.1, 5.39)]  # Optimal DMPC bilayer underlayer.
    # underlayers = [(76.5, 9.00)] # Optimal DPPC/Ra LPS bilayer underlayer.

    # Use nested sampling to validate the improvements.
    bilayer.nested_sampling([-0.56, 6.36], angle_times, save_path,
                            'D2O_H2O_without_underlayer', underlayers=[])

    bilayer.nested_sampling([-0.56, 6.36], angle_times, save_path,
                            'D2O_H2O_with_underlayer', underlayers=underlayers)


def _underlayer_results_optimise(save_path):
    """Optimises choice of underlayer thickness and SLD for a bilayer sample.

    Args:
        save_path (str): path to directory to save results to.

    """

    # Choose sample here.
    bilayer = BilayerDMPC()

    # Number of points and counting times for each angle to simulate.
    angle_times = [(0.7, 100, 10), (2.3, 100, 40)]

    # Contrast to simulate.
    contrasts = [-0.56, 6.36]

    # Intervals containing underlayer thicknesses and SLDs to consider.
    thick_bounds = (0, 500)
    sld_bounds = (1, 9)

    # Create a new .txt file for the results.
    save_path = os.path.join(save_path, bilayer.name)
    file_path = os.path.join(save_path, 'optimised_underlayers.txt')
    with open(file_path, 'w') as file:
        optimiser = Optimiser(bilayer)  # Optimiser for the experiment.

        # Calculate the objective value using no underlayers.
        g = optimiser.sample.underlayer_info(angle_times, contrasts, [])
        val = -np.linalg.eigvalsh(g)[0]
        val = np.format_float_positional(val, precision=4, unique=False,
                                         fractional=False, trim='k')
        file.write('------------ No Underlayer ------------\n')
        file.write('Thicknesses: {}\n'.format([]))
        file.write('SLDs: {}\n'.format([]))
        file.write('Objective value: {}\n\n'.format(val))

        # Calculate the objective value using a gold underlayer.
        underlayer = [(100, 4.7)]
        g = optimiser.sample.underlayer_info(angle_times, contrasts,
                                             underlayer)
        val = -np.linalg.eigvalsh(g)[0]
        val = np.format_float_positional(val, precision=4, unique=False,
                                         fractional=False, trim='k')
        file.write('------------ Au Underlayer ------------\n')
        file.write('Thicknesses: {}\n'.format([underlayer[0][0]]))
        file.write('SLDs: {}\n'.format([underlayer[0][1]]))
        file.write('Objective value: {}\n\n'.format(val))

        # Calculate the objective value using a Permalloy underlayer.
        underlayer = [(100, 8.4)]
        g = optimiser.sample.underlayer_info(angle_times, contrasts,
                                             underlayer)
        val = -np.linalg.eigvalsh(g)[0]
        val = np.format_float_positional(val, precision=4, unique=False,
                                         fractional=False, trim='k')
        file.write('--------- Permalloy Underlayer --------\n')
        file.write('Thicknesses: {}\n'.format([underlayer[0][0]]))
        file.write('SLDs: {}\n'.format([underlayer[0][1]]))
        file.write('Objective value: {}\n\n'.format(val))

        # Optimise the experiment using 1-4 underlayers.
        for i, num_underlayers in enumerate([1, 2, 3]):
            # Display progress.
            print('>>> {0}/{1}'.format(i, 3))

            # Time how long the optimisation takes.
            start = time.time()
            results = optimiser.optimise_underlayers(num_underlayers,
                                                     angle_times, contrasts,
                                                     thick_bounds, sld_bounds,
                                                     verbose=False)
            end = time.time()

            # Round the optimisation function value to 4 significant figures.
            thicknesses, slds, val = results
            val = np.format_float_positional(val, precision=4, unique=False,
                                             fractional=False, trim='k')

            # Save the conditions, objective value and computation time.
            file.write('------------ {} Underlayers ------------\n'.format(
                num_underlayers))
            file.write('Thicknesses: {}\n'.format(
                list(np.round(thicknesses, 1))))

            file.write('SLDs: {}\n'.format(list(np.round(slds, 2))))
            file.write('Objective value: {}\n'.format(val))
            file.write('Computation time: {}\n\n'.format(
                round(end - start, 1)))


if __name__ == '__main__':
    save_path = './results'
    _underlayer_results_visualise(save_path)
    _underlayer_results_optimise(save_path)
