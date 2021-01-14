
from scipy.stats import uniform
import numpy as np


def _round_down(val, size):
    return int(val) - int(val) % size


def get_stochastic_disturbance_years(
        mean_disturbance_time,
        end_age,
        step_size) -> list:
    """
    Returns a list of disturbance years using a uniform probability distribution.

    This is a simplified model that assumes a uniform distribution
    for disturbance events.  For fire, the most common stand replacing
    disturbance in forests, the conditional probability is determined
    by weather (consecutive days without rain), fuel loads and the
    probability of ignition.

    Parameters
    -------------
    mean_disturbance_time : int
    end_age : int
    step_size : int

    Note
    -------------
    CBM-CFS3 returns an error message `returned non-zero exit status 3221225477`
    when multiple disturbances are used and the first occurs at year 0. To avoid
    this error we shift (bias) the distribution by the step_size using the `loc`
    parameter in the uniform distribution.

    """

    # We need a number of samples that will cumulatively be over
    average_disturbances = int(np.ceil(mean_disturbance_time/end_age))
    samples = average_disturbances * 10
    disturbances = uniform.rvs(
        loc=step_size, 
        scale=mean_disturbance_time*2 + step_size,
        size=samples)
    cumulative_disturbances = np.cumsum(disturbances)
    # For the slim chance that the cumulative sum of `disturbances,
    # defined by (`samples`), is less than end_age.
    if cumulative_disturbances[-1] < end_age:
        new_disturbances = uniform.rvs(
            loc=step_size,
            scale=mean_disturbance_time + step_size,
            size=samples)
        disturbances = np.append(disturbances, new_disturbances)
        cumulative_disturbances = np.cumsum(disturbances)
    cumulative_disturbances = [
        _round_down(year, step_size) for year in cumulative_disturbances]
    cumulative_disturbances = np.array(cumulative_disturbances)
    mask = (cumulative_disturbances <= end_age)
    return list(cumulative_disturbances[mask])


def get_disturbance_index_for_each_simulation(
        simulations,
        mean_disturbance_time,
        end_age,
        step_size):
    """
    Parameters
    -------------
    simulations : int
        Number of events (e.g. stands) to consider.
    """

    # A list to lump the disturbance events from each simulation together.
    # We will use this to inspect the properties of results.
    all_disturbances = []
    # This list will hold the random disturbance years for each simulation 
    disturbances_by_simulation = []

    for i in range(simulations):
        disturbance_years = get_stochastic_disturbance_years(mean_disturbance_time, end_age, step_size)
        all_disturbances += disturbance_years
        disturbances_by_simulation.append(disturbance_years)

    return disturbances_by_simulation, all_disturbances
