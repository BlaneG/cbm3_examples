import warnings

from scipy.stats import uniform
import numpy as np


def get_stochastic_disturbance_years(
        mean_disturbance_time,
        end_age,
        step_size,
        simulations,
        random_state=None,
        disturbance_delay=0
        ) -> list:
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
    simulations : int
    disturbance_delay : int
        Delay in years until disturbances begin.
    seed : int
        

    Note
    -------------
    CBM-CFS3 returns an error message `returned non-zero exit status 3221225477`
    when multiple disturbances are used and the first occurs at year 0. To avoid
    this error we shift (bias) the distribution by the step_size using the `loc`
    parameter in the uniform distribution.

    """

    # We need a number of samples that cumulatively sum to end_age.
    average_disturbances = np.ceil(mean_disturbance_time/end_age).astype(int)
    samples = average_disturbances * 20

    # This allows us to evaluate disturbances on mature stands
    loc = step_size + disturbance_delay

    disturbances = uniform.rvs(
        loc=loc, 
        scale=mean_disturbance_time*2 + step_size,
        size=(simulations, samples),
        random_state=random_state)
    cumulative_disturbances = np.cumsum(disturbances, axis=1)
    # For the slim chance that the cumulative sum of `disturbances,
    # defined by (`samples`), is less than end_age.
    # print(cumulative_disturbances[:, -1])
    if np.any(cumulative_disturbances[:, -1] < end_age):
        warnings.warn(f"The cumulative disturbance time of at least 1 simulation is less than {end_age}.")

    cumulative_disturbances = _round_down(cumulative_disturbances, step_size)

    return _get_disturbances_before_end_year(
        cumulative_disturbances, end_age)


def _round_down(val, size):
    return val.astype(int) - val.astype(int) % size


def _get_disturbances_before_end_year(cumulative_disturbances, end_age):
    masks = cumulative_disturbances < end_age
    return [disturbances[mask].tolist()
     for disturbances, mask
     in zip(cumulative_disturbances, masks)]
