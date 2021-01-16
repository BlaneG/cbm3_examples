"""
Some examples from the literature of yield curves (m3/ha)
"""
import numpy as np



def site_class_yield_curve(b1, b2, b3, b4, age) -> float or np.ndarray:
    """
    Returns a Chapman-Richards's growth and yield curve paramaterized by site classification BC.
    
    Parameters
    ------------
    b1, b2, b3 : float
        Empirical parameters for chapman richards function.
    b4 : int or float
        Initial age used to evaluate growth and yield curve.
    age : int, float or np.ndarray
        Age(s) at which the growth curve is evaluated.
        
    Notes
    ----------
    From Appendix 2 in BCMOF:  "The Site Class System is best suited to predict yields of unmanaged stands where forest cover
        types are classified according to the old MOF inventory standards, superior yield prediction systems
        are not available, or generalized, average yield values are adequate.This system is appropriate for 
        estimating current volumes for use in broad, forest level planning. The model structure is not suitable for
        yield estimation in uneven-age stands. Within the MOF, this system receives only occasional use to
        supply “first approximation” volume estimates or to provide a benchmark against which the predictions
        of newer systems can be compared."
    
    References
    ----------
    BCMOF, 1991. Growth & Yield Prediction Systems.
        https://www.for.gov.bc.ca/hfd/pubs/docs/srs/Srs07.pdf

    Returns
    ------------
    chapman_richards : float or np.ndarray
    
    """
    
    return b1 * (1-np.exp(b2 * (age - b4))) ** b3