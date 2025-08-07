import numpy as np

def sind(x):
    """Sine of angle in degrees"""
    return np.sin(np.deg2rad(x))

def cosd(x):
    """Cosine of angle in degrees"""
    return np.cos(np.deg2rad(x))

def tand(x):
    """Tangent of angle in degrees"""
    return np.tan(np.deg2rad(x))

def cotd(x):
    """Cotangent of angle in degrees"""
    return 1 / np.tan(np.deg2rad(x))

def asind(x):
    """Arcsine result in degrees"""
    return np.rad2deg(np.arcsin(x))

def acosd(x):
    """Arccosine result in degrees"""
    return np.rad2deg(np.arccos(x))

def atand(x):
    """Arctangent result in degrees"""
    return np.rad2deg(np.arctan(x))

def atan2d(y, x):
    """Two-argument arctangent result in degrees"""
    return np.rad2deg(np.arctan2(y, x))
