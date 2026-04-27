import numpy as np
from pyrfu import pyrf
from py_space_zc import background_B, plot, ts_vec_xyz

def SVD_B(Bwave, window_length=20.0, overlap=10.0, freq_range=[0.1, 16.0],
          m_width_coeff = 1):
    """
    Perform Singular Value Decomposition (SVD)-based polarization analysis of
    magnetic field fluctuations using the `pyrfu.ebsp` routine.

    This function analyzes the wave magnetic field `Bwave` using the
    Singular Value Decomposition method, computing polarization parameters
    such as perpendicular/parallel PSD, ellipticity, planarity, and wavevector angle.
    The results are derived from the eigenvalue decomposition of the spectral matrix.

    Parameters
    ----------
    Bwave : xarray.DataArray
        Wave magnetic field time series with shape (time, 3) in units of nT.
        Should be high-resolution magnetic field fluctuations, e.g., B_high['Bmso'].
        - Bwave.time.data : np.ndarray of np.datetime64
        - Bwave.data      : np.ndarray of shape (N, 3), magnetic field in MSO or similar.
    window_length : float, optional (default=20.0)
        Length of each time window (in seconds) used for background field averaging.
    overlap : float, optional (default=10.0)
        Overlap between adjacent windows (in seconds).
    freq_range : list of float, optional (default=[0.1, 16.0])
        Frequency range (in Hz) over which to perform the polarization analysis.
        Used to limit the output of `ebsp`.

    Returns
    -------
    res : dict
        Dictionary containing polarization and spectral results:
        - 'Bperp'       : Perpendicular magnetic power spectral density (PSD), scalar.
        - 'Bpara'       : Parallel magnetic power spectral density (PSD), scalar.
        - 'ellipticity' : Ellipticity (dimensionless, from -1 to 1).
        - 'thetak'      : Wavevector angle θ_k with respect to background B [deg].
        - 'planarity'   : Planarity of wave polarization (0 to 1).

    Notes
    -----
    - The `ebsp` function is part of the `pyrfu` package and implements the
      SVD-based method from Santolík et al. (2003), widely used in wave analysis.
    - The background magnetic field is computed using sliding window averaging
      via `py_space_zc.background_B`.
    - The electric field is not used; a zero `e_xyz` placeholder is passed for compatibility.
    - The results are returned as time × frequency xarray objects (inherited from `ebsp`).

    References
    ----------
    Santolík, O., Parrot, M., & Lefeuvre, F. (2003).
    Singular value decomposition methods for wave propagation analysis.
    Radio Science, 38(1), 1010. https://doi.org/10.1029/2000RS002523

    Example
    -------
    >>> from py_space_zc import maven
    >>> tint = ["2022-01-24T08:06:30", "2022-01-24T08:09:00"]
    >>> B = maven.get_data(tint, 'B_high')['Bmso']
    >>> result = SVD_B(B, window_length=10.0, overlap=5.0, freq_range=[0.5, 8.0])
    >>> Bperp = result['Bperp']
    >>> Bpara = result['Bpara']
    """

    # Placeholder electric field (zeros) — not used in this analysis
    e_xyz = ts_vec_xyz(Bwave.time.data, np.zeros_like(Bwave.data))

    # Radial vector placeholder (all ones), used for FAC projection
    r_xyz = ts_vec_xyz(Bwave.time.data, np.ones_like(Bwave.data))

    # Compute background field using sliding time-window average
    Bbgd = background_B(Bwave, window_length=window_length, overlap=overlap)

    # Call pyrfu.ebsp for SVD-based polarization analysis
    polarization_options = dict(freq_int=freq_range,
                                polarization=True,
                                fac=True,
                                m_width_coeff = m_width_coeff)
    polarization = pyrf.ebsp(e_xyz, Bwave, Bbgd, Bbgd, r_xyz, **polarization_options)

    # Extract results from output dictionary
    Bperp_psd = polarization["bb_xxyyzzss"][..., 0] + polarization["bb_xxyyzzss"][..., 1]  # perp = xx + yy
    Bpara_psd = polarization["bb_xxyyzzss"][..., 2]  # parallel = zz
    ellipticity = polarization["ellipticity"]
    thetak = polarization["k_tp"][..., 0]            # angle between k and B0
    planarity = polarization["planarity"]
    thetak_deg = thetak.copy(data=thetak.data * 180.0 / np.pi)
    # Package into result dictionary
    res = {
        'Bperp': Bperp_psd,
        'Bpara': Bpara_psd,
        'ellipticity': ellipticity,
        'theta': thetak_deg,
        'planarity': planarity
    }

    return res
