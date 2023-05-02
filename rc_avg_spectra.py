def rotary_coefficient_profile_avg_spectra(initial_lag_m, window_length_m, u, v, N,\
                               vel_depth,ctd_depth, v_spacing = 5,\
                               cutoffs=[(2*np.pi)/(60),(2*np.pi)/(180)],
                              N_normalize=True,transfer=True,density=False,method='coeff last'):
    '''Produces a rotary coefficient with depth profile for velocity and shear using half-overlapping windows
    
    Parameters
    ----------
    initial_lag_m : float or int
    window_length_m : float or int
    u : xarray.core.dataarray.DataArray or np.ndarray
        1D zonal velocity depth array  
    v : xarray.core.dataarray.DataArray or np.ndarray 
        1D meridional velocity depth array 
    N : xarray.core.dataarray.DataArray or np.ndarray 
        1D buoyancy frequency depth array 
    vel_depth : xarray.core.dataarray.DataArray or np.ndarray 
        1D LADCP depth array 
    ctd_depth : xarray.core.dataarray.DataArray or np.ndarray 
        1D CTD depth array 
    v_spacing : float or int, optional 
        spacing in [m] of the LADCP measurements. Default is set to 5m as in the original FSDB_SR1b_V0.1_20221013
        dataset. This is used to index properly. 
    cutoffs : list 
        Spectral wavenumber cutoffs for integration of power. Must contain two numbers, unordered is fine: 
        low wavenumber cutoff and high wavenumber cutoff. 
    N_normalize : bool, optional 
        Determines if the Fourier coefficients are computed from the u + iv series normalized 
        by the depth mean of the given buoyancy frequency array. Default value is True. 
    transfer : bool, optional 
        Determines if the Fourier coefficients are spectrally corrected (transferred) according
        to Thurnherr, 2012. Default value is True
    density : bool, optional
        Determines if the resulting rotary coefficients are calculated from power density (PSD). 
        Default is False. 
    method : str, optional
    	'coeff last' or 'coeff first'. The computational method used for calculating the rotary coefficients. 
    	Default is 'coeff last', the typical implementation. See `rotary_coefficient` function for more. 
    
    Returns
    -------
    (R_c_vel_prof, R_c_sh_prof, midp) : tuple
        R_c_vel_prof : np.ndarray  
            Array of velocity rotary coefficients computed in each window. Size is the number of windows. 
        R_c_sh_prof : np.ndarray
            Array of shear rotary coefficients computed in each window. Size is the number of windows. 
        midp : np.ndarray
            Array of the depth midpoints of the successive windows. These are the depths to which each
            rotary coefficient is attributed in a rotary coefficient depth profile. 
    
    Notes
    -----
    This implementation is hard coded for half-overlapping windows. Will catch errors computing rotary 
    coefficients and store them as np.nan for the profile. 
    
    [1]_ Thurnherr, A. M. (2012). The finescale response of lowered ADCP velocity measure-ments 
    processed with different methods. Journal of Atmospheric and Oceanic Technology, 29(4), 
    597–600. https://doi.org/10.1175/JTECH-D-11-00158.1
    
    '''
    
    max_n = int((vel_depth[-1] - window_length_m)/(window_length_m/2))
    lags  = [initial_lag_m + n*(window_length_m/2) for n in range(max_n)]
    midp  = np.array(lags[1:] + [lags[-1] + window_length_m / 2])

    # for cruise in cruises
    # make shear for all cruises 
    
    u_sh,v_sh = shear(u,v)
    
    R_c_vel_prof = np.empty(max_n)
    R_c_sh_prof  = np.empty(max_n)
    
    for n in range(max_n):
        try:
            # make data for each cruise
            vel_data = window_data(lags[n], window_length_m, u, v, N, vel_depth, ctd_depth, v_spacing)
            shear_data = window_data(lags[n], window_length_m, u_sh, v_sh, N, vel_depth, ctd_depth, v_spacing)

        # need to do this for each, this could be a problem with the spectra averaging 
        # will have to throw out some cruises? 

        # catch CTD/LADCP overlap problem described in `window_data`
        except ValueError:
            R_c_vel = np.nan
            R_c_shear = np.nan 
            R_c_vel_prof[n] = R_c_vel
            R_c_sh_prof[n]  = R_c_shear

            continue

        try:
            ############################################## need to put loop over whole procedure 
            # HERE: average spectra across time 
            vel_spectra = rotary_spectra(vel_data[0],vel_data[1],vel_data[2],N_normalize=N_normalize,transfer=transfer,density=density,shear=False)
            shear_spectra = rotary_spectra(shear_data[0],shear_data[1],shear_data[2],N_normalize=N_normalize,transfer=transfer,density=density,shear=True)
            
            ##############################################

            # compute for average spectra

            R_c_vel   = rotary_coefficient(vel_spectra[0],vel_spectra[1],vel_spectra[2],cutoffs=cutoffs,method=method)
            R_c_shear = rotary_coefficient(shear_spectra[0],shear_spectra[1],shear_spectra[2],cutoffs=cutoffs,method=method)
        
        except (AssertionError, ValueError, TypeError):
            R_c_vel = np.nan
            R_c_shear = np.nan
        
        R_c_vel_prof[n] = R_c_vel
        R_c_sh_prof[n]  = R_c_shear
        
    return (R_c_vel_prof, R_c_sh_prof, midp)