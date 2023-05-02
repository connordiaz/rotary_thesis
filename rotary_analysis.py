####### Dependencies ##########

# [ numpy, matplotlib.pyplot, xarray ]

# Imports 

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

## MUST SET FOR PLOTTING FUNCTIONS TO WORK ###
plt.rcParams['text.usetex'] = True
from matplotlib.ticker import StrMethodFormatter 

#############################################################################################################
### Processing Function ### 

def process_data(station_number, profile_number, data_dir='',workaround=False,ctd_profile_number=None):  
    
    '''Processes raw FSDB_SR1b_V0.1_20221013 database files
    
    process_data(station_number, profile_number, data_dir='')
    
    Processes the raw LADCP and CTD files from the FSDB_SR1b_V0.1_20221013 database of .nc netcdf files
    created by Andreas Thurnherr from the SR1b repeat hydrography of the Drake Passage.  
    
    Parameters
    ----------
    
    station_number : str or int or float 
        Number between 1 and 27. `station_number` refers to the repeat hydrography station as in Thurnherr et al. and follows
        from the naming convention in the paper, for example 'R09' for the 9th station. If the station number is 
        less than 10, the leading 0 is automatically added. 
    profile_number : int
        `profile number` is chosen to select the specific LADCP and CTD profile at the given station. This will
        determine from which cruise the data is selected. The ordering of the cruises is inherent to the database
        and is chronological.
    data_dir : str, optional
        Directory string for the data files. Default value is the empty string, '', which assumes the data files
        are in the same directory as the script. 
        
    Returns
    -------
    
    (u,v,N,vel_depth,ctd_depth,cruise_name) : tuple 
        data variables as follows: 
        u : xarray.core.dataarray.DataArray of numpy.float64 
            zonal velocity (m/s) 1D depth array 
        v : xarray.core.dataarray.DataArray of numpy.float64
            meridional velocity (m/s) 1D depth array 
        N : xarray.core.dataarray.DataArray of numpy.float64
            buoyancy frequency (rad/s) 1D depth array 
        vel_depth : xarray.core.dataarray.DataArray of numpy.float64
            depth (m) of velocity measurements 
        ctd_depth : xarray.core.dataarray.DataArray of numpy.float64
            depth (m) of CTD measurements 
        cruise_name : str
            name of the specific cruise for the above measurements, as in Thurnherr et al. 
        workaround : bool, optional
        	Indicates whether the indexing correction for the first version of the database is implemented. 
        	This corrects for missing CTD cruise data from JR170001. Default is False. 
        ctd_profile_number : int, optional
            **ONLY SUPPLY IF** `workaround`=True. Separate profile number to select correct CTD data due to error 
            caused by missing data in original database. Default is None and is not used if `workaround`=False. 
    
    Raises
    ------
    
    ValueError 
        if there is any error thrown by xarray opening the data files 
    ValueError
        if the station number is not passed in as a readable format 
        
    Notes
    -----
    
    process_data(station_number, profile_number, data_dir='') reads in the netcdf files for CTD and LADCP
    data at the given station, selects the given profile, replaces missing values with np.nan, and finally 
    masks u, v, N according to where there are depth measurements for those instruments (they are placed 
    on a uniform grid and filled with missing values, originally). The resulting arrays are of values where
    each respective instrument measured a depth. The arrays may still have nan values where the specific 
    instrument could not register a value. This is often the first two values for the LADCP. 
    
    Note that u, v will have the same dimensions, but that N will be on a different grid. 
     
    This function assumes the files are named according to 'R02_CTD.nc' and 'R02_LADCP.nc'. 
        
    [1]_ Thurnherr, A. M., Firing, Y., Waterman, S., & Chereskin, T. (n.d.). Summertime
    internal-wave properties along the SR1b section across drake passage. Journal
    of Physical Oceanography, 16.
    
    '''
    
    if type(station_number) == str:
        station = station_number
    elif (type(station_number) == int) or (type(cruise_number) == float):
        if station_number < 10:
            station = '0'+str(station_number)
        else:
            station = str(station_number)
    else:
        raise ValueError("Cruise number must be either a string like '06' or '13' \n or a float between 1 and 27")
    
    ladcp_station_file = 'R'+station+'_LADCP.nc'
    ctd_station_file   = 'R'+station+'_CTD.nc'
    
    # open file data
    try:
        ladcp = xr.open_dataset(data_dir+ladcp_station_file)
        ctd   = xr.open_dataset(data_dir+ctd_station_file)
    except:
        raise ValueError('Data not found')

    if workaround:
        ctd_idx = ctd_profile_number 
    else: 
        ctd_idx = profile_number

    # select data
    u = ladcp.zonal_velocity.sel(profile_number=profile_number)
    v = ladcp.meridional_velocity.sel(profile_number=profile_number)
    N = ctd.buoyancy_frequency.sel(profile_number=ctd_idx)
    vel_depth = ladcp.depth.sel(profile_number=profile_number)
    ctd_depth = ctd.depth.sel(profile_number=ctd_idx)
    
    cruise_name = str((ladcp.sel(profile_number=profile_number).cruise_id).to_numpy())[2:-1]
    
    # replace missing values with np.nan

    missing_val = 9.9E36

    u[u>missing_val] = np.nan
    v[v>missing_val] = np.nan
    N[N>missing_val] = np.nan

    vel_depth[vel_depth>missing_val] = np.nan
    ctd_depth[ctd_depth>missing_val] = np.nan
    
    # mask data to non depth nan values 
    # (i.e. where data wasn't recorded, not where non-pressure instrument errors)

    vel_mask = ~np.isnan(vel_depth)
    ctd_mask = ~np.isnan(ctd_depth)

    vel_depth = vel_depth[vel_mask]
    ctd_depth = ctd_depth[ctd_mask]

    u = u[vel_mask]
    v = v[vel_mask]
    N = N[ctd_mask]
    
    return (u,v,N,vel_depth,ctd_depth,cruise_name)

#############################################################################################################
### Velocity/Shear Plotting ### 

def plot_profiles(u,v,N,vel_depth,ctd_depth,shear=False,whole_shear_profile=False):
    
    '''plots u, v, and N profiles
    
    Parameters
    ----------
    u : xarray.core.dataarray.DataArray or np.ndarray 
        zonal velocity or shear 1D array 
    v : xarray.core.dataarray.DataArray or np.ndarray
        meridional velocity or shear 1D array 
    N : xarray.core.dataarray.DataArray or np.ndarray
        buoyancy frequency 1D array 
    vel_depth : xarray.core.dataarray.DataArray or np.ndarray
        LADCP depth 1D array 
    ctd_depth : xarray.core.dataarray.DataArray or np.ndarray
        CTD depth 1D array 
    shear : bool, optional
        default value is False. Determines whether to plot velocity or vertical shear of velocity 
    whole_shear_profile : bool, optional 
        if the shear data is for the entire vertical profile, the last vel_depth, ctd_depth, and N values 
        are omitted, as the shear, by definition, only exists for N - 1 depths. False means that you 
        are plotting a windowed section of the values. 
    
    Returns
    -------
    None 
    
    Notes
    -----
    Assumes data are in m/s (velocity) 1/s (shear) m (depth) and rad/s (buoyancy frequency). 
    
    Sets symmetric velocity/shear scales based on the largest magnitude measurement. 
    
    Plots are visually scaled to have an interpretable axis aspect ratio, given the sizes of the data. This
    is not customizable here, but is sensitive to the specific data. This may cause problems with plotting in 
    case of data that has a disagreeable size to the aspect ratio parameters, but has not been observed. 
    
    '''
    
    if type(u) != np.ndarray:
        u = u.to_numpy()
        v = v.to_numpy()
        N = N.to_numpy()
        vel_depth = vel_depth.to_numpy()
        ctd_depth = ctd_depth.to_numpy()
      
    # if whole shear profile, chop off last depth 
    
    if whole_shear_profile:
        vel_depth = vel_depth[:-1]
        ctd_depth = ctd_depth[:-1]
        N         = N[:-1]

    fig, axes = plt.subplots(2, 2)

    axes_1 = axes[0]
    axes_2 = axes[1]
    
    # set aspect
    
    uv_range        = np.max( [ np.nanmax(u) - np.nanmin(u), np.nanmax(v) - np.nanmin(v) ] )
    N_range         = np.nanmax(N) - np.nanmin(N)
    uv_depth_range  = np.nanmax(vel_depth) - np.nanmin(vel_depth)
    ctd_depth_range = np.nanmax(ctd_depth) - np.nanmin(ctd_depth)
    
    if shear:
        uv_aspect = (uv_range / uv_depth_range) 
        N_aspect  = 1 * (N_range / ctd_depth_range)
    else:
        uv_aspect = 3.5 * (uv_range / uv_depth_range) 
        N_aspect  = 1 * (N_range / ctd_depth_range)
    
    # x-axis lims
    
    xmin1 = np.min([np.nanmin(u) , np.nanmin(v)])
    xmax1 = np.max([np.nanmax(u) , np.nanmax(v)]) 
    
    if (xmin1 < 0) and (xmax1 > 0):
        xsc = np.max([abs(xmin1),xmax1])
    elif (xmin1 > 0) and (xmax1 > 0):
        xsc = xmax1
    elif (xmin1 < 0) and (xmax1 < 0):
        xsc = abs(xmin1)
    elif (xmin1 == 0) and (xmax1 != 0):
        xsc = xmax1
    elif (xmin1 != 0) and (xmax1 == 0):
        xsc = abs(xmin1)
    elif (xmin1 == 0) and (xmax1 == 0):
        raise ValueError('Your data is 0 everywhere')
    else:
        raise ValueError('Your velocity data is problematic')
        
    xscale = [-xsc,xsc]
    
    # y-axis lims
    
    vel_yscale = [vel_depth[0], vel_depth[-1]]
    ctd_yscale = [ctd_depth[0], ctd_depth[-1]]
    
    # plot
        
    axes_1[0].set_aspect(uv_aspect)
    axes_1[0].plot(u, vel_depth, linewidth=0.5, c='k')
    #axes_1[0].scatter(u, vel_depth, s = 1, c = 'b')
    axes_1[0].set_ylim(vel_yscale)
    axes_1[0].invert_yaxis()
    axes_1[0].set_xlim(xscale)
    axes_1[0].axvline(0)
    axes_1[0].axhline(150, c='r')
    if shear:
        axes_1[0].set_title('du/dz')
        axes_1[0].set_xlabel('1/s')
    else:
        axes_1[0].set_title('u')
        axes_1[0].set_xlabel('m/s')
    axes_1[0].set_ylabel('depth (m)')

    axes_1[1].set_aspect(uv_aspect)
    axes_1[1].plot(v, vel_depth, linewidth=0.5, c='k')
    #axes_1[1].scatter(v, vel_depth, s=1,c='m')
    axes_1[1].set_ylim(vel_yscale)
    axes_1[1].invert_yaxis()
    axes_1[1].set_xlim(xscale)
    axes_1[1].axvline(0)
    axes_1[1].axhline(150, c='r')
    if shear:
        axes_1[1].set_title('dv/dz')
        axes_1[1].set_xlabel('1/s')
    else:
        axes_1[1].set_title('v')
        axes_1[1].set_xlabel('m/s')
    axes_1[1].set_ylabel('depth (m)')

    axes_2[0].set_aspect(N_aspect)
    axes_2[0].plot(N, ctd_depth, linewidth=0.5, c='k')
    axes_2[0].scatter(N, ctd_depth, s=1, c='y')
    axes_2[0].axhline(150, c='r')
    axes_2[0].set_ylim(ctd_yscale)
    axes_2[0].invert_yaxis()
    axes_2[0].set_title('N')
    axes_2[0].set_ylabel('depth (m)')
    axes_2[0].set_xlabel('rad/s')

    axes_2[1].set_aspect(uv_aspect)
    axes_2[1].set_xlim(xscale)
    axes_2[1].set_ylim(vel_yscale)
    axes_2[1].scatter(u, vel_depth, s=1, c='b', label='u')
    axes_2[1].scatter(v, vel_depth, s=1, c='m', label='v')
    axes_2[1].invert_yaxis()
    axes_2[1].axvline(0)
    axes_2[1].axhline(150, c='r')
    if shear:
        axes_2[1].set_title('du/dz, dv/dz')
        axes_2[1].set_xlabel('1/s')
    else:
        axes_2[1].set_title('u, v')
        axes_2[1].set_xlabel('m/s')
    axes_2[1].set_ylabel('depth (m)')
    axes_2[1].legend()


    plt.tight_layout()
    
    return None

#############################################################################################################
### Windowing Function ###

def window_data(lag_m, window_length_m, u, v, N, vel_depth, ctd_depth, v_spacing = 5, verbose=False, uv_mean_removal=True):
    
    '''selects and returns a vertical subset of u, v, and N data 
    
    Parameters
    ----------
    
    lag_m : int or float 
        the distance (in [m]) from the surface that indicates the first measurement desired in the window
    window_length_m : int or float 
        the distance (in [m]) that indicates the length of the window 
    u : xarray.core.dataarray.DataArray or np.ndarray 
        zonal velocity or shear 1D array 
    v : xarray.core.dataarray.DataArray or np.ndarray 
        meridional velocity or shear 1D array
    N : xarray.core.dataarray.DataArray or np.ndarray 
        buoyancy frequency 1D array
    vel_depth : xarray.core.dataarray.DataArray or np.ndarray 
        LADCP 1D depth array
    ctd_depth : xarray.core.dataarray.DataArray or np.ndarray 
        CTD 1D depth array
    v_spacing : int or float, optional 
        spacing in [m] of the LADCP measurements. Default is set to 5m as in the original FSDB_SR1b_V0.1_20221013
        dataset. This is used to index properly. 
    verbose : bool, optional
    	Determines if various settings are printed. Default is False. 
    uv_mean_removal : bool, optional
        Determines if the window mean is removed from the u and v series. Default is True. 
    
    Returns
    -------
    (u_w, v_w, N_w, vel_depth_w, ctd_depth_w) : tuple 
        u, v, N, vel_depth, and ctd_depth subselected for the window given by lag_m and window_m 
        
    Raises
    ------
    IndexError if the given window is not possible. 
    AssertionError if any input data is size 0. 
    ValueError if CTD measurements do not cover 90 percent of the windowed LADCP grid. 

    Notes
    -----
    Prints the size of the windowed data. 
    
    Because the arrays need to be indexed with integers, the first and last indices are truncated
    by int(lag_m/v_spacing) and int(window_length_m/v_spacing). This may give unexpected indices if floats
    closer to the integer ceiling (rounding up) are given, as int() truncates, giving the integer
    floor of the number. 
    
    The convention used here leads the indices by 1. I.e. if the lag is 1000m from the top and the window
    size is 320m, then the data is windowed from 1005 (the next indexed depth in a 5m spaced grid) to 1320m. 
    This method insures that the last value in the window is included and the first omitted. The rational here
    is that this produces size 64 windows if we use 320m windows. 

    Occasionally, CTD measurements do not extend past the velocity measurements. In this case, the returned buoyancy
    frequency values (N) simple extend to their last measurement so long as it is less than 10 percent off of the velocity
    grid i.e. there is a 90 percent overlap. If this is not the case, a ValueError is thrown. 
    
    '''
    
    # convert to numpy arrays 
    
    if type(u) != np.ndarray:
        u = u.to_numpy()
        v = v.to_numpy()
        N = N.to_numpy()
        vel_depth = vel_depth.to_numpy()
        ctd_depth = ctd_depth.to_numpy()

    assert len(u) > 0
    assert len(v) > 0 
    assert len(N) > 0
    assert len(vel_depth) > 0
    assert len(ctd_depth) > 0
    
    # convert meters to indices
    lag_n           = int(lag_m/v_spacing)
    window_length_n = int(window_length_m/v_spacing)
     
    # select data in window (lead first index by one)
    u_w         = u[lag_n + 1:lag_n + window_length_n + 1]
    v_w         = v[lag_n + 1:lag_n + window_length_n + 1]
    vel_depth_w = vel_depth[lag_n + 1:lag_n + window_length_n + 1]
    
    # find equivalent N^2 window (it is on a different grid)
    # if window bounds don't match exactly, pick first value outside velocity grid min \
    # and outside velocity grid max
            
        # find first ctd depth

    idx1  = np.where(ctd_depth < float(vel_depth_w[0]))[0][-1]

        # find second ctd depth for window
    try:
        idx2  = np.where(ctd_depth > float(vel_depth_w[-1]))[0][0] # no + 1 because we want value
    except IndexError:
        # IndexError is thrown when ctd profile is shallower than last windowed velocity depth 
        # in this case, we will simply use the last ctd depth if it is within 10% of the window length. 
        # I.e we assume that we can normalize the velocities by buoyancy frequency in the future fft if 
        # we assume the CTD had 90% spatial coverage of the LADCP measurements. 
        if vel_depth[-1] - ctd_depth[-1] < (window_length_m * 0.1):
            idx2 = np.where(ctd_depth == ctd_depth[-1])[0][0]
        else:
            raise ValueError('CTD measurments do not cover 90 percent of the windowed velocity grid')
        
        # mask buoyancy frequency
    N_w = N[idx1:idx2+1]
    
        # get ctd depth masked
    
    ctd_depth_w = ctd_depth[idx1:idx2+1]


    if verbose:
    	print('the size of your velocity arrays is '+str(u_w.size))

    if uv_mean_removal:
        u_w = u_w - u_w.mean()
        v_w = v_w - v_w.mean()
    
    return (u_w, v_w, N_w, vel_depth_w, ctd_depth_w)

#############################################################################################################
### Velocity FFT ### 

def uv_fft(u,v,N,N_normalize=True,transfer=True, verbose=False): 
    
    '''Computes the Fourier Transform of u + iv or du/dz + idv/dz depth series 
    
    Parameters
    ----------
    u : xarray.core.dataarray.DataArray or np.ndarray 
        1D zonal velocity depth array 
    v : xarray.core.dataarray.DataArray or np.ndarray 
        1D meridional velocity depth array 
    N : xarray.core.dataarray.DataArray or np.ndarray 
        1D buoyancy frequency depth array 
    N_normalize : bool, optional 
        Determines if the Fourier coefficients are computed from the u + iv series normalized 
        by the depth mean of the given buoyancy frequency array. Default value is True. 
    transfer : bool, optional 
        Determines if the Fourier coefficients are spectrally corrected (transferred) according
        to Thurnherr, 2012. Default value is True
    verbose : bool, optional
    	Determines if various settings are printed. Default is False. 
    
    Returns
    -------
    (fft, wavenumbers) : tuple
        fft : np.ndarray 
            Array of Fourier coefficients for the transform of u + iv 
        wavenumbers : np.ndarray 
            Array of wavenumbers in [rad/m], ordered as described in np.fft.fftfreq
            
    Raises
    ------
    TypeError if a NaN is found in the velocity series. 
    ValueError if the size of u and v is not a power of 2. 
    AssertionError if the mean buoyancy frequency is < 6.5*(10**-4) 
    
    See Also
    --------
    np.fft.fft for a full description of the specific Discrete Fourier Transform implementation. 
    https://numpy.org/doc/stable/reference/generated/numpy.fft.fft.html#numpy.fft.fft
    
    Notes
    -----
    This routine requires the input arrays to be a power of 2 in order to avoid zero padding and the 
    associated edge effects. 
    
    [1]_ Thurnherr, A. M. (2012). The finescale response of lowered ADCP velocity measure-ments 
    processed with different methods. Journal of Atmospheric and Oceanic Technology, 29(4), 
    597–600. https://doi.org/10.1175/JTECH-D-11-00158.1
    
    
    '''
    
    if type(u) != np.ndarray:
        u = u.to_numpy()
        v = v.to_numpy()
        
    if np.any(np.isnan(u)):
        raise TypeError('Not a continuous profile: NaN found in velocity profile.')
    if np.any(np.isnan(v)):
        raise TypeError('Not a continuous profile: NaN found in velocity profile.')
        
    # check if array is power of 2
    
    def is_power_of_2(n):
        
        # written by chat GPT
        
        if n <= 0:
            return False
        return n & (n - 1) == 0
    
    if not is_power_of_2(u.size):
        hamming_numbers_100 = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 27, 30,\
        32, 36, 40, 45, 48, 50, 54, 60, 64, 72, 75, 80, 81, 90, 96, 100]
        
        if u.size not in hamming_numbers_100:
            raise ValueError('Array length is not a power of 2 or a Hamming Number <= 100')
        
    # create complex series of u + iv
    
    uv = u + 1j*v
    
    # normalize by buoyancy frequency 
    
    if N_normalize:
        
        if verbose:
        	print('(a) FFT of buoyancy frequency normalized value')
        
        assert float(np.mean(N)) >= 6.5*(10**-4),\
        'Buoyancy frequency is too low --> too weakly stratified --> CTD noise --> spectra contaminated\
        --> this window must be thrown out'
        
        uv = uv / float(np.mean(N))
    
    # fft
    
    fft = np.fft.fft(uv)
    
    wavenumbers = 2*np.pi * np.fft.fftfreq(u.size,5) # rad/m
    
    if transfer:
        
        if verbose:
        	print('(b) SPECTRAL COEFFICIENTS ARE BEING TRANSFERRED FOR THIS CALCULATION')
        
        # spectral transfer function

        def vel_transfer(z_bin, z_vres, m):

            '''
            implementation omitting tilt velocity spectral transfer function from 
            Thurnherr 2012, DOI: 10.1175/JTECH-D-11-00158.1

            wavenumber must be in rad/m
            '''

            z_pulse  = z_bin

            z_supens = z_vres

            t_ra     = ( np.sinc( (m*z_pulse) / (2*np.pi) ) )**2 * ( np.sinc( (m*z_bin) / (2*np.pi) ) )**2

            t_bin    = ( np.sinc( (m*z_vres) / (2*np.pi) ) )**2   

            t_supens = ( np.sinc( (m*z_supens) / (2*np.pi) ) )**2     

            return t_ra * t_bin * t_supens
        
        transfer = vel_transfer(10,5,wavenumbers)
        
        fft = fft / transfer 
    
    return (fft, wavenumbers)

#############################################################################################################
### Rotary Spectra ###

def rotary_spectra(u,v,N,N_normalize=True,transfer=True,density=True,shear=False,verbose=False):
    '''Computes the rotary power spectrum of u + iv 
    
    Parameters
    ----------
    u : xarray.core.dataarray.DataArray or np.ndarray 
        1D zonal velocity or vertical shear depth array 
    v : xarray.core.dataarray.DataArray or np.ndarray 
        1D meridional velocity or vertical shear depth array 
    N : xarray.core.dataarray.DataArray or np.ndarray 
        1D buoyancy frequency depth array 
    N_normalize : bool, optional 
        Determines if the Fourier coefficients are computed from the u + iv series normalized 
        by the depth mean of the given buoyancy frequency array. Default value is True. 
    transfer : bool, optional 
        Determines if the Fourier coefficients are spectrally corrected (transferred) according
        to Thurnherr, 2012. Default value is True
    density : bool, optional
        Determines if the resulting power is reported as power density (PSD). Default is True. 
    shear : bool, optional 
        Indicates that the Fourier transform is done on shear values. Default is False. Must be 
        selected for correct shear transform. 
    verbose : bool, optional
    	Determines if various settings are printed. Default is False. 
    
    Returns
    -------
    (ccw_power, cw_power, ccw_wn, cw_wn, zero_f_power) : tuple 
        ccw_power : np.ndarray
            Array of counterclockwise power spectrum from the Fourier transform of u + iv
        cw_power : np.ndarray
            Array of counterclockwise power spectrum from the Fourier transform of u + iv
        ccw_wn : np.ndarray
            Array of counterclockwise wavenumbers, in ascending order, corresponding to `ccw_power`
        cw_wn : np.ndarray 
            Array of counterclockwise wavenumbers, in ascending order, corresponding to `cw_power`. 
            Identical to `ccw_wn` and reported as positive wavenumbers even though they correspond to 
            negative wavenumbers in the Fourier transform. 
        zero_f_power : float 
            Power of the zero-wavenumber component. 
    
    Raises
    ------
    ValueError if the size of u and v is not a power of 2. 
    AssertionError if the mean buoyancy frequency is < 6.5*(10**-4)
    AssertionError if the np.trapz trapezoidal quadrature (integral) of the PSD (if density selected)
        is not within 1.0 of the mean square magnitude of the original signal (Parseval's Theorem). This
        tolerance is arbitrarily chosen and remains to be changed. 
    
    Notes
    -----
    Power is defined here as 1/N^2 * |fft_coeff|^2 where N is the size of the data and the normalization
    coefficient, fft_coeff is the given Fourier coefficient, and |•| is the complex norm. 
    
    '''
    
    # get fft and wavenumbersif 
    
    if shear:
        fft,wavenumbers = uv_sh_fft(u,v,N,N_normalize,transfer)
    else:
        fft,wavenumbers = uv_fft(u,v,N,N_normalize,transfer)
    
    N_data = u.size
    
    # partition fft and wavenumbers by positive and negative wavenumber
    
    neg_mask = wavenumbers<0
    pos_mask = wavenumbers>0
        
    pos_wn = wavenumbers[pos_mask]
    neg_wn = wavenumbers[neg_mask]
        
    zero_f  = fft[0]
    pos_fft = fft[pos_mask]
    neg_fft = fft[neg_mask]
    
    # calculate power
    
    pos_power    = (1/N_data**2) * np.absolute(pos_fft)**2
    neg_power    = (1/N_data**2) * np.absolute(neg_fft)**2
    zero_f_power = (1/N_data**2) * np.absolute(zero_f)**2  
    
    # flip negative power array because the negative wavenumbers are descending in magnitude
    # reformulate this as clockwise and counterclockwise power
    
    ccw_wn = -1*np.flip(neg_wn)
    cw_wn  = pos_wn
    
    ccw_power = np.flip(neg_power)
    cw_power  = pos_power
    
    # number of powers check
    assert (ccw_power.size + cw_power.size + zero_f_power.size) == N_data,\
    "The number of power values in frequency space for your partition is not equal to the number of signal values"
    
    # Discrete Parseval's Theorem check
    # mean of signal power == 1/N mean frequency power
    # only works for un-transferred spectral coefficients
    
    if not transfer:
        
        if N_normalize:
            bf_mean = float(np.mean(N))
            square_mag_signal = [np.absolute(t)**2 for t in (u + 1j*v)/bf_mean]
        else:
            square_mag_signal = [np.absolute(t)**2 for t in (u + 1j*v)]
           
        assert np.isclose(np.mean(square_mag_signal), (ccw_power.sum() + cw_power.sum() + zero_f_power),atol=np.finfo(float).eps), \
        "Parseval's theorem does not hold for the partition of the power"
        
        
    # omit the nyquist wavenumber power in the negative power
    ccw_wn = ccw_wn[:-1]
    ccw_power = ccw_power[:-1]
    
    # report as spectral density (divide power by wavenumber bin length)
    
    if density:
        
        if verbose:
        	print('(c) Reporting as power spectral *density*')
        
        bin_diff   = cw_wn[2] - cw_wn[1]
        assert np.isclose(bin_diff, (2*np.pi)/(N_data * 5)) # check resolution bandwidth 2 ways
        ccw_power  = ccw_power / bin_diff 
        cw_power   = cw_power / bin_diff 
        
        if not transfer: # can only check for raw transform
            # check Parseval's theorem in integral form

            ordered_spectra = list(neg_power) + [zero_f_power] + list(pos_power)
            ordered_spectra = np.array(ordered_spectra) / bin_diff

            # must be within 1.0 to account for numerical approximation

            assert np.isclose(np.mean(square_mag_signal), np.trapz(ordered_spectra,dx=bin_diff),atol=1),\
            "numerical integral of Parseval's theorem doesn't hold"
        
        
    
    return (ccw_power, cw_power, ccw_wn, cw_wn, zero_f_power) 

#############################################################################################################
### Rotary Spectra Plotting ### 

def plot_rotary_spectra(ccwpower,cwpower,ccwwn,cwwn,density=True,N_normalize=True,shear=False):
    '''Plots rotary spectrum 
    
    Parameters
    ----------
    ccwpower : np.ndarray or list 
        counterclockwise power values
    cwpower : np.ndarray or list 
        clockwise power values
    ccwwn : np.ndarray or list 
        counterclockwise wavenumbers 
    cwwn : np.ndarray or list 
        clockwise wavenumbers 
    density : bool, optional 
        Indicates units for PSD or periodogram values. Default is True. 
    N_normalize : bool, optional 
        Indicates units for buoyancy frequency normalized DFT derived power or straight velocity/shear
        DFT derived power. Default is True. 
    shear : bool, optional 
        Indicate units for velocity or shear values. Default is False. 
    
    Returns
    -------
    None
    
    Notes
    -----
    Units are hard coded based on dimensional analysis of the procedures used in the work on 
    the SR1b dataset FSDB_SR1b_V0.1_20221013 in the thesis by Diaz, 2023. 
    
    Plots in log-log space. 
    
    Plots a -2 slope line for velocity spectra and a 0 slope line for shear spectra. 
    
    Plots spectral cutoffs in Thurnherr et al. (n.d.): 60m and 180m wavelengths. These are
    hard coded and have yet to be changed to a parameter. 
    
    '''
    
    plt.figure(dpi=200)
    
    plt.plot(ccwwn, ccwpower,label = 'CCW',c='g')
    plt.plot(cwwn, cwpower, label = 'CW',c='m')
    plt.scatter(ccwwn, ccwpower,marker = '.',c='g',edgecolor='k')
    plt.scatter(cwwn, cwpower,marker='.',c='m',edgecolor='k')
    
    x = np.linspace(0.01, 1, 100)
    if shear:
        # constant line for white shear spectra 
        y = np.full(x.size,np.median(ccwpower)) 
    else:
        # slope -2 in loglog space -- empirical observation of velocity spectra
        y = 0.001*x**-2
    
    plt.plot(x,y,c='b',linestyle=':',alpha=0.5)
    
    plt.axvline((2*np.pi)/60, c='k',alpha=0.5,linestyle = ":")
    plt.axvline((2*np.pi)/180, c='k',alpha=0.5,linestyle = ":")
    
    plt.gca().set_xscale('log',base=10)
    plt.gca().set_yscale('log',base=10)
    plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
    
    plt.grid(linewidth=0.2)
    plt.legend()
    
    # sort out units 
    
    if shear:
        
        if density and N_normalize:
            plt.ylabel(r'Power Spectral Density\ [(rad/m)$^{-1}$)]')
        elif density:
            plt.ylabel(r'Power Spectral Density\ [(1/s$^2$)$\cdot$(rad/m)$^{-1}$]')
        elif N_normalize:
            plt.ylabel(r'Spectral Power [1/rad]')
        elif (not density) and (not N_normalize):
            plt.ylabel(r'Spectral Power [1/s$^2$]')
    else:
        
        if density and N_normalize:
            plt.ylabel(r'Power Spectral Density\ [m$^2$ $\cdot$ (rad/m)$^{-1}$]')
        elif density:
            plt.ylabel(r'Power Spectral Density\ [(m$^2$/s$^2$)$\cdot$(rad/m)$^{-1}$]')
        elif N_normalize:
            plt.ylabel(r'Spectral Power [m$\cdot$(rad/m)$^{-1}$]')
        elif (not density) and (not N_normalize):
            plt.ylabel(r'Spectral Power [m$^2$/s$^2$]')

    plt.xlabel(r'Wavenumber ($\frac{rad}{m}$)')
    
    
    return None

#############################################################################################################
### Shear Calculation ### 

def shear(u,v,resolution=5):
    
    '''Calculates the whole profile vertical shear of horizontal velocity. 
    
    Parameters
    ----------
    u : xarray.core.dataarray.DataArray or np.ndarray 
        1D zonal velocity or vertical shear depth array 
    v : xarray.core.dataarray.DataArray or np.ndarray 
        1D meridional velocity or vertical shear depth array 
    resolution : int or float, optional
        vertical resolution of the u, v grid in [m] for shear calculation. Default is 5m. 
    
    Returns
    -------
    (u_sh, v_sh) : tuple
        u_sh : np.ndarray
            Array of vertical shear of zonal velocity 
        v_sh : np.ndarray
            Array of vertical shear of meridional velocity 
    
    Notes:
    ------
    Must be implemented for full profile
    
    You lose the last depth with the first differencing shear calculation
    the resulting profile is then to [:N_data]
    
    This method differences with z increasing into the ocean. 
    '''
    
    if type(u) != np.ndarray:
        u = u.to_numpy()
        v = v.to_numpy()
    
    diffu = np.diff(u)
    diffv = np.diff(v)
    
    u_sh = diffu / resolution
    v_sh = diffv / resolution
    
    return (u_sh, v_sh)

#############################################################################################################
### Shear FFT ### 

def uv_sh_fft(u_sh,v_sh,N,N_normalize=True,transfer=True,verbose=False):
    '''Fast Fourier Transform of vertical shear profile du/dz + idv/dz
        
    Parameters
    ----------
    u_sh : xarray.core.dataarray.DataArray or np.ndarray 
        1D zonal vertical shear depth array 
    v_sh : xarray.core.dataarray.DataArray or np.ndarray 
        1D meridional vertical shear depth array 
    N : xarray.core.dataarray.DataArray or np.ndarray 
        1D buoyancy frequency depth array 
    N_normalize : bool, optional 
        Determines if the Fourier coefficients are computed from the u + iv series normalized 
        by the depth mean of the given buoyancy frequency array. Default value is True. 
    transfer : bool, optional 
        Determines if the Fourier coefficients are spectrally corrected (transferred) according
        to Thurnherr, 2012. Default value is True
    verbose : bool, optional
    	Determines if various settings are printed. Default is False. 
    
    Returns
    -------
    (fft, wavenumbers) : tuple
        fft : np.ndarray 
            Array of Fourier coefficients for the transform of du/dz + idv/dz 
        wavenumbers : np.ndarray 
            Array of wavenumbers in [rad/m], ordered as described in np.fft.fftfreq
            
    Raises
    ------
    TypeError if a NaN is encountered in the shear series. 
    ValueError if the size of u and v is not a power of 2. 
    AssertionError if the mean buoyancy frequency is < 6.5*(10**-4)
    
    See Also
    --------
    np.fft.fft for a full description of the specific Discrete Fourier Transform implementation. 
    https://numpy.org/doc/stable/reference/generated/numpy.fft.fft.html#numpy.fft.fft
    
    Notes
    -----
    This routine requires the input arrays to be a power of 2 in order to avoid zero padding and the 
    associated edge effects. 
    
    Should be rolled into the velocity FFT function `uv_fft` as an option. 
    
    [1]_ Thurnherr, A. M. (2012). The finescale response of lowered ADCP velocity measure-ments 
    processed with different methods. Journal of Atmospheric and Oceanic Technology, 29(4), 
    597–600. https://doi.org/10.1175/JTECH-D-11-00158.1
    
    '''
    
    if type(u_sh)!= np.ndarray:
        u_sh = u_sh.to_numpy()
        v_sh = v_sh.to_numpy()
        
    if np.any(np.isnan(u_sh)):
        raise TypeError('Not a continuous profile: NaN found in velocity profile.')
    if np.any(np.isnan(v_sh)):
        raise TypeError('Not a continuous profile: NaN found in velocity profile.')
    
    # chop off last N value
        
    # check if array is power of 2
    
    def is_power_of_2(n):
        
        # written by chatGPT
        
        if n <= 0:
            return False
        return n & (n - 1) == 0

    if not is_power_of_2(u_sh.size):
        hamming_numbers_100 = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 27, 30,\
        32, 36, 40, 45, 48, 50, 54, 60, 64, 72, 75, 80, 81, 90, 96, 100]
        
        if u_sh.size not in hamming_numbers_100:
            raise ValueError('Array length is not a power of 2 or a Hamming Number <= 100')
        
    # create complex series of u + iv
    
    uv_sh = u_sh + 1j*v_sh
    
    # normalize by buoyancy frequency 
    
    if N_normalize:
        
        if verbose:
        	print('(a) FFT of buoyancy frequency normalized value')
        
        assert np.mean(N) >= 6.5*(10**-4),\
        'Buoyancy frequency is too low --> too weakly stratified --> CTD noise --> spectra contaminated\
        --> this window must be thrown out'
        
        uv_sh = uv_sh / float(np.mean(N))
    
    # fft
    
    fft = np.fft.fft(uv_sh)
    
    wavenumbers = 2*np.pi * np.fft.fftfreq(u_sh.size,5) # rad/m
    
    if transfer:
        
        if verbose:
        	print('(b) SPECTRAL COEFFICIENTS ARE BEING TRANSFERRED FOR THIS CALCULATION')
        
        # spectral transfer function

        def vel_transfer(z_bin, z_vres, m):

            '''
            implementation omitting tilt velocity spectral transfer function from 
            Thurnherr 2012, DOI: 10.1175/JTECH-D-11-00158.1

            wavenumber must be in rad/m
            '''

            z_pulse  = z_bin # adcp pulse width = adcp velocity bin

            z_supens = z_vres

            t_ra     = ( np.sinc( (m*z_pulse) / (2*np.pi) ) )**2 * ( np.sinc( (m*z_bin) / (2*np.pi) ) )**2

            t_bin    = ( np.sinc( (m*z_vres) / (2*np.pi) ) )**2   

            t_supens = ( np.sinc( (m*z_supens) / (2*np.pi) ) )**2     

            return t_ra * t_bin * t_supens
        
        def shear_transfer(z_bin, z_vres, m):
            
            t_rs = ( np.sinc( (m*z_bin)/(2*np.pi) ) )**2
            
            t_fd = 1 / ( np.sinc( (m*z_vres)/(2*np.pi) ) )**2
                    
            return vel_transfer(10,5,wavenumbers) * t_rs * t_fd
        
        transfer = shear_transfer(10,5,wavenumbers)
        
        fft = fft / transfer 
        
    return (fft, wavenumbers)

#############################################################################################################
### Single Rotary Coefficient Calculation ### 
                    
def rotary_coefficient(ccw_power,cw_power,wavenumbers,cutoffs,density=True,method='coeff last',bin_diff=(2*np.pi/320)):
    '''Computes the rotary coefficient for a given rotary spectrum 
    
    Parameters
    ----------
    ccw_power : np.ndarray 
        Array of power in the counterclockwise rotation sense. 
    cw_power : np.ndarray 
        Array of power in the clockwise rotation sense. 
    wavenumbers : np.ndarray
        Ordered (increasing) positive wavenumbers. Is symmetric to the negative wavenumbers.
    cutoffs : list of two floats or ints
        Spectral wavenumber cutoffs for integration of power. Must contain two numbers, unordered is fine: 
        low wavenumber cutoff and high wavenumber cutoff. 
    density : bool, optional
        Determines if we integrate a PSD (True) or sum a periodogram (False). Default is True. 
    method : string, optional 
        method='coeff first' calculates the rotary coefficient as the average rotary coefficient at 
        each wavnumber (see notes). method='coeff last' is the traditional and defaulty implementation 
        of the rotary coefficient calculation based on total power in the integration band. 
    bin_diff : float or int, optional
        Bandwidth resolution of the spectra. Default value is (2*np.pi/320) corresponding to the 320m 
        standard windows as in Thurnherr et al. (n.d.). 
    
    Returns
    -------
    R_c : float 
        Rotary coefficient (also stylized as C_m) for the given power spectrum 
    
    Raises
    ------
    AssertionError if any rotary coefficient is >1 or <-1 
    
    Notes
    -----
    method='coeff first' is an experimental technique that does not exactly correspond to 
    the traditional rotary coefficient as computed in method='coeff last'. 'coeff first' computes
    intermediate rotary coefficients at each wavenumber, and then averages them. This is different than
    the typical implementation which considers the ratio represented by using total power in the clockwise 
    and counterclockwise directions across all wavenumbers. method='coeff first' represents the average
    ratios of energy at each wavenumber, effectively un-weighting high energy wavelength waves. This tells
    us more about, theoretically, the dominant number of waves at a depth. 
    
    [1]_ Thurnherr, A. M., Firing, Y., Waterman, S., & Chereskin, T. (n.d.). Summertime
    internal-wave properties along the SR1b section across drake passage. Journal
    of Physical Oceanography, 16.
    '''

    # sort out left and right cutoffs for integration if high/low wavenumber cutoffs
    # are ordered otherwise 
    
    if cutoffs == None:
        pass
    else:
        left_cutoff, right_cutoff = sorted(cutoffs)

        integration_limits1 = wavenumbers >= left_cutoff
        integration_limits2 = wavenumbers <= right_cutoff
        integration_limits  = integration_limits1 & integration_limits2

        cw_power  = cw_power[integration_limits]
        ccw_power = ccw_power[integration_limits]
        
    if density:
        # PSD integration
        if method == 'coeff last':
            total_ccw_power = np.trapz(ccw_power,dx=bin_diff)
            total_cw_power  = np.trapz(cw_power,dx=bin_diff)
            R_c = (total_cw_power - total_ccw_power)/(total_cw_power + total_ccw_power)
            assert abs(R_c) <= 1, 'rotary coeffient is >1 or <-1'
            return R_c
        elif method == 'coeff first':
            interm_R_c = np.array( [(cw_power[i] - ccw_power[i]) / ((cw_power[i] + ccw_power[i])) for i in range(len(ccw_power))])
            assert np.all(interm_R_c >= -1) and np.all(interm_R_c <= 1), 'some rotary coefficient >1 or <-1'
            return interm_R_c.mean()
    else:
        # periodogram sum 
        if method == 'coeff last':
            total_ccw_power = np.sum(ccw_power)
            total_cw_power  = np.sum(cw_power)
            R_c = (total_cw_power - total_ccw_power)/(total_cw_power + total_ccw_power)
            assert abs(R_c) <= 1, 'rotary coeffient is >1 or <-1'
            return R_c
        elif method == 'coeff first':
            interm_R_c = np.array( [(cw_power[i] - ccw_power[i]) / ((cw_power[i] + ccw_power[i])) for i in range(len(ccw_power))])

            assert np.all(interm_R_c >= -1) and np.all(interm_R_c <= 1), 'some rotary coefficient >1 or <-1'
            return interm_R_c.mean()
                
    print('R_c not computed')
            
#############################################################################################################
### Rotary Coefficient Profile Function (Half-Overlapping Window) ### 

def rotary_coefficient_profile(initial_lag_m, window_length_m, u, v, N,\
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
    # if window is past last velocity depth, raise error 
    if (initial_lag_m + window_length_m) > vel_depth[-1]:
        raise ValueError('Data too shallow for this windowing procedure')

    max_n = int((vel_depth[-1] - window_length_m)/(window_length_m/2))
    lags  = [initial_lag_m + n*(window_length_m/2) for n in range(max_n)]
    midp  = np.array(lags[1:] + [lags[-1] + window_length_m / 2])
    
    u_sh,v_sh = shear(u,v)
    
    R_c_vel_prof = np.empty(max_n)
    R_c_sh_prof  = np.empty(max_n)
    
    for n in range(max_n):
        try:
            vel_data = window_data(lags[n], window_length_m, u, v, N, vel_depth, ctd_depth, v_spacing)
            shear_data = window_data(lags[n], window_length_m, u_sh, v_sh, N, vel_depth, ctd_depth, v_spacing)

        # catch CTD/LADCP overlap problem described in `window_data`
        except ValueError:
            R_c_vel = np.nan
            R_c_shear = np.nan 
            R_c_vel_prof[n] = R_c_vel
            R_c_sh_prof[n]  = R_c_shear

            continue

        try:
            vel_spectra = rotary_spectra(vel_data[0],vel_data[1],vel_data[2],N_normalize=N_normalize,transfer=transfer,density=density,shear=False)
            shear_spectra = rotary_spectra(shear_data[0],shear_data[1],shear_data[2],N_normalize=N_normalize,transfer=transfer,density=density,shear=True)
        
            R_c_vel   = rotary_coefficient(vel_spectra[0],vel_spectra[1],vel_spectra[2],cutoffs=cutoffs,method=method)
            R_c_shear = rotary_coefficient(shear_spectra[0],shear_spectra[1],shear_spectra[2],cutoffs=cutoffs,method=method)
        
        except (AssertionError, ValueError, TypeError):
            R_c_vel = np.nan
            R_c_shear = np.nan
        
        R_c_vel_prof[n] = R_c_vel
        R_c_sh_prof[n]  = R_c_shear
        
    return (R_c_vel_prof, R_c_sh_prof, midp)

#############################################################################################################
### Rotary Coefficient Profile Plotting ###  

def plot_rotary_coeff(vel_coeff, sh_coeff, depths, vel_depth, vel_err=[], sh_err=[], error_bars=False, subtitle=''):
    '''Plots a vertical profile of velocity and shear coefficients 
    
    Parameters
    ----------
    vel_coeff : np.ndarray
        Array of velocity derived rotary coefficients with depth 
    sh_coeff : np.ndarray
        Array of shear derived rotary coefficients with depth 
    depths : np.ndarray
        Array of depths in [m] at which the rotary coefficients were calculated or assigned. 
    vel_depth : np.ndarray
        Array of depths at which velocity measurements that yielded above rotary coefficients were made. 
    vel_err : list or np.ndarray, optional 
        Rotary coefficient confidence intervals for velocity derived data. Must be ordered shallow to deep.  
    sh_err : list or np.ndarray, optional
        Rotary coefficient confidence intervals for velocity derived data. Must be ordered shallow to deep.  
    error_bars : bool, optional
        Indicates whether error bar data is provided and will be plotted. Default is False. 
   
    Returns
    -------
    None 
    
    Notes
    -----
    Plots horizontal red lines at 150m and the last actual velocity measurement depth. 
    
    '''
    plt.figure(dpi=200)

    plt.scatter(vel_coeff,depths,c='teal',label = 'velocity',s=20)
    plt.plot(vel_coeff,depths,ls=':',c='k',alpha=0.5)
    
    plt.scatter(sh_coeff,depths,c='violet',label = 'shear',s=20)
    plt.plot(sh_coeff,depths,ls=':',c='k',alpha=0.5)
    
    if error_bars:
        plt.errorbar(vel_coeff, depths, xerr=vel_err, c='teal', fmt='none', elinewidth=0.5, capsize=1.5)
        plt.errorbar(sh_coeff, depths, xerr=sh_err, c='violet', fmt='none', elinewidth=0.5, capsize=1.5)
    
    plt.gca().set_ylim([0,vel_depth[-1] + 100])
    plt.axhline(vel_depth[-1],c='r')
    plt.axhline(150,c='r')
    plt.axvline(0,c='k',alpha=0.4)
    plt.gca().invert_yaxis()
    plt.grid()
    plt.xlabel(r'$C_m$')
    plt.ylabel('Depth [m]')
    plt.xlim([-1,1])
    plt.gca().set_aspect(0.001)
    plt.legend(loc = 'best')
    plt.title(r'Velocity and Shear Derived $C_m$'+'\n'+subtitle)
