# This code was translated by chatGPT from the original Perl script from Dr. Andreas Thurnherr (personal communication)

# N = nData for my implementation 
# C is an array of fourier coefficients produced by the FOUR1 algorithm 

######## Periodogram Functions ##########

def pgram_onesided(nData, C):
    N = (len(C)+1) // 2  # number of Fourier components
    Pfac = N**(-2) * N/nData  # normalized to mean-sq amp
    P = [0] * (N//2+1)  # initialize periodogram

    P[0] = Pfac * (C[0]**2 + C[1]**2)  # calc periodogram
    for k in range(1, N//2):
        P[k] = Pfac * (C[2*k]**2 + C[2*k+1]**2 + C[2*(N-k)]**2 + C[2*(N-k)+1]**2)
    P[N//2] = Pfac * (C[2*(N//2)]**2 + C[2*(N//2)+1]**2)
    
    return P

def pgram_pos(nData, C):
    N = (len(C)+1) // 2  # number of Fourier components
    Pfac = N**(-2) * N/nData  # normalized to mean-sq amp
    P = [0] * (N//2+1)  # initialize periodogram

    P[0] = 0.5 * Pfac * (C[0]**2 + C[1]**2)  # calc periodogram
    for k in range(1, N//2):
        P[k] = Pfac * (C[2*k]**2 + C[2*k+1]**2)
    P[N//2] = 0.5 * Pfac * (C[2*(N//2)]**2 + C[2*(N//2)+1]**2)
    
    return P

def pgram_neg(nData, C):
    N = (len(C)+1) // 2  # number of Fourier components
    Pfac = N**(-2) * N/nData  # normalized to mean-sq amp
    P = [0] * (N//2+1)  # initialize periodogram

    P[0] = 0.5 * Pfac * (C[0]**2 + C[1]**2)  # calc periodogram
    for k in range(1, N//2):
        P[k] = Pfac * (C[2*(N-k)]**2 + C[2*(N-k)+1]**2)
    P[N//2] = 0.5 * Pfac * (C[2*(N//2)]**2 + C[2*(N//2)+1]**2)
    
    return P

########## FOUR1 FFT Function #############

from math import sin

def FOUR1(isign, data):
    N = len(data) // 2
    n = N << 1
    j = 0

    for i in range(0, n-1, 2):
        if j > i:
            data[j], data[i] = data[i], data[j]
            data[j+1], data[i+1] = data[i+1], data[j+1]
        if n % 2 != 0:
            raise ValueError(f"{N} is not a power of two")
        m = n >> 1
        while m >= 2 and j >= m:
            if m % 2 != 0:
                raise ValueError(f"{N} is not a power of two")
            j -= m
            m >>= 1
        j += m

    mmax = 2
    while n > mmax:
        istep = mmax << 1
        theta = isign * (6.28318530717959 / mmax)
        wtemp = sin(0.5 * theta)
        wpr = -2.0 * wtemp * wtemp
        wpi = sin(theta)
        wr = 1.0
        wi = 0.0
        for m in range(0, mmax-1, 2):
            for i in range(m, n, istep):
                j = i + mmax
                tempr = wr * data[j] - wi * data[j+1]
                tempi = wr * data[j+1] + wi * data[j]
                data[j] = data[i] - tempr
                data[j+1] = data[i+1] - tempi
                data[i] += tempr
                data[i+1] += tempi
            wtemp = wr
            wr = wr * wpr - wi * wpi + wr
            wi = wi * wpr + wtemp * wpi + wi
        mmax = istep

    return data

############ A.T. notes: ##############
#----------------------------------------------------------------------
# Periodogram (p.421; (12.7.5) -- (12.7.6))
#----------------------------------------------------------------------

# Miscellaneous Notes:
#
# - there are N/2 + 1 values in the (unbinned) PSD (see NR)
    
# Notes regarding the effects of zero padding:
# 
# - using zero-padding on a time series where the mean is not removed
#   can result in TOTALLY DIFFERENT RESULTS, try it e.g. with
#   temperatures!!! (A moment's thought will reveal why)
#   
# - Because the total power is normalized to the mean squared amplitude
#   0-padded values depress the power; this was taken care of below by
#   normalizing the power by multiplying it with nrm=(nData+nZeroes)/nData;
#   
# - this was checked using `avg -m' (in case of complex input the total
#   power is given by sqrt(Re**2+Im**2));
# 
# - if zero-padding is used sqrt(nrm*P[0]) is mean value (done in [pgram])
    
# Notes on interpreting the power spectrum (Hamming [1989] & NR):
#
# - the frequency spectrum encompasses the frequencies between 0 (the
#   mean value) and the Nyquist frequency (1 / (2 x sampling interval))
# 
# - higher frequencies are aliased into the power spectrum in a mirrored
#   way (e.g. noise tends to (linearly?) approach zero as f goes to inft;
#   the downsloping spectrum `hits' the Nyquist frequency, turns around  
#   and continues falling towards the zero frequency, where it gets mirrored
#   again => spectrum flattens towards Nyquist frequency
#   
# - the sum over all P's (total power) is equal to the mean square value;
#   when one-sided spectra are used, P[0] and P[N/2] are counted doubly
#   and must be subtracted from the total; NB: the total power is reduced
#   if data are padded with 0es
#   
# - sqrt(P[0]) is an estimate for the mean value which is only accurate
#   if no zero-padding is perfomed; removing the mean will
#   strongly change the spectrum near the origin which might
#   or might not be a good thing, depending on the physics behind it (e.g.
#   it makes sense to remove the mean if a power spectrum from a temperature
#   record is calculated but not if flow velocity is used).
#   Removing higher order trends will also affect the spectrum but not
#   in such a simple fashion. Note that the problem is mainly restricted
#   to cases where the signal is in the low frequency and thus affected
#   by the strong changes in the spectrum there.

# Notes on the two-sided spectra:
# - the power of the mean flow (sqrt(P[0])) is non-rotary. To have the sum
#   of both one-sided spectra equal the two-sided one, each one-sided
#   spectrum gets half of the total value.
# - the same is true for the highest frequency; at the Nyquist frequency
#   every rotation is sampled exactly twice => polarization cannot be
#   determined (imagine a wheel with one spoke...)
