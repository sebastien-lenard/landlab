# cython: profile=True
# distutils: language = c++

import numpy as np
cimport numpy as cnp
cimport cython
cimport openmp
from cython.parallel cimport prange
from cython.parallel cimport threadid
from cython.operator cimport dereference
from libcpp cimport bool
from libc.math cimport exp, log

def count_c(cnp.int_t n, cnp.int_t multithread):
    
    cdef:
        cnp.int_t i, sum = 0
    if multithread == 1:
        for i in prange(n, nogil=True):
            #with gil: print(threadid(), flush=True)
            sum += i
    else:
        for i in range(n):
            #print(threadid(), flush=True)
            sum += i
    return sum

    
cdef cnp.double_t [:] toto_c(cnp.double_t [:] array) nogil:
    cdef:
        cnp.double_t [:] _return
        cnp.int_t i, n = len(array)
    with gil: _return = np.empty(n)
    for i in prange(n, num_threads=2):
        _return[i] = array[i] + array[i] #array[i] * array[i] + 
    return _return

def toto(cnp.double_t [:] array):
    return toto_c(array)

# This function is not as performant as expected with high power_n (4096 for accuracy)
cdef cnp.double_t exp_approximation(cnp.double_t x, cnp.double_t inv_power_n, cnp.int_t n) nogil:
    """ Approximate the math exponential function (for performance purpose)
    based on exp(x) = lim(n-> inf)(1 + x/n)^n
    works for small (or negative values of x) < 3
    used here, probably lead to less than 1% error on scaling factors
    
    power_n is calculated thanks to:
    n = 12 #16
    k = 2
    power_n = 2
    for i in range(n-1):
        power_n *= k
    power_n = 1 / power_n
    => power_n = 1/4096 = 0.000244140625 (beware for computation on x, power_n should be the same type)
    """
    x = 1.0 + x * inv_power_n
    for i in range(n):
        x *= x
    return x

@cython.boundscheck(False) #doesn't speed up?
@cython.wraparound(False)  #doesn't speed up? and contiguous memory views ::1, does it really speed up?
def _get_cosmogenic_scaling_factor_lal_stone_2000(cnp.double_t [:,::1] stone_scaling_coefficient_interp, 
    cnp.double_t [:] z, cnp.double_t pressure_mean, cnp.double_t temperature_mean, cnp.double_t dT_dz_mean,
    cnp.int_t threading):
    """atmospheric_pressure in hPa
    """
    cdef:
        cnp.double_t [:,::1] s
        cnp.double_t [:,:] _return
        cnp.double_t [:] P
        cnp.int_t i, n
        cnp.double_t p, p_2, p_3, p_exp, inv_dt_dz_mean, log_temperature_mean
    s = stone_scaling_coefficient_interp
    n = len(z)
    _return = np.empty((2, n)) # 1st line: spallation factors, 2nd line: local atmospheric pressure
    
    inv_dt_dz_mean = -0.03417 / dT_dz_mean
    log_temperature_mean = log(temperature_mean)
    if threading == 1:
        for i in prange(n, nogil=True):
            p = pressure_mean * exp(inv_dt_dz_mean * (log_temperature_mean - \
                log(temperature_mean - dT_dz_mean * z[i]))) 
            _return[1, i] = p
            p_2 = p * p
            p_3 = p_2 * p
            p_exp = exp(- p / 150.) # 0.000244140625 = 1/4096.exp_approximation(- P[i] / 150., 0.000244140625, 12) if fast_exp == 1 else 
            _return[0, i] = s[0, i] + s[1, i] * p_exp + s[2, i] * p + s[3, i] * p_2 + s[4, i] * p_3
    else:
        for i in range(n): #, nogil=True)
            p = pressure_mean * exp(inv_dt_dz_mean * (log_temperature_mean - \
                log(temperature_mean - dT_dz_mean * z[i]))) 
            _return[1, i] = p
            p_2 = p * p
            p_3 = p_2 * p
            p_exp = exp(- p / 150.) # 0.000244140625 = 1/4096.exp_approximation(- P[i] / 150., 0.000244140625, 12) if fast_exp == 1 else 
            _return[0, i] = s[0, i] + s[1, i] * p_exp + s[2, i] * p + s[3, i] * p_2 + s[4, i] * p_3
            
    return np.asarray(_return[0, :]), np.asarray(_return[1, :])

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def _get_cosmogenic_scaling_factor_lal_stone_2000_braucher_2011(\
    cnp.double_t [:,::1] stone_scaling_coefficient_interp, cnp.double_t [:] z, 
    cnp.double_t pressure_mean, cnp.double_t temperature_mean, 
    cnp.double_t dT_dz_mean, cnp.double_t e_folding_length_in_air_muon_fast,
    cnp.double_t e_folding_length_in_air_muon_slow, cnp.int_t threading):
    """atmospheric_pressure in hPa
    """
    cdef:
        cnp.double_t [:,::1] s, _return
        cnp.int_t i, n
        cnp.double_t p, p_2, p_3, p_exp, p_muon, inv_dt_dz_mean, log_temperature_mean
    s = stone_scaling_coefficient_interp
    n = len(z)
    _return = np.empty((4, n))# _return 1st line: spallation factors, 2nd line: local atmospheric pressure
    
    inv_dt_dz_mean = -0.03417 / dT_dz_mean
    log_temperature_mean = log(temperature_mean)
    if threading == 1:
        for i in prange(n, nogil=True):
            p = pressure_mean * exp(inv_dt_dz_mean * (log_temperature_mean - \
                log(temperature_mean - dT_dz_mean * z[i]))) 
            _return[3, i] = p
            p_2 = p * p
            p_3 = p_2 * p
            p_exp = exp(- p / 150.) # 0.000244140625 = 1/4096.  if fast_exp == 1 else exp_approximation(- p / 150., 0.000244140625, 12)
            _return[0, i] = s[0, i] + s[1, i] * p_exp + s[2, i] * p + s[3, i] * p_2 + s[4, i] * p_3 # spallation
            p_muon = (1013.25 - p) * 10. # *10. because length is in SI kg/m2
            _return[1, i] = exp(p_muon / e_folding_length_in_air_muon_fast)
            _return[2, i] = exp(p_muon / e_folding_length_in_air_muon_slow)
    else:
        for i in range(n):
            p = pressure_mean * exp(inv_dt_dz_mean * (log_temperature_mean - \
                log(temperature_mean - dT_dz_mean * z[i]))) 
            _return[3, i] = p
            p_2 = p * p
            p_3 = p_2 * p
            p_exp = exp(- p / 150.) # 0.000244140625 = 1/4096.exp_approximation(- P[i] / 150., 0.000244140625, 12) if fast_exp == 1 else 
            _return[0, i] = s[0, i] + s[1, i] * p_exp + s[2, i] * p + s[3, i] * p_2 + s[4, i] * p_3 # spallation
            p_muon = (1013.25 - p) * 10. # *10. because length is in SI kg/m2
            _return[1, i] = exp(p_muon / e_folding_length_in_air_muon_fast)
            _return[2, i] = exp(p_muon / e_folding_length_in_air_muon_slow)
    
    
    return np.asarray(_return[0, :]), np.asarray(_return[1, :]), np.asarray(_return[2, :]), np.asarray(_return[3, :])

def _set_cosmogenic_scaling_factor_lal_stone_2000_from_mem(
    cnp.double_t [:] z,
    cnp.double_t [:] cosmogenic_scaling_factor_spallation, 
    cnp.double_t [:] atmospheric_pressure_hPa,
    cnp.double_t [:] mem_cosmogenic_scaling_factor_spallation, 
    cnp.double_t [:] mem_atmospheric_pressure_hPa):
    cdef cnp.int_t i, idx, nodes_n = len(z)
    for i in range(nodes_n):
        idx = int(z[i])
        cosmogenic_scaling_factor_spallation[i] = mem_cosmogenic_scaling_factor_spallation[idx]
        atmospheric_pressure_hPa[i] = mem_atmospheric_pressure_hPa[idx]
    

def _set_cosmogenic_scaling_factor_lal_stone_2000_braucher_2011_from_mem(
    cnp.double_t [:] z,
    cnp.double_t [:] cosmogenic_scaling_factor_spallation, 
    cnp.double_t [:] cosmogenic_scaling_factor_muon_fast,
    cnp.double_t [:] cosmogenic_scaling_factor_muon_slow, 
    cnp.double_t [:] atmospheric_pressure_hPa,
    cnp.double_t [:] mem_cosmogenic_scaling_factor_spallation, 
    cnp.double_t [:] mem_cosmogenic_scaling_factor_muon_fast,
    cnp.double_t [:] mem_cosmogenic_scaling_factor_muon_slow, 
    cnp.double_t [:] mem_atmospheric_pressure_hPa):  
    cdef cnp.int_t i, idx, nodes_n = len(z)
    for i in range(nodes_n):
        idx = int(z[i])
        cosmogenic_scaling_factor_spallation[i] = mem_cosmogenic_scaling_factor_spallation[idx]
        cosmogenic_scaling_factor_muon_fast[i] = mem_cosmogenic_scaling_factor_muon_fast[idx]
        cosmogenic_scaling_factor_muon_slow[i] = mem_cosmogenic_scaling_factor_muon_slow[idx]
        atmospheric_pressure_hPa[i] = mem_atmospheric_pressure_hPa[idx]
                                                                         
