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

def _get_concentration_contribution_top_bedrock_cosmogenic_exponential(\
    cnp.double_t [:] denudation_rate, cnp.double_t decay_constant, cnp.double_t [:,:] absorption_coefficient,
    cnp.double_t [:,:] concentration_contribution_previous, cnp.double_t [:] slhl_prod_rate,
    cnp.double_t [:,:] scaling_factor, cnp.double_t dt, cnp.double_t [:] target_mineral_presence,
    cnp.double_t [:] thickness_cover):
    
    cdef:
        cnp.double_t eps, lam, mu, N_0, P_0, scaling, t, target, z, val, concentration, \
            exp_lam_t, exp_mu_t_eps, mu_eps, mu_eps_exp_mu_t_eps_1
        cnp.int_t pathway, node_id, pathways_n, nodes_n
        cnp.double_t [:,:] _return 
    pathways_n = len(slhl_prod_rate); nodes_n = len(denudation_rate)
    lam = decay_constant
    t = dt
    _return = np.empty((pathways_n + 1, nodes_n)) # contain the 1 to 3 pathway contribution + total concentration, for each node
    
    exp_lam_t = exp(- lam * t)
            
    for node_id in range(nodes_n):
        _return[pathways_n, node_id] = 0 # total concentration
        for pathway in range(pathways_n):
            eps = denudation_rate[node_id] 
            mu = absorption_coefficient[pathway, node_id] # BEWARE: coefficient for cover
            N_0 = concentration_contribution_previous[pathway, node_id]           
            P_0 = slhl_prod_rate[pathway]
            scaling = scaling_factor[pathway, node_id]
            target = target_mineral_presence[node_id]
            z = thickness_cover[node_id]

            mu_eps = mu * eps
            exp_mu_t_eps = exp(- mu_eps * t) # exp(-mu * eps * t) | Taylor polynomial: 1 - mu * eps * t
            
            
            if eps > 1.e-6:
                mu_eps_exp_mu_t_eps_1 = 1 / (mu_eps) * (exp_mu_t_eps - 1) # 1/ (mu * eps) * (exp(-mu * eps * t) - 1)  | Taylor polynomial: -t
                
                val = scaling * target * P_0 * (- mu_eps_exp_mu_t_eps_1) \
                        + P_0 * lam / (mu_eps) * (t * exp_mu_t_eps + mu_eps_exp_mu_t_eps_1) 
            else: #Taylor polynomial to prevent anomalous behaviors for eps close to zero.
                    # below value 1.e-9  the formula gives anomalous result. Depends on MACHINE ?
                    # Taylor polynomial: scaling * target * P_0 * (t) \
                   # + P_0 * lam / (mu_eps) * (t * (1 - mu * eps * t) - t) = scaling * target * P_0 * t - P_0 * lam * t^2
                val = scaling * target * P_0 * t - P_0 * lam * t * t
                
            val += N_0 * exp_mu_t_eps * (exp_lam_t)  # radioactive decay-corrected inherited concentration related to production pathway
                                                           # applied to the slice of rock advected up to surface
            """
            val = N_0 * np.exp(-mu * eps * t ) * (np.exp(- lam * t))  # radioactive decay-corrected inherited concentration related to production pathway
                                                       # applied to the slice of rock advected up to surface
        
            val += scaling * target * P_0 / ( -mu * eps) * (np.exp(-mu * eps * t) - 1) \
                + P_0 * lam / (mu * eps) * (t * np.exp(-mu * eps * t) + 1 / (mu * eps) * (np.exp(-mu * eps * t) - 1)) """

            _return[pathways_n, node_id] += val
            _return[pathway, node_id] = val
        
    return _return