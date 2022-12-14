"""Calculates scaling coefficients to apply to sea-level high-latitude
cosmogenic production rates to obtain local cosmogenic production rates.

@author Sebastien Lenard
@creation 2022, May
"""

import numpy as np # mathematical operations and ndarrays
from landlab import Component, ModelGrid, RasterModelGrid # class inheritance and type of the main attribute of the class
import csv # to get the Lal-Stone coefficients, the cosmogenic parameters, the atmospheric ERA40 parameters
import landlab.core, landlab.core.messages # to get the yaml file and format messages
import scipy as sci # interpolation of 2-D array
from math import floor # extract average lat-long of the grid 
import copy # to solve the problem of modifying by componenent of optional arguments 
    #each time it is called (caused by the concatenation with path_prefix in code)

class CosmogenicProductionScaler(Component):
    """Calculates scaling coefficients to apply to sea-level high-latitude
    cosmogenic production rates to obtain local cosmogenic production rates.
    
    :func:`run_one_step`
    
    The scaling model of **Lal-Stone, 2000**, updated with the *ERA40 atmospheric data* by (Lifton et al., 2014) 
    their matlab code, Charreau et al., 2019 is used. 
    such that:: 1 + 0 = 1
    
    Scaling models usually yield similar results but some discrepancies are observed
    pecularly for low latitude 
    and/or high elevation settings (Lifton et al., 2014; Charreau et al., 2019).
   
    **Lal-Stone scaling model for spallation by neutron pathway.**

    We implemented the altitudinal-latitudinal scaling model of Lal-Stone (Lal, 1991; Stone, 2000)
    Nuclear disintegration rates and cosmogenic production rates vary as a function of the geomagnetic latitude and the altitude.
    The available cosmogenic production rates (e.g. in the Crep/Ice-D database) are calibrated with experiments 
    and natural measurements and scaled down to sea-level high-latitude (SLHL). To calculate production 
    at a certain location, we need to scale the SLHL production rate 
    to the geomagnetic latitude and elevation of the location
    
    The first version of the model, based on geomagnetic latitude and elevation, was proposed p. 427 in (Lal, 1991). 
    Scaling was obtained following a polynomial equation:
    s(y) = al + a2.y + a3.y^2 + a4.y^3, with the coefficients depending on geomagnetic latitude, and y the elevation.
    However, this model is valid only at mid-latitudes. The model relies on the standard atmospheric model
    (N.O.A.A. et al., 1976), which allows the
    use of elevation instead of atmospheric depth, but deviations in atmospheric pressure make it unrealistic,
    e.g. in Antarctica and Iceland (Stone, 2000; Lifton et al., 2014)
    
    A second version of the model, reparametered using geographic latitude and atmospheric pressure,
    was elaborated p. 23754 in (Stone, 2000).
    s(P) = a + b*exp[-P/150] + c*P + d*P^2 + e*P^3, with the coefficients depending on the geographic latitude 
    and P the atmospheric pressure. The pressure is determined from elevation using the hydrostatic equation described 
    in (N.O.A.A. et al., 1976) and formulated as equation (1) in (Stone, 2000).
    This model applies "to exposure periods long enough to average the motion of the 
    geomagnetic dipole axis but short enough for this pressure field to be representative" (Stone, 2000).
    
    **Muon scaling model.**

    We implemented an altitudinal-only scaling model for cosmogenic nuclide production from muon pathways (Braucher et al., 2011).
    Not considering the geomagnetic field probably has negligible
    impact on producte rate uncertainty for Be10 and Al26, but might induce more substantial error
    for C14     (see (Braucher et al., 2011) p.7 and discussion in (Balco et al., 2017) p. 169 and in (Charreau et al., 2019)
    
    We artificially separate scaling for low and high energy particles, following (Braucher et al., 2011), p.7-8, 
    that is for slow muon capture and fast muon interactions. We use Eq. (3) in (Braucher et al., 2011), with
    the e-folding length of 260 and 510 g/cm^2
    for slow and fast  muon contributions, calculated by (Braucher et al., 2011) using (Balco et al., 2008) calculator.
    
        
    Atmospheric model.
    ------------------
    The models representing the atmosphere are defined at sea-level.
    Initially (Lal, 1991), the U.S. Standard Atmosphere (N.O.A.A., 1976) was used
    A new atmospheric model, NCEP (reanalysis) was introduced by (Balco et al., 2008). This model yields little difference with
    the more uptodate model ERA40 model (Uppala et al., 2005) introduced by (Lifton et al., 2014), who
    uses data from ERA40 reanalysis and takes into account variable adiabatic lapse rate dT/dz 
    (in former models, dT/dz = constant = 0.0065 K/m). 
    Both atmospheric models perform better than the U.S. Standard Atmosphere model.
    See discussion here on Balco's blog:
    https://cosmognosis.wordpress.com/2015/10/16/elevationatmospheric-pressure-models/
    
    And check atmospheric equations vulgarization here:
    https://www.chemeurope.com/en/encyclopedia/Atmospheric_pressure.html#_ref-USSA1976_1/
    
    
    Implemented.
    Model of Lal-Stone, 2000 for spallation scaling
    Model of Braucher et al., 2011 for muon scaling
    Takes average latitude/longitude of the grid as input REFORMULATE
    Interpolation of ERA40 made only for the latitude and longitude on the center of the grid
    if topographic__elevation is not supplied, calculate scaling for sea-level topography
    
    Not implemented.
    Model of LSD, Lifton et al., 2014
    Temporal variations of the geomagnetic dipole
    Can't calculate for each latitude/long of the DEM (might be a problem for mega-watersheds (e.g. Ganga)
        requires something on the BMI Topography?
    Refined interpolation of ERA40 for an array of latitude/longitudes
    Although not done here, it is often better to estimate the average pressure at your 
        site using a pressure-altitude relation obtained from nearby station
        data (comment in Balco et al., 2017 ERA40atm script)
    Attenuation length model for neutrons in air of Merreto et al. 2016
    Muon scaling might be also implemented following 
        a refined way described in (Balco et al., 2017) and not implemented here
    
    Properties

    Examples
    --------
    >>>from landlab import RasterModelGrid, CosmogenicProductionScaler
    >>> # create a grid instance
    >>>myGrid = RasterModelGrid((8, 8), 10) #3 4
    >>># add random elevation (in m)
    >>>myGrid.add_field('topographic__elevation', 1000 * np.random.rand(myGrid.number_of_nodes), at='node', clobber=True)
    >>>obj = CosmogenicProductionCorrector(myGrid)
    >>>dt=1
    >>>obj.run_one_step(dt)
    >>> # get scaling factors
    >>> cosmogenic_scaling_factor
    >>> print(obj.cosmogenic_scaling_factor["spallation"])
    >>> print(obj.cosmogenic_scaling_factor["muon_slow"])
    >>> print(obj.cosmogenic_scaling_factor["muon_fast"])
    
    Notes
    -----
    
    
    Attributes
    ----------
    exposure : float
        Exposure in seconds.

    Methods
    -------
    
    References
    ----------
    **Required Software Citation(s) Specific to this Component**
    
    None listed

    **Additional References**
    
    Scaling models.
    Lal, D., Peters, B. 1967. Cosmic Ray Produced Radioactivity on the Earth. In: Sitte, K. (eds) 
        Kosmische Strahlung II / Cosmic Rays II. Handbuch der Physik / Encyclopedia of Physics, vol 9 / 46 / 2. Springer, Berlin, Heidelberg.
        https://doi.org/10.1007/978-3-642-46079-1_7
    Lal, D. 1991. Cosmic ray labeling of erosion surfaces: in situ nuclide production rates and erosion models. Earth Planet. Sci. Let. 104, 424-439
    Lifton, N. et al. 2014. Scaling in situ cosmogenic nuclide production rates using analytical approximations 
        to atmospheric cosmic-ray fluxes. Earth Planet. Sci. Let. 386, 149-160
    Stone, J. O. 2000. Air pressure and cosmogenic isotope production. J. Geophys. Res. Solid Earth 105, 23753–23759.
    
    Atmospheric models
    N.O.A.A. et al. 1976, U.S. Standard Atmosphere. U.S. Government Printing Office, Washington, D.C.
    Uppala, S. M. et al. 2005. The ERA-40 re-analysis. Q. J. R. Meteorol. Soc. 131, 2961e3012.
    
    Muons:
    Balco, G. 2017. Production rate calculations for cosmic-ray-muon-produced 10Be and 26Al benchmarked against geological calibration data. 
        Quat. Geochrono. 39, 150-173
    Braucher, R. 2011. Production of cosmogenic radionuclides at great depth: A multi element approach. Earth. Planet. Sci. Lett. 309, 1-9
    
    Implementation.
    Balco, G. et al. 2008. A complete and easily accessible means of calculating surface exposure ages 
        or erosion rates from 10Be and 26Al measurements. Quat. Geochron. 3(3), 174-195.
    Charreau, J. et al. 2019. Basinga: A cell-by-cell GIS toolbox for computing basin average scaling factors, 
        cosmogenic production rates and denudation rates. Earth Surf. Process. Landforms 44, 2349–2365.
    Martin, L. C. P. et al. 2017. The CREp program and the ICE-D production rate calibration database: A fully 
        parameterizable and updated online tool to compute cosmic-ray exposure ages. Quat. Geochron., 38, 25-49.
    
    """

    from .ext import scaling as _cfuncs

    _name = "CosmogenicProductionScaler"

    _unit_agnostic = False

    _info = {
        "topographic__elevation": {
            "dtype": float,
            "intent": "in",
            "optional": True,
            "units": "m",
            "mapping": "node",
            "doc": "Topographic elevation at location",
        },  
    }
    
    def __init__(
        self,
        grid,
        location_parameter={"extract_latitude_long_from_grid": False,
                            "latitude": 40.0, "longitude": -105.0}, # CHECK if negative long raises error
        scaling_model={"spallation":"Lal_stone_2000",
                       "muon":"Braucher_2011"},
        scaling_filename={"stone_scaling_coefficient": "config/Stone_scaling_coefficient.csv"},
        cosmogenic_parameter={"e_folding_length_in_air_muon_slow": 2600.0,
                              "e_folding_length_in_air_muon_fast": 5100.0},
        atmospheric_model="ERA40",
        atmospheric_model_filename={"ERA40_lat_long_mean_pressure_hPa":"config/ERA40_lat_long_mean_pressure_hPa.csv",
                                   "ERA40_lat_long_mean_temperature_K":"config/ERA40_lat_long_mean_temperature_K.csv",
                                   "ERA40_lifton_adiabatic_dT_dz_coefficient":"config/ERA40_lifton_adiabatic_dT_dz_coefficient.csv"},
        path_prefix={"config": "", "data": ""},
        verbose=False,
        performance=True, # if set True, precomputes scaling factors for a range of elevations spanned by one meter and then affect scaling factor if floor( topographic__elevation in the range of elevations REFORMULATE
        max_elevation=9000 # used for precomputation (linked to performance = True
        ):
        """Initializes the CosmogenicProductionCorrector model.
       
        Parameters
        ----------
        grid : ModelGrid
            A Landlab Modelgrid object. A field "topographic__elevation" should be defined at nodes else
            component add this field with zero values and scaling will be done at sea-level.
            if the grid is an instance of RasterModelGrid, the component can use the average latitude/longitude
            (not implemented for other types of ModelGrids)
        location_parameter : dictionary of str, optional
            Dictionary yielding the location/coordinates of the center of the grid
            Format: {"extract_latitude_long_from_grid": False, "latitude": 40.0, "longitude": -105.0}
            - extract_latitude_long_from_grid: When True, indicate that latitude and longitude can be
                extracted from grid. In that case, average values from the grid replace the latitude
                and longitude indicated in the location_parameter.
                Note that the grid must be a RasterModelGrid (other grids not implemented)
            - latitude: average latitude of the grid in decimal degrees Negative values for southern latitudes
            - longitude: average longitude of the grid in decimal degress. Negative values for western
            longitudes
        scaling_model : dictionary of str, optional
            Format: scaling_model={"spallation":"Lal_stone_2000",
                       "muon":"Braucher_2011"}
            Models used to scale productions as a function of latitude and elevation. 
            For spallation, only "Lal_stone_2000" and "None" accepted.
            For muons, only "Braucher_2011" and "None" accepted.
            When scaling model is set to "None", the componenent creates an array 
            (or 2 arrays, for muons) of scaling factors = 1
        scaling_filename : dictionary of str, optional
            Filenames of the csv files containing the values of the scaling model.
            Stone coefficients are without units
            The coefficients should be yielded by a file formatted this way:
            Latitude, coeff a, coeff b, coeff c, coeff d, coeff e, M (p. 23754, Stone, 2000)
            Format: {"stone_scaling_coefficient": "config/Stone_scaling_coefficient.csv"}
        cosmogenic_parameter : str, optional
            Name of the yaml file containing the dictionary of cosmogenic parameter (see BedrockCosmogenicProducer)
            For this component, the file should contain:
            - e_folding_length_in_air_muon_slow, float > 0 in [kg/m^2]
                E-folding length in atmosphere for the production pathway by muon capture by low-energy muons (slow muons) 
                260 g/cm^2 (Braucher et al., 2011), p.8.
                Note that there might be a bias introduced by the assumption of a single e-folding length 
                common to all nuclides (see (Balco et al., 2017), p. 168). 
            - e_folding_length_in_air_muon_fast, float > 0 in [kg/m^2]
                E-folding length in atmosphere for the production pathway by muon interactions by high-energy muons (fast muons) 
                510 g/cm^2 (Braucher et al., 2011), p.8       
        atmospheric_model : str, optional
            Model used to represent the atmosphere. Only "ERA40" accepted
        atmospheric_model_filename : dictionary of str, optional
            Containing the names of the 3 files having latitudinal/longitudinal mean pressures, 
            mean temperatures and lapse rate polynomial coefficients
            Formatted this way:
            {"ERA40_lat_long_mean_pressure_hPa":"config/ERA40_lat_long_mean_pressure_hPa.csv",
               "ERA40_lat_long_mean_temperature_K":"config/ERA40_lat_long_mean_temperature_K.csv",
               "ERA40_lifton_adiabatic_dT_dz_coefficient":"config/ERA40_lifton_adiabatic_dT_dz_coefficient.csv"}
            ERA40_lat_long_mean_pressure_hPa file should be a csv file, with latitudes in the 1st column, 
                longitudes in the 1st row (in decimal degrees, southern latitudes being negative
                and western longitudes being > 180°) and with sea-level mean pressure values in hPa
                remaining cells
            ERA40_lat_long_mean_temperature_K file should be a csv file, with latitudes in the 1st column, 
                longitudes in the 1st row (in decimal degrees, southern latitudes being negative
                and western longitudes being > 180° up to 360°) and with sea-level mean temperatures in K
                remaining cells
                This file should have the same latitudes and longitudes as the pressure file.
            ERA40_lifton_adiabatic_dT_dz_coefficient file should be a csv file containing the polynomial coefficients, 
            formatted this way:
            a_0, a_1, a_2, a_3, a_4, a_5, a_6 
        path_prefix: dict of str, optional
            prefix to add to the filenames given as arguments of this method
            Format: {"config": "config/", "data": "data/"} {"config": "../../../config"}
        verbose: Boolean, optional
            If set to True, information messages are displayed
            
        """
        super(CosmogenicProductionScaler, self).__init__(grid)
        self.initialize_output_fields()
        self._init_parameter = {"location_parameter": copy.copy(location_parameter),
            "scaling_model": copy.copy(scaling_model),
            "scaling_filename": copy.copy(scaling_filename),
            "cosmogenic_parameter": copy.copy(cosmogenic_parameter),
            "atmospheric_model": copy.copy(atmospheric_model),
            "atmospheric_model_filename": copy.copy(atmospheric_model_filename),
            "path_prefix": copy.copy(path_prefix),
            "verbose": verbose,
            "performance": performance,
            "max_elevation": max_elevation}
        self._verbose = bool(verbose)
   
    # Property getters, setters and deleters
    @property
    def atmospheric_model(self):
        """Get the atmospheric model name of the component
        """
        return self._atmospheric_model
    
    def _set_atmospheric_model(self):
        """ Set the atmospheric model of the component          
        """
        v = self._init_parameter["atmospheric_model"]
        self._check_var_type(v, str)
        if v not in ["ERA40"]:
            raise ValueError("Accepted atmospheric models: ERA40 only")
        self._atmospheric_model = v
        
    @property
    def atmospheric_pressure_hPa(self):
        """Get the atmospheric pressure for the nodes fo the grid of the component
        """
        return self._personal_grid_field_dict["atmospheric_pressure_hPa"]
    
    def _set_atmospheric_pressure_hPa(self):
        """ Set the atmospheric pressure at each nodes of the grid
        using the Standard atmosphere equation Eq. (1) in (Stone, 2000)
        and the sea-level pressure and temperature, with elevation at locations.         
        """
        
        m = self.personal_grid_field_dict
        z = m["topographic__elevation"]
        v = self.sea_level_atmospheric_parameter_interp; P = v["mean_pressure_hPa"]
        T = v["mean_temperature_K"]; dT_dz = v["adiabatic_dT_dz_K_m"]
        m["atmospheric_pressure_hPa"] = P * np.exp(-0.03417 / dT_dz * (np.log(T) - np.log(T - dT_dz * z))) 
    
    @property
    def cosmogenic_parameter(self):
        """Get the dictionary of cosmogenic paramaters (float values)
        """
        return self._cosmogenic_parameter
    
    def _set_cosmogenic_parameter(self):
        """ Set the dictionary of cosmogenic parameters
        """
        # Initialize the property dictionaries and check the (default) values for the cosmogenic parameters
        # with converting all values to float
        w = self._init_parameter["cosmogenic_parameter"]
        s = self.scaling_model
        if "muon" in s.keys() and s["muon"] == "Braucher_2001":
            self._check_key_in_dictionary(w, ["e_folding_length_in_air_muon_slow",
                                                     "e_folding_length_in_air_muon_fast"])
        self._cosmogenic_parameter = w #dict([key, float(val)] for key, val in w.items())
    
    @property
    def cosmogenic_scaling_factor(self):
        """ Get the dictionary of cosmogenic scaling factors/correctors  
        Format: {"spallation": ndarray(float), 
            "muon_slow": ndarray(float), 
            "muon_fast": ndarray(float)}
        """ 
        m = self.personal_grid_field_dict
        s = "cosmogenic_scaling_factor_"
        if s + "muon_slow" not in m.keys() and s + "muon_fast" not in m.keys():
            return {"spallation": m[s + "spallation"]}
        if s + "muon_slow" in m.keys() and s + "muon_fast" in m.keys():
            return {"spallation": m[s + "spallation"],
               "muon_slow": m[s + "muon_slow"],                    
               "muon_fast": m[s + "muon_fast"]}
           
    def _set_cosmogenic_scaling_factor(self):
        """ Calculate the cosmogenic scaling factors/correctors, which are used to scale the cosmogenic
        production rate to the local latitudes and elevations of the grid of the component
        (for each node of the grid). Factors are 
        calculated for 3 cosmogenic pathways: spallation by high energy neutrons, slow muons, fast muons
                
        Set the scaling correctors as key,values in the personal_grid_field_dict dictionary of the component  
        with this format:
        {"cosmogenic_scaling_factor_spallation": ..., "cosmogenic_scaling_factor_muon_slow": ...
            "cosmogenic_scaling_factor_muon_fast": ...} if spallation and muon scaling models are defined
        or
        only {"cosmogenic_scaling_factor_spallation": ...} if only spallation is defined
        """
        """scm = self.scaling_model
        m = self.personal_grid_field_dict
        P = m["atmospheric_pressure_hPa"]
                    
        if scm["spallation"] == "Lal_stone_2000":
            s = self.stone_scaling_coefficient_interp;          
            # Determine scaling for spallation at each node of the grid
            
            if "muon" in scm.keys() and scm["muon"] == "Braucher_2011":
                l = self.cosmogenic_parameter            
                m["cosmogenic_scaling_factor_spallation"], m["cosmogenic_scaling_factor_muon_fast"],
                m["cosmogenic_scaling_factor_muon_slow"] = \
                    self._cfuncs._get_cosmogenic_scaling_factor_lal_stone_2000_braucher_2011(s, P, 
                        l["e_folding_length_in_air_muon_fast"], l["e_folding_length_in_air_muon_slow"]) 
            else:
                m["cosmogenic_scaling_factor_spallation"] = \
                    self._cfuncs._get_cosmogenic_scaling_factor_lal_stone_2000(s, P)
            # gain in performance using cython here is not extraordinary (1/3)
         
        Parameters
        ----------
        z: ndarray(float)
            Elevations 
        """
        scm = self.scaling_model
        m = self.personal_grid_field_dict
        z = m["topographic__elevation"]
        v = self.sea_level_atmospheric_parameter_interp; 
        P_mean = v["mean_pressure_hPa"]
        T_mean = v["mean_temperature_K"]; dT_dz_mean = v["adiabatic_dT_dz_K_m"]
        nodes_n = len(z)
        
        if self._init_parameter["performance"] == True: # We catch the pre-computed values for a range of elevations (spanned by 1-m)
            if scm["spallation"] == "Lal_stone_2000":
                if "muon" in scm.keys() and scm["muon"] == "Braucher_2011":
                    self._cfuncs._set_cosmogenic_scaling_factor_lal_stone_2000_braucher_2011_from_mem(
                        z,
                        m["cosmogenic_scaling_factor_spallation"], m["cosmogenic_scaling_factor_muon_fast"],
                        m["cosmogenic_scaling_factor_muon_slow"],  m["atmospheric_pressure_hPa"],
                        m["mem_cosmogenic_scaling_factor_spallation"], m["mem_cosmogenic_scaling_factor_muon_fast"],
                        m["mem_cosmogenic_scaling_factor_muon_slow"],  m["mem_atmospheric_pressure_hPa"])
                else:
                    self._cfuncs._set_cosmogenic_scaling_factor_lal_stone_2000_from_mem(
                        z,
                        m["cosmogenic_scaling_factor_spallation"], m["atmospheric_pressure_hPa"],
                        m["mem_cosmogenic_scaling_factor_spallation"], m["mem_atmospheric_pressure_hPa"])
        else:
            if scm["spallation"] == "Lal_stone_2000":
                s = self.stone_scaling_coefficient_interp;          
                # Determine scaling for spallation at each node of the grid
                
                if "muon" in scm.keys() and scm["muon"] == "Braucher_2011":
                    l = self.cosmogenic_parameter            
                    (m["cosmogenic_scaling_factor_spallation"], m["cosmogenic_scaling_factor_muon_fast"],
                    m["cosmogenic_scaling_factor_muon_slow"],  m["atmospheric_pressure_hPa"]) = \
                        self._cfuncs._get_cosmogenic_scaling_factor_lal_stone_2000_braucher_2011(s, z, 
                            P_mean, T_mean,  dT_dz_mean,
                            l["e_folding_length_in_air_muon_fast"], l["e_folding_length_in_air_muon_slow"],
                            threading = 1) 
                else:
                    (m["cosmogenic_scaling_factor_spallation"], m["atmospheric_pressure_hPa"]) = \
                        self._cfuncs._get_cosmogenic_scaling_factor_lal_stone_2000(s, z, 
                            P_mean, T_mean,  dT_dz_mean, threading = 1)
                # gain in performance using cython here is not extraordinary (1/3)
            
    def precompute_mem_cosmogenic_scaling_factors(self):
        """ Description to do TO DO !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        """
        scm = self.scaling_model
        m = self.personal_grid_field_dict
        v = self.sea_level_atmospheric_parameter_interp; 
        P_mean = v["mean_pressure_hPa"]
        T_mean = v["mean_temperature_K"]; dT_dz_mean = v["adiabatic_dT_dz_K_m"]
        
        mem_n = self._init_parameter["max_elevation"]
        z = m["mem_z"] = np.array([float(i) for i in range(mem_n)])
        
        # VERY Dirty !!! 
        # We take the mean latitude and the rest of calculations is cumbersome 
        # (retaken from the stone scaling coefficient interp method because we
        # chose a cell by cell approach for latitude and stone interpolation coeff, 
        # which is stupid for performance
        latitude = np.abs(self._location_parameter["latitude"])
        latitude_array = np.full(mem_n, latitude)   
        coeff = self.stone_scaling_coefficient
        stone_scaling_coefficient_interp = np.array([np.interp(latitude_array, coeff[0,:], coeff[i,:])
                                                           .tolist() for i in range(1, np.shape(coeff)[0])])
                                                         
        s = stone_scaling_coefficient_interp
                
        m["mem_cosmogenic_scaling_factor_spallation"] = np.empty(mem_n) 
        m["mem_cosmogenic_scaling_factor_muon_fast"] = np.empty(mem_n) 
        m["mem_cosmogenic_scaling_factor_muon_slow"] = np.empty(mem_n) 
        m["mem_atmospheric_pressure_hPa"] = np.empty(mem_n) 
        if scm["spallation"] == "Lal_stone_2000":
            s = stone_scaling_coefficient_interp          
            # Determine scaling for spallation at each node of the grid
            if "muon" in scm.keys() and scm["muon"] == "Braucher_2011":
                l = self.cosmogenic_parameter       
                
                (m["mem_cosmogenic_scaling_factor_spallation"],
                    m["mem_cosmogenic_scaling_factor_muon_fast"],
                    m["mem_cosmogenic_scaling_factor_muon_slow"],
                    m["mem_atmospheric_pressure_hPa"]) = \
                    self._cfuncs._get_cosmogenic_scaling_factor_lal_stone_2000_braucher_2011(s, z, 
                        P_mean, T_mean,  dT_dz_mean,
                        l["e_folding_length_in_air_muon_fast"], l["e_folding_length_in_air_muon_slow"],
                        threading = 0)
            else:
                (m["mem_cosmogenic_scaling_factor_spallation"], m["mem_atmospheric_pressure_hPa"]) = \
                    self._cfuncs._get_cosmogenic_scaling_factor_lal_stone_2000(s, z, 
                        P_mean, T_mean,  dT_dz_mean, threading = 0)
            # gain in performance using cython here is not extraordinary (1/3)
    
    @property
    def ERA40_atmospheric_model_parameter(self):
        """Get the dictionary of the parameters of the ERA40 atmospheric model
        """
        return self._ERA40_atmospheric_model_parameter
    
    def _set_ERA40_atmospheric_model_parameter(self):
        """ 
        Set the dictionary of parameters for the ERA40 atmospheric model, formatted this way
        {"latitude": ndarray,
         "longitude": ndarray,
         "sea_level_mean_pressure_hPa": ndarray(float)(x=lat, y=long, z=pressure),
         "sea_level_mean_temperature_K": ndarray(float)(x=lat, y=long, z=temperature),
         "lifton_adiabatic_dT_dz_coefficient": ndarray(float)}
        
        where 
        - latitude and longitude are the lat/long references extracted from the pressure file,
        - sea_level_mean_pressure_hPa: the sea-level mean pressure dataset, in hPa
        - sea_level_mean_temperature_K: the sea-level mean temperature dataset, in K
        - lifton_adiabatic_dT_dz_coefficient: contains the polynomial coefficients
            of the adiabatic (temperature) lapse rate (dT/dz) fit to COSPAR CIRA-86 <10 km altitude - K/m, (CHECK where this come from!!) 
            by Lifton for ERA40 atmospheric model (Lifton et al., 2014, supplementary)
        """
        v = copy.copy(self._init_parameter["atmospheric_model_filename"])
        self._check_var_type(v, dict)
        self._check_key_in_dictionary(
            v, ["ERA40_lat_long_mean_pressure_hPa", 
                "ERA40_lat_long_mean_temperature_K",
                "ERA40_lifton_adiabatic_dT_dz_coefficient"])
        for vk in v.keys():
            v[vk] = self.path_prefix["config"] + v[vk]
        
        m = {}
        a = np.genfromtxt(v["ERA40_lat_long_mean_pressure_hPa"], delimiter=',')
        m["latitude"] = a[1:, 0]
        m["longitude"] = a[0, 1:]
        m["sea_level_mean_pressure_hPa"] = a[1:, 1:] 
        m["sea_level_mean_temperature_K"] = np.genfromtxt(v["ERA40_lat_long_mean_temperature_K"], delimiter=',')[1:, 1:] 
        m["lifton_adiabatic_dT_dz_coefficient"] = np.genfromtxt(v["ERA40_lifton_adiabatic_dT_dz_coefficient"], delimiter=',')
        self._ERA40_atmospheric_model_parameter = m
    
    @property
    def location_parameter(self):
        """Get the location where the atmospheric model is interpolated
        """
        return self._location_parameter

    def _set_location_parameter(self):
        """ Set the location where the atmospheric model is interpolated
        """
        l = self._init_parameter["location_parameter"]
        self._check_var_type(l, dict)
        g = self.grid
        if "extract_latitude_long_from_grid" in l.keys() and l["extract_latitude_long_from_grid"] == True \
            and isinstance(g, RasterModelGrid):
            l["latitude"] = g.y_of_node[floor((g.number_of_cell_columns + 2)/2)]
            l["longitude"] = g.x_of_node[floor((g.number_of_cell_rows + 2)/2)]
        else:    
            self._check_key_in_dictionary(l, ["latitude", "longitude"])
        self._location_parameter = l
    
    @property
    def path_prefix(self):
        """Get the dictionary of prefixes to append to paths of files
        """
        return self._path_prefix
    
    def _set_path_prefix(self):
        """ Set the dictionary of prefixes to append to paths of files
        """
        l = self._init_parameter["path_prefix"]
        self._check_var_type(l, dict)
        self._check_key_in_dictionary(l, ["config", "data"])
        self._path_prefix = l
    
    @property
    def personal_grid_field_dict(self):
        """Get the dictionary of personal fields associated with the grid of the component
        """
        return self._personal_grid_field_dict
    
    def _set_personal_grid_field_dict(self):
        """Set the dictionary of personal fields associated with the grid of the component at nodes
        The personal grid field values are calculated from the public grid fields which 
        were possibly updated during the former run one step 
        These values are only used in this component and might be destroyed at the end of each run (for memory usage concern)
        
        NOTA BENE: not sure it might be usedful to create an array of latitudes in this component
        since we implemented the interpolation from the average of latitudes
        
        Format: {"topographic__elevation": ndarray(float) in [m],
            "latitude": ndarray(float) in [°],
            "atmospheric_pressure_hPa": ndarray(float) in [hPa],
            "scaling_spallation": ndarray(float) in [-]}
        
        - topographic__elevation: field of the grid of the component, if doesn't exist, create the field and
            set it at zeros
        - latitude: presently, all latitudes are the same and equal = latitude of the center of the grid
        - atmospheric_pressure_hPa: determined using the hydrostatic equation of the standard atmospheric model, described in (N.O.A.A. et al., 1976)
            and formulated as equation (1) in (Stone, 2000) , p. 23,753.
        - cosmogenic_scaling_factor_spallation: cosmogenic scaling corrector/factor for production rates associated with
            spallation by high energy neutrons.
            determined using the equation (2) of (Stone, 2000), p. 23,754.
        - cosmogenic_scaling_factor_muon_slow:
        - cosmogenic_scaling_factor_muon_fast:
        """
        
        m = {}; g = self.grid
        # Set unset grid fields to default values
        for my_field in ["topographic__elevation", "latitude", "longitude"]:        
            if my_field not in g.at_node:
                if my_field in ["latitude", "longitude"]: 
                    val = self.location_parameter[my_field] * np.ones(g.number_of_nodes)
                else:
                    val = np.zeros(g.number_of_nodes) 
                m[my_field] = g.add_field(my_field, val, at='node', clobber=True)
            else:
                m[my_field] = g.at_node[my_field]
        
        self._personal_grid_field_dict = m
        
    @property
    def scaling_model(self):
        """Get the scaling model name of the component
        """
        return self._scaling_model
    
    def _set_scaling_model(self):
        """ Set the scaling model of the component
        """
        v = self._init_parameter["scaling_model"]
        self._check_var_type(v, dict)
        if "spallation" not in v.keys():
            raise ValueError("Scaling model for 'spallation' must be defined")
        if v["spallation"] not in ["Lal_stone_2000", "None"]:
            raise ValueError("Accepted scaling models for spallation: 'Lal_stone_2000' and 'None' only")
        if "muon" in v.keys() and v["muon"] not in ["Braucher_2011", "None"]:
            raise ValueError("Accepted scaling models for muons: 'Braucher_2011' and 'None' only")
        self._scaling_model = v
        
    @property
    def sea_level_atmospheric_parameter_interp(self):
        """Get the sea-level values of the atmospheric model interpolated at a location
        """
        return self._sea_level_atmospheric_parameter_interp
    
    def _set_sea_level_atmospheric_parameter_interp(self):
        """ Creates a dictionary of local sea-level atmospheric parameters
        by interpolating the sea-level parameters at a location, 
        using the sea-level atmospheric model of the component
        Implemented only for ERA40 atmospheric model.
        
        Format:
        {"mean_pressure_hPa": float in [hPa],
         "mean_temperature_K": float in [K],
         "adiabatic_dT_dz_K_m": float in [K/m]}
        """
        l = self.location_parameter
        y = l["latitude"]
        x = l["longitude"] if floor(l["longitude"]/360) == 0 else l["longitude"] + 360
        m = {}
        if self.atmospheric_model == "ERA40":
            s = self.ERA40_atmospheric_model_parameter
            X = s["longitude"]; Y = s["latitude"]; Z = s["sea_level_mean_pressure_hPa"]
            m["mean_pressure_hPa"] = float(sci.interpolate.interp2d(X, Y, Z).__call__(x, y))
            Z = s["sea_level_mean_temperature_K"]
            m["mean_temperature_K"] = float(sci.interpolate.interp2d(X, Y, Z).__call__(x, y)) 
            
            a = s["lifton_adiabatic_dT_dz_coefficient"]
            b = a[0]
            for i in range(1,7):
                b += a[i] * y ** i
            m["adiabatic_dT_dz_K_m"] = -b
            
            self._sea_level_atmospheric_parameter_interp = m
        
    @property
    def stone_scaling_coefficient_interp(self):
        """Get the ndarray of the scaling equation coefficients interpolated 
        from the Lal-Stone coefficients (float values)
        """
        return self._stone_scaling_coefficient_interp
    
    def _set_stone_scaling_coefficient_interp(self):
        """ Set the ndarray of scaling coefficient interpolated for an ndarray of latitudes 
        from the Lal-Stone coefficients
        
        Format: ndarray(float)
            TO FILL by an example
        """
        latitude_array = self.personal_grid_field_dict["latitude"]                
        if abs(latitude_array.max()) > 90:
            raise ValueError("The latitude should be in the -90° to 90° range")
        
        latitude_array = np.abs(latitude_array)
        m = self.stone_scaling_coefficient
        self._stone_scaling_coefficient_interp = np.array([np.interp(latitude_array, m[0,:], m[i,:])
                                                           .tolist() for i in range(1, np.shape(m)[0])]) # CHECK performance
        # https://stackoverflow.com/questions/35215161/most-efficient-way-to-map-function-over-numpy-array
        # or use of map function?       
   
    @property
    def stone_scaling_coefficient(self):
        """Get the ndarray of the scaling equation coefficients of Lal-Stone (float values)
        """
        return self._stone_scaling_coefficient
    
    def _set_stone_scaling_coefficient(self):
        """Set the scaling coefficients of Lal-Stone (Stone, 2000).
        The coefficients should be yielded by a file formatted by column this way:
        Latitude (in decimal degrees, coeff a, coeff b, coeff c, coeff d, coeff e, M (p. 23754, Stone, 2000)
        and the file should include latitude from 0 to 90°.
        The result is an array formatted by line this way
        Latitude (in decimal degrees, coeff a, coeff b, coeff c, coeff d, coeff e, M
        """
        v = copy.copy(self._init_parameter["scaling_filename"])
        self._check_var_type(v, dict)
        self._check_key_in_dictionary(v, ["stone_scaling_coefficient"])
        for vk in v.keys():
            v[vk] = self.path_prefix["config"] + v[vk]
        self._stone_scaling_coefficient = \
            np.genfromtxt(v["stone_scaling_coefficient"], delimiter=',').transpose()
            
    # utils
    def _check_var_type(self, var, type1):
        """Typically used to check the type of the arguments yielded to functions or methods.
        check if the type of the variable corresponds to the one expected

        Parameters
        ----------
        var : Any type
            Variable to check type
        type1 : type
           Type which is expected for the variable var
        """
        if not isinstance(var, type1):
            raise ValueError("The variable doesn't have the expected type (Check documentation)")
                
    def _check_var_list_type(self, var_list, type_list):
        """Typically used to check the type of the arguments yielded to functions or methods.
        For each variable in the list, check if its type corresponds to the one expected

        Parameters
        ----------
        var_list : List
            List of variables (of different types) to check type
        type_list : List of type
            List of the types which are expected for each variable of var_list
        """
        if not isinstance(var_list, list) or not isinstance(type_list, list) :
            raise ValueError("var_list must be a list of variables and expected_type_list a list of types")
        
        if len(var_list) != len(type_list):
            raise ValueError("The list of variables must have the same length as the list of types")    
        for i in range(0, len(var_list)):
            if not isinstance(var_list[i], type_list[i]):
                raise ValueError("At least one of the variables doesn't have the expected type (Check documentation)")   
        
    def _get_dictionary_from_file(self, filename):
        """DEPRECATED
        Get a dictionary from a csv file where keys are stored in the 1st column
        and values are stored in the other columns. These values are either unique (1 column)
        or a list (several columns)
        Note that files should at least contain one couple of key, value

        Parameters
        ----------
        filename : str
            Directory + filename where the data are stored
        Returns
        -------
        """
        self._check_var_type(filename, str)
        my_dictionary = {}
        with open(filename, mode='r') as file_resource:
            my_reader = csv.reader(file_resource)
            for row in my_reader:
                if bool(row) and row[0] != "#": #test if row not empty and doesn't contain comment mark
                    if len(row) == 1:
                        break
                    elif len(row) == 2:
                        my_dictionary[row[0]] = row[1]
                    else:
                        myList = []
                        for i in range(1, len(row)):
                            myList.append(row[i])
                        my_dictionary[row[0]] = myList
        return my_dictionary

    def _check_key_in_dictionary(self, myDict, keyList):
        """Check whether the list of keys are available in the dictionary

        Parameters
        ----------
        myDict : dictionary
            Dictionary to check keys
        keyList : list of str
            List of keys to check within the dictionary (List of Strings)
        """
        self._check_var_list_type([myDict, keyList], [dict, list]);
        for my_key in keyList:
            if my_key not in myDict.keys():
                raise ValueError("The dictionary (or input file) does not contain all necessary keys (check documentation)")
        
    def prepare_first_run(self):
        """Prepare the object to undergo its first run.
        Initializes attributes and grid fields to default values using the configuration indicated within
        the _init_parameter attribute if necessary.
        """
        self._set_path_prefix()       
        # Check if scaling model and atmospheric model are implemented
        self._set_scaling_model()
        self._set_atmospheric_model()
                
        # Check the input dictionaries and atmospheric/scaling parameters
        self._set_location_parameter()
        self._set_ERA40_atmospheric_model_parameter()      
        self._set_cosmogenic_parameter()                    
                                          
        # Set up the personal node field grid, with topography
        self._set_personal_grid_field_dict()
        
        if self._scaling_model["spallation"] == "Lal_stone_2000":
            self._set_stone_scaling_coefficient()
             # Interpolate the scaling coefficients (Stone only)   
            self._set_stone_scaling_coefficient_interp()
        
        # prepare array of scaling factors and atmospheric pressure for the grid nodes
        scm = self.scaling_model
        m = self.personal_grid_field_dict
        
        if scm["spallation"] == "None":
            m["cosmogenic_scaling_factor_spallation"] = np.ones(self.grid.number_of_nodes)
        else:
            m["cosmogenic_scaling_factor_spallation"] = np.empty(self.grid.number_of_nodes)
        if "muon" not in scm.keys():
            pass
        elif scm["muon"] == "None":
            m["cosmogenic_scaling_factor_muon_slow"] = np.ones(self.grid.number_of_nodes)
            m["cosmogenic_scaling_factor_muon_fast"] = np.ones(self.grid.number_of_nodes)
        else:
            m["cosmogenic_scaling_factor_muon_slow"] = np.empty(self.grid.number_of_nodes)
            m["cosmogenic_scaling_factor_muon_fast"] = np.empty(self.grid.number_of_nodes)        
        m["atmospheric_pressure_hPa"] = np.empty(self.grid.number_of_nodes)
        
        # Precomputation of mean pressure, temperature and adiabatic dT_dz
        # and of an array of scaling factors for each range of 1-m elevation
        # Difference in scaling factors is lower than 0.1% by 1-m elevation for spallation
        # and even lower for muons
        ###############################################
        
        # Interpolate atmospheric model (ERA40 only) using topographic elevation (use of property setters) 
        self._set_sea_level_atmospheric_parameter_interp()
        
        # Computation of scaling factors for each meter of elevation
        if self._init_parameter["performance"] == True:
            self.precompute_mem_cosmogenic_scaling_factors()
                
        if self._verbose:
            print(landlab.core.messages.format_message("prepare_first_run() executed"))

    def run_one_step(self, dt):
        """Advance BedrockCosmogenicProducer component by one time step of size dt.

        Parameters
        ----------
        dt : float
            The imposed timestep, (time) in [y]
        """
        dt = float(dt)
        
        
        
        # Determine pressure at each node of the grid
        #self._set_atmospheric_pressure_hPa()
       
        # Calculate the scaling factors
        self._set_cosmogenic_scaling_factor()
        if self._verbose:
            print(landlab.core.messages.format_message("run_one_step(dt) executed"))