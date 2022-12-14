"""Calculates concentrations of cosmogenic nuclides in bedrock.
We suppose that density, presence of mineral target are steady over the simulation
Written by Sebastien Lenard, 2022, May
"""

import numpy as np # mathematical operations and ndarrays
from landlab import Component, ModelGrid # class inheritance and type of the main attribute of the class
import landlab.core, landlab.core.messages # to get the yaml file and format messages
import csv # to get the default values of parameters TO CHECK IF NECESSARY
import copy # to solve the problem of modifying by componenent of optional arguments 
    #each time it is called (caused by the concatenation with path_prefix in code)

from landlab.components import CosmogenicProductionScaler

class CosmogenicProducerInBedrock(Component):

    """Calculates concentrations of cosmogenic nuclides in bedrock.
    
    Landlab component that calculates concentrations of cosmogenic nuclides using
    the steady state formula of Lal, 1991.

    Faire lagrangien pour décapage

    (avec calcul du taux de production à chaque étape)

    This component can TO FILL
    EXPLAIN the models of production using Lal, 1991; Braucher et al., 2011, Balco et al., 2017; Gosse and Phillips, 2000
        spallation (since Lal, 1991)
        muons, more complex. most of the cases, small component of the total (discussion in (Balco et al., 2017)) 
        except for burial dating and depth profiles
        and also in rapidly denuding environments (e > 5 mm/y, in(Balco et al., 2017)
        we only implemented the model of (Braucher et al., 2011). A even simpler model (Braucher et al., 2013) exists 
        but not commonly used. More complex models are discussed in (Balco et al., 2017) and not implemented here.
        We implemented the muon component, but it's important to remember that this component needs much more time to 
        get steady state than the spallation component
        
        The exponential model of (Braucher et al., 2011), which is similar to the exponential model for spallation
        does not represent the reality of the physical processes (notably the reinforcement
        and collimation of the fast muon flux through depth) but when it calibrated well with natural data, and it's much more computationally efficient
        this is why it is used here.
        We should undeline, that in case of deep-seated landslides, concentrations are mainly due to muons, and since landslides broke the steady state
        the concentrations computed might be false (even though they will be globally negligible compared to the rocks not landslided.
    
    the basic hypothesis of this producer is that all cosmogenic pathways have an exponential depth profile in bedrock,
    spallation but also muons.
    
    We have to split the three pathways and for each of them integrate production and decay through a time step.
    
    either we start from zero concentrations over the profile.
    Or we start supposing that cosmogenic nuclide budget over the profile is at steady-state 
    (only possible for unstable nuclides) since long irradiation. In that case, time to reach steady state and
    steady-state concentrations can be calculated. This calculation depends on each pathway (and denudation rate)
    
    The componenet doesn't accept input 10Be concentrations right now.
    We should start the simulation with initial concentrations at surface = 0 and wait a certain equilibrium?
    
    Not that the theorical local production rate (scaling factor x slhl production rate) 
        is not yielded by the component and the calculation is done on the fly
    NB: steady-state cosmogenic production : when secular equilibrium is reached, loss by decay = production by cosmic rays
    NB : steady state concentrations are calculated supposing elevation remains steady
    NB : before the first run, if steady state is parametered as initial condition, 
    the grid should have all input fields corresponding to a long-term, steady-state setting
    (this includes cover and denudation rate)
    
    Implemented.
    Concentrations are calculated at the nodes of the grid
    Concentrations are concentrations at top of bedrock
    Compute a concentration at bedrock surface given a production rate and an inherited concentration
    Partial shielding from 1 cover only (whatever cover sediment or ice or even bedrock)
    Sea-level high-latitude (no scaling)
    Be10 only
    spallation by high energy neutrons only
    cosmogenic production rate doesn't vary over time (= probably work on short timescales)
    homogeneous quartzic lithology
    Model of (Braucher et al., 2011) for muon production
    steady state initial concentrations
    We suppose that concentration only result from denudation (which is probably true for bedrock but not for sediment)
   
    Not implemented.
    Denudation rates equalling zero.
    inherited concentrations
    Muons
    No combined cover sediment + ice
    Topographic shielding
    Latitudinal + altitudinal scaling of production rate (another component should do that)
    dipole variations impact on production rate
    don't exclude part of the watershed which don't contain mineral target for the production of nuclides
    AMS standards corrections of concentrations (Cosmogenic_AMS_standards.csv from Balco calculator)
    
    QUESTIONS: do we have to put density to each node, while most oftenly, density will be the same everywhere?
    
    Properties
    ----------
    
    Examples
    --------
    >>> from landlab.components.uniform_precip import PrecipitationDistribution
    >>> import numpy as np
    >>> np.random.seed(np.arange(10))
    TO FILL

    References
    ----------
    **Required Software Citation(s) Specific to this Component**

    **Additional References**
    Gosse, J. C. and Phillips, F. M. 2001. Terrestrial in situ cosmogenic nuclides: theory and application. Quat. Sci. Rev. 20, 1475-1560
    Lal, D. 1991. Cosmic ray labeling of erosion surfaces: in situ nuclide production rates and erosion models. Earth Planet. Sci. Lett. 104, 424-439
    Lal, D., B. Peters, B. 1967. Cosmic ray produced radioactivity on the earth, in: S. flugge (Ed.), 
        Handbook of Physics, Springer, Verlag, Berlin, 551–612.
    
    Decay constants:
    Chmeleff, J. et al. 2010. Determination of the 10Be half-life by multicollector ICP-MS and liquid scintillation counting. 
        Nucl. Instr. Meth. Phys. Res. B. 268, 192–199
    Korschinek, G. et al. 2010. A new value for the 10Be half-life by heavy-ion elastic recoil detection and liquid scintillation 
        counting. Nucl. Instr. Meth. Phys. Res. B. 268, 187–191
        
    Muons:
    Balco, G. 2017. Production rate calculations for cosmic-ray-muon-produced 10Be and 26Al benchmarked against geological calibration data. 
        Quat. Geochrono. 39, 150-173
    Braucher, R. 2011. Production of cosmogenic radionuclides at great depth: A multi element approach. Earth. Planet. Sci. Lett. 309, 1-9
    Heisinger, B. et al. 2002. Production of selected cosmogenic radionuclides by muons; 2. Capture of negative muons. 
        Earth Planet. Sci. Lett. 200, 357–369
    
    Production rates:
    Borchers, B. et al. 2016. Geological calibration of spallation production rates in the CRONUS-Earth project. Quat. Geochron. 31, 188-198.
    Martin, L. C. P. et al. 2017. The CREp program and the ICE-D production rate calibration database: A fully 
        parameterizable and updated online tool to compute cosmic-ray exposure ages. Quat. Geochron., 38, 25-49.
    """

    from .ext import production_in_bedrock as _cfuncs # cython functions
    
    _name = "CosmogenicProducerInBedrock"

    _unit_agnostic = False

    _info = {
        "cosmogenic_scaling_factor": {
            "dtype": float,
            "intent": "in",
            "optional": True,
            "units": "-",
            "mapping": "node",
            "doc": "Cosmogenic scaling factor, depending on elevation and latitude at location",
        },        
        "concentration_Be10_quartz_top_bedrock_atom_g": {
            "dtype": float,
            "intent": "inout",
            "optional": True,
            "units": "atom/g",
            "mapping": "node",
            "doc": "Concentration of the cosmogenic nuclide Be10 in quartz at the bedrock surface at location.",
        },
        "density_bedrock_kg_m3": {
            "dtype": float,
            "intent": "in",
            "optional": True,
            "units": "kg/m^3",
            "mapping": "node",
            "doc": "Average density of the bedrock (the first 10 to 30 meters) at location",
        },
        "density_cover_kg_m3": {
            "dtype": float,
            "intent": "in",
            "optional": True,
            "units": "kg/m^3",
            "mapping": "node",
            "doc": "Average density of material covering bedrock",
        },
        "denudation_rate_bedrock_mm_y": {
            "dtype": float,
            "intent": "in",
            "optional": True,
            "units": "mm/y",
            "mapping": "node",
            "doc": "Denudation rate of the bedrock at location. Denudation is assumed as vertical advection of material.Should not be equal to zero",
        },
        "target_mineral_presence_bedrock": {
            "dtype": float,
            "intent": "in",
            "optional": True,
            "units": "-",
            "mapping": "node",
            "doc": "Presence in bedrock of the target mineral for the production of cosmogenic nuclide. 0: absent, 1: present",
        },
        "thickness_cover_m": {
            "dtype": float,
            "intent": "in",
            "optional": True,
            "units": "m",
            "mapping": "node",
            "doc": "Thickness of any material (ice or sediment or soil) covering bedrock",
        },
    }

    def __init__(
        self,
        grid,
        system={"nuclide":"Be10", "mineral":"quartz"},
        production_pathway=["spallation", "muon_slow", "muon_fast"],
        production_model={"muon": "Braucher_2011"},
        init_condition={"steady_state_cosmogenic_production": False},
        geo_setting="config/Geo_setting.yaml",
        geochemical_setting="config/Geochemical_setting.yaml",
        cosmogenic_parameter="config/Cosmogenic_parameter.yaml",
        path_prefix={"config": "../../../", "data": ""},
        verbose=False):
        """Initializes the BedrockCosmogenicProducer model.
        Complementary initialization should be done calling prepare_first_run()
        before the first run of the component

        Parameters
        ----------
        grid : ModelGrid
            A Landlab Modelgrid object.
        system: dict(str), optional
            Format: {"nuclide":"Be10", "mineral":"quartz"}
            - nuclide: Abbreviation of the cosmogenic nuclide which the Component makes the budget. Presently only "Be10"
            (in the future: Al26, He3, C14 ...)
            - mineral: Target mineral where the nuclide is produced. Presently only "quartz"
        production_pathway: list(string), optional
            List of the pathways included into the calculation of the cosmogenic nuclide concentration budget.
            Implemented pathways for Be10 in quartz : "spallation", "muon_slow", "muon_fast".
            Although radioactive decay is also a "production" pathway, it is not indicated here because it is taken
            automatically into account depending on the setting of the half-life in the cosmogenic_setting file
        production_model: dict(str)
            Format: {"muon": "Braucher_2011"}
            Specific models to calculate production (in particular for muons)
        init_condition= dict(str)
            Format: {"steady_state_cosmogenic_production": False}
            set initial conditions/hypotheses on the system. 
            Presently only initial  steady-state cosmogenic production is implemented. If set to true, 
            component consider that denudation rate and exposure to cosmic rays remain steady for sufficient time
            to equilibrate (radioactive) loss and production of all cosmogenic pathways. In that case, it 
            will calculate steady-state concentrations
            and will add them to the initial top bedrock concentrations (considered here as inherited) 
        geo_setting: string, optional
            Filename of the yaml file containing the dictionary of the initial geomorphological and lithological properties with mean values.
            The means are used when the properties are not defined at the local level (that is explicitly using a grid layer).
            - density_bedrock_kg_m3: float >= 0 in [kg/m^3]
                Density of bedrock, 2700 kg/m^3 for granites. Symbol: rho letter
            - denudation_rate_bedrock_mm_y, float >= 0 in [mm/y]
                Denudation rate that impacts the bedrock. Symbol: epsilon letter.
                This SHOULD be the initial, long-term denudation rate if we start the component with initial steady state. 
                Should not be equal to zero.
            - target_mineral_presence_bedrock, float between 0 and 1 in [-]
                When the mineral target of the cosmogenic reactions of the nuclide is present, bedrock_target_mineral_presence = 1, else = 0.
            - density_cover_kg_m3, float > 0 in [kg/m^3]
                Density of the cover. Cover can be anything covering (=shielding) the bedrock: sediment, soil, solid or liquid water.
            - thickness_cover_m, float > 0 in [m]
                Thickness of the cover.            
        geochemical_setting: string, optional
            Filename of the yaml file containing the dictionary of the initial geochemical properties with mean values (e.g. concentrations)
            The means are used when the properties are not defined at the local level (that is explicitly using a grid layer).
            - concentration_Be10_quartz_top_bedrock_atom_g, float >= 0 in [atom/g]
                Be10 concentration (at bedrock surface)
            CURRENTLY ONLY ZEROS are accepted.
        cosmogenic_parameter: string, optional
            Filename of the yaml file containing the dictionary of cosmogenic parameters (e.g. production rate, half-life).
             - cosmogenic_scaling_factor, float > 0 in [-]
                Mean cosmogenic scaling factor. Cosmic ray intensity varies with latitude and elevation. 
                This factor allows to scale the sea-level high-latitude production rate to the local latitude and elevation.
                For watersheds extended beyond 1° of latitude and with high relief, it is recommended to use this mean 
                only as a rough approximation and rather use values defined at the local level (grid layer "cosmogenic_scaling_factor")
            - decay_half_life_Be10_y, float > 0 in [y]
                Decay half-life of the nuclide Be10. 1.39e6 y (Chmeleff et al., 2010; Korschinek et al., 2010)
                Symbol: lambda letter
                Set to 10^10 for stable nuclides (CHECK whether it works)
            - e_folding_length_in_rock_spallation_g_cm2, float > 0 in [g/cm^2]
                Absorption mean free path in rocks for the production pathway by spallation by high energy neutrons.
                Also called attenuation length or e-folding length, symbol: Lambda letter.
                It is the "thickness of a slab 
                of rock or other mass (air, water, sediment, snow) required to attenuate the intensity 
                of the energetic cosmic-ray flux by a factor of 1/e due to scattering and absorption processes" (Gosse and Phillips, 2001).
                We assume the watershed being at sea-level and high latitude.
                Values vary from 121 to >170, depending on latitude and elevation.
                Following the discussion p. 1504 of (Gosse and Phillips, 2001), we stick to the 160 g/cm^2 mean value.
            - e_folding_length_in_rock_muon_slow_g_cm2, float > 0 in [g/cm^2]
                E-folding length in rocks for the production pathway by muon capture by low-energy muons (slow muons)
                1500 g/cm^2 (Heisinger et al., 2002)
            - e_folding_length_in_rock_muon_fast_g_cm2, float > 0 in [g/cm^2]
                E-folding length in rocks for the production pathway by muon interactions by high-energy muons (fast muons) 
                4320 g/cm^2 (Heisinger et al., 2002)
            - e_folding_length_in_air_muon_slow_g_cm2, float > 0 in [g/cm^2]
                E-folding length in atmosphere for the production pathway by muon capture by low-energy muons (slow muons) 
                260 g/cm^2 (Braucher et al., 2011), p.8.
                Note that there might be a bias introduced by the assumption of a single e-folding length 
                common to all nuclides (see (Balco et al., 2017), p. 168). 
                Used by the scaling production scaler componenent only.
            - e_folding_length_in_air_muon_fast_g_cm2, float > 0 in [g/cm^2]
                E-folding length in atmosphere for the production pathway by muon interactions by high-energy muons (fast muons) 
                510 g/cm^2 (Braucher et al., 2011), p.8
                Used by the scaling production scaler component only.           
            - slhl_prod_rate_Be10_spallation_atom_g_y, float > 0 in [atom/g/y]
                Sea-level high-latitude (SLHL) production rate through spallation pathway by high energy neutrons for Be10.
                The total Be10 production rate for all pathways is 4.11 atom/g/y +/- 0.19 etom/g/y (i.e. 5%) (1-sigma)
                which is a mean from a global compilation of production rates, 
                retrieved from CREP  22/05/24 (last pub on Ice-D 2019)
                https://crep.otelo.univ-lorraine.fr/#/production-rate 
                (Martin et al., 2017). See also discussion in (Borchers et al., 2016)
                The SLHL production rate for spallation pathway is obtained by subtracting to the total production rate
                the values of production rates from slow muon capture and fast muon interaction pathways:
                0.012±0.012 and 0.039± 0.004 atom/g/y in (Braucher et al., 2011), p.8
                which results in slhl_prod_rate_spallation_Be10_atom_g_y = 4.06 atom/g/y
            - sl_prod_rate_Be10_muon_slow_atom_g_y, float > 0 in [atom/g/y]
                Sea-level (SL) production rate through slow muon capture pathway by low energy muon for Be10
                0.012±0.012 atom/g/y in (Braucher et al., 2011), p.8
            - sl_prod_rate_Be10_muon_fast_atom_g_y, float > 0 in [atom/g/y]
                Sea-level (SL) production rate through fast muon interaction pathway by high energy muon for Be10
                0.039± 0.004 atom/g/y in (Braucher et al., 2011), p.8 
            - steady_state_time_to_reach_factor, float > 0 in [-]
                Factor for time so as to denudation reach steady-state in the equation t > factor * (lambda + mu * epsilon)
                with lambda: decay constant, mu: absorption_coefficient, and epsilon: denudation rate
            path_prefix: dict of str, optional
                prefix to add to the filenames given as arguments of this method
                Format: {"config": "config/", "data": "data/"} {"config": "../../../config"}
            verbose: boolean, optional
                if set to True, display information messages
        """
        super(CosmogenicProducerInBedrock, self).__init__(grid)

        self.initialize_output_fields() # USEFUL?  
        self._init_parameter = {"system": copy.copy(system),
            "production_pathway": copy.copy(production_pathway),
            "production_model": copy.copy(production_model),
            "init_condition": copy.copy(init_condition),
            "geo_setting": copy.copy(geo_setting),
            "geochemical_setting": copy.copy(geochemical_setting),
            "cosmogenic_parameter": copy.copy(cosmogenic_parameter),
            "path_prefix": copy.copy(path_prefix),
            "verbose": verbose}
        self._verbose = bool(verbose)
                
    # Property getters, setters, deleters
    
    @property
    def concentration_contribution_top_bedrock_previous_atom_g(self):
        """Get the dictionary of the contributions to nuclide concentration 
        at bedrock surface
        obtained from the different 
        production pathways at
        the end of the previous run.
        """
        d = dict()
        pg = self.private_grid_field_dict
        for p in self.production_pathway:
            d[p] = pg["concentration_contribution_top_bedrock_previous_from_" + p + "_atom_g"]
        return d
    
    def _set_concentration_contribution_top_bedrock_previous_atom_g(self):
        """Record the contribution to nuclide concentration
        at bedrock surface, obtained from the different 
        production pathways at
        the end of the previous run.
        """ 
        pg = self.private_grid_field_dict
        for p in self.production_pathway:
            pg["concentration_contribution_top_bedrock_previous_from_" + p + "_atom_g"] = \
                pg["concentration_contribution_top_bedrock_from_" + p + "_atom_g"]
            
    @property
    def cosmogenic_parameter(self):
        """Get the dictionary of cosmogenic paramaters (float values)
        """
        return self._cosmogenic_parameter
    
    def _set_cosmogenic_parameter(self):
        """ Set the dictionary of cosmogenic parameters
        """
        v = self.path_prefix["config"] + self._init_parameter["cosmogenic_parameter"]
        self._check_var_type(v, str)
        # Initialize the property dictionaries and check the (default) values for the cosmogenic parameters
        # with converting all values to float
        cp = landlab.core.load_params(v)
        
        pp = self.production_pathway
        pm = self.production_model
        n = self.nuclide
        m = self.mineral
        if "muon" in pm and pm["muon"] in ["Braucher_2011"]:
            if "muon_fast" in pp:
                self._check_key_in_dictionary(cp, ["e_folding_length_in_rock_muon_fast_g_cm2", 
                                                   "slhl_prod_rate_" + n + "_" + m + "_muon_fast_atom_g_y"])
            if "muon_slow" in pp:
                self._check_key_in_dictionary(cp, ["e_folding_length_in_rock_muon_slow_g_cm2", 
                                                   "slhl_prod_rate_" + n + "_" + m + "_muon_slow_atom_g_y"])
        
        if "spallation" in pp:
            self._check_key_in_dictionary(cp, ["e_folding_length_in_rock_spallation_g_cm2", 
                                               "slhl_prod_rate_" + n + "_" + m + "_spallation_atom_g_y"])
        self._check_key_in_dictionary(cp, ["decay_half_life_" + n + "_y", "steady_state_time_to_reach_factor"])
        
        self._cosmogenic_parameter = dict([key, float(val)] for key, val in cp.items())
    
    @property
    def cosmogenic_production_scaler(self):
        """Get the radioactive decay constant of the cosmogenic nuclide, in [y^-1]
        """
        return self._cosmogenic_production_scaler
    
    @cosmogenic_production_scaler.setter
    def _set_cosmogenic_production_scaler(self, cosmogenic_production_scaler):
        """ Set the cosmogenic production scaler of the component
        cosmogenic_production_scaler: landlab.CosmogenicProductionScaler
            Will give the scaling factors of the grid linked to the component.
        
        Format: CosmogenicProductionScaler
        
        Parameters
        ----------
        cosmogenic_production_scaler: landlab.component.CosmogenicProductionScaler
        """

        self._check_var_type(cosmogenic_production_scaler, CosmogenicProductionScaler)
        self._cosmogenic_production_scaler = cosmogenic_production_scaler
        
    
    @property
    def decay_constant_y_1(self):
        """Get the radioactive decay constant of the cosmogenic nuclide, in [y^-1]
        """
        return self._decay_constant_y_1
    
    def _set_decay_constant_y_1(self):
        """ Set the radioactive decay constant of the cosmogenic nuclide, in [y^-1]
        """
        half_life_y = self.cosmogenic_parameter["decay_half_life_" + self.nuclide + "_y"]
        self._check_var_type(half_life_y, float)
        # Decay constant of the cosmogenic nuclide, also called lambda (in y^-1), Zero for stable nuclides
        if half_life_y == 0:
            raise ValueError("Half-life must not be 0")
        self._decay_constant_y_1 = np.log(2) / half_life_y 
        
    @property
    def geo_setting(self):
        """Get the dictionary of geological setting parameters (float values)
        """
        return self._geo_setting
    
    def _set_geo_setting(self):
        """ Set the dictionary of geological setting parameters
        """
        v = self.path_prefix["config"] + self._init_parameter["geo_setting"]
        self._check_var_type(v, str)
        # Initialize the property dictionaries and check the presence of values for the attributes 
        # of the geomorphological/lithological setting
        # with converting all values to float
        w = landlab.core.load_params(v)
        self._check_key_in_dictionary(w, ["denudation_rate_bedrock_mm_y", "density_bedrock_kg_m3",
                        "density_cover_kg_m3", "target_mineral_presence_bedrock", "thickness_cover_m"])
        self._geo_setting = dict([key, float(val)] for key, val in w.items())
    
    @property
    def geochemical_setting(self):
        """Get the dictionary of geochemical data (float values)
        """
        return self._geochemical_setting
    
    def _set_geochemical_setting(self):
        """ Set the dictionary of geochemical data
        """
        v = self.path_prefix["config"] + self._init_parameter["geochemical_setting"] 
        self._check_var_type(v, str)
        # Initialize the property dictionaries and check the presence of geochemical data
        # with converting all values to float
        w = landlab.core.load_params(v)
        n = self.nuclide
        m = self.mineral
        self._check_key_in_dictionary(w, ["concentration_" + n + "_" + m + "_top_bedrock_atom_g"])
        if w["concentration_" + n + "_" + m + "_top_bedrock_atom_g"] != 0:
             raise ValueError("Initial concentrations should be set at 0.") # TO REMOVE when component will accept other values.
        self._geochemical_setting = dict([key, float(val)] for key, val in w.items())
    
    @property
    def init_condition(self):
        """Get the init conditions
        """
        return self._init_condition
    
    def _set_init_condition(self):
        """Set the init condition
        Type: dict(varied)
        Format: {"steady_state_cosmogenic_production": False}   
        """
        p = self._init_parameter["init_condition"]
        self._check_var_type(p, dict)
        self._check_key_in_dictionary(p, ["steady_state_cosmogenic_production"])
        self._init_condition = p
  
    @property
    def nuclide(self):
        """Get the cosmogenic nuclide
        """
        return self._nuclide
    
    def _set_nuclide(self):
        """Set the cosmogenic nuclide
        Format: "Be10"
        """
        v = self._init_parameter["system"]["nuclide"]
        self._check_var_type(v, str)
        # Check if nuclide is implemented
        if v not in ["Be10"]:
            raise ValueError("Component is implemented only for Beryllium-10 cosmogenic nuclide (Be10)")
        self._nuclide = v
        
    @property
    def mineral(self):
        """Get the target mineral
        """
        return self._mineral
    
    def _set_mineral(self):
        """Set the target mineral
        Format: "quartz"
        """
        v = self._init_parameter["system"]["mineral"]
        self._check_var_type(v, str)
        # Check if nuclide is implemented
        if v not in ["quartz"]:
            raise ValueError("Component is implemented only for Be10 in the quartz mineral")
        self._mineral = v
    
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
    def private_grid_field_dict(self):
        """Get the dictionary of private fields associated with the grid of the component
        """
        return self._private_grid_field_dict
    
    def _set_private_grid_field_dict(self):
        """Set the dictionary of private fields associated with the grid of the component
        The private grid field values are calculated from the public grid fields which 
        were possibly updated during the former run one step 
        These values are only used in this component and should be destroyed at the end of each run (for memory usage concern)
        
        TO THINK ABOUT: it might be interesting to make this property an instance of a ModelGrid
        Format:
        -------
        Point to fields of grid at nodes (if they are defined) or to default values from config files
        - denudation_rate_bedrock_mm_y
        - density_bedrock_kg_m3
        - target_mineral_presence_bedrock
        - density_cover_kg_m3
        - thickness_cover_m
        - concentration_ + nuclide + _ + mineral + _top_bedrock_atom_g
        Calculated fields:
        - absorption_coefficient_spallation_cm
        - absorption_coefficient_muon_fast_cm
        - absorption_coefficient_muon_slow_cm
        - thickness_cover_cm
        - denudation_rate_bedrock_cm_y
        """
               
        m = {}
        g = self.grid
        # Set unset grid fields to default values
        for my_field in ["density_bedrock_kg_m3", "density_cover_kg_m3","denudation_rate_bedrock_mm_y",  
                          "target_mineral_presence_bedrock","thickness_cover_m"]:        
            if my_field not in g.at_node:
                m[my_field] = self.geo_setting[my_field] * np.ones(g.number_of_nodes)
            else:
                m[my_field] = g.at_node[my_field]
            
        for my_field in ["concentration_" + self.nuclide + "_" + self.mineral + "_top_bedrock_atom_g"]:        
            if my_field not in g.at_node:
                m[my_field] = self.geochemical_setting[my_field] * np.ones(g.number_of_nodes)
            else:
                m[my_field] = g.at_node[my_field];
                
        # absorption coefficient (cm^-1) from absorption mean free path (in g/cm^-2) and bedrock density (kg/m^3)
        m["absorption_coefficient_spallation_cm_1"] = \
            m["density_cover_kg_m3"] * 1e-3 / self.cosmogenic_parameter["e_folding_length_in_rock_spallation_g_cm2"]
        
        
        if "muon_slow" in self.production_pathway and "muon_fast" in self.production_pathway and \
            "muon" in self.production_model.keys() and "Braucher_2011" in self.production_model.values():
            m["absorption_coefficient_muon_fast_cm_1"] = \
                m["density_cover_kg_m3"] * 1e-3 / self.cosmogenic_parameter["e_folding_length_in_rock_muon_fast_g_cm2"]
            m["absorption_coefficient_muon_slow_cm_1"] = \
                m["density_cover_kg_m3"] * 1e-3 / self.cosmogenic_parameter["e_folding_length_in_rock_muon_slow_g_cm2"]
        
        
        
        self._private_grid_field_dict = m
    
    @property
    def production_model(self):
        """Get the list of production pathways for the Component
        """
        return self._production_model
    
    def _set_production_model(self):
        """ Set the dictionary of production models used (peculiarly for muons
        
        Type: dict(str)
        Format: {"muon": "Braucher_2011"}
        """
        v = self._init_parameter["production_model"]
        self._check_var_type(v, dict)
        
        for key, val in v.items():
            if key not in ["muon"]:
                raise ValueError("Only the 'muon' model can be parametered")
        if "muon_slow" in self.production_pathway and \
            ("muon" not in v.keys() or v["muon"] != "Braucher_2011"):
            raise ValueError("With muon production pathway, the 'muon' model must be parametered to 'Braucher_2011'")
        self._production_model = v
        
    @property
    def production_pathway(self):
        """Get the list of production pathways for the Component
        """
        return self._production_pathway
    
    def _set_production_pathway(self):
        """ Set the list of production pathways the component
        
        Type: list(str)
        """
        v = self._init_parameter["production_pathway"]
        self._check_var_type(v, list)
        for val in v:
            if val not in ["spallation", "muon_slow", "muon_fast"]:
                raise ValueError("At least one inpu production pathway is not implemented (see documentation)")
        self._production_pathway = v
    
    # utils
    def _check_var_type(self, var, type1):
        """Typically used to check the type of the arguments yielded to functions or methods.
        check if the type of the variable corresponds to the one expected

        Parameters
        ----------
        var : Multiple
            Variable to check type
        expected_type
           Type which is expected for the variable var
        """
        if not isinstance(var, type1):
            raise ValueError("The variable doesn't have the expected type (Check documentation)")
                
    def _check_var_list_type(self, var_list, expected_type_list):
        """Typically used to check the type of the arguments yielded to functions or methods.
        For each variable in the list, check if its type corresponds to the one expected

        Parameters
        ----------
        var_list : List
            List of variables (of different types) to check type
        expected_type_list
            List of the types which are expected for each variable of var_list
        """
        if not isinstance(var_list, list) or not isinstance(expected_type_list, list) :
            raise ValueError("var_list must be a list of variables and expected_type_list a list of types")
        
        if len(var_list) != len(expected_type_list):
            raise ValueError("The list of variables must have the same length as the list of types")    
        for i in range(0, len(var_list)):
            if not isinstance(var_list[i], expected_type_list[i]):
                raise ValueError("At least one of the variables doesn't have the expected type (Check documentation)")   
        
    def _get_dictionary_from_file(self, filename):
        """Get a dictionary from a csv file where keys are stored in the 1st column
        and values are stored in the other columns. These values are either unique (1 column)
        or a list (several columns)
        Note that files should at least contain one couple of key, value
        CHECK IF NECESSARY

        Parameters
        ----------
        filename : String
            Directory + filename where the data are stored            
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
        myDict: Dictionary
            Dictionary to check keys
        keyList: List
            List of keys to check within the dictionary (List of Strings)
        """
        self._check_var_list_type([myDict, keyList], [dict, list])
        for my_key in keyList:
            if my_key not in myDict.keys():
                raise ValueError("The dictionary (or input file) does not contain all necessary keys (check documentation)")
            
    # initialize the component    
    def prepare_first_run(self, cosmogenic_production_scaler):
        """Prepare the object to undergo its first run.
        Initializes attributes and grid fields to default values from init params f necessary.
        Parameters
        ----------
        cosmogenic_production_scaler: landlab.component.CosmogenicProductionScaler
            cosmogenic production scaler which will yields the scaling factors (as a function of latitude and elevation)
        """
        
        # Initialize attributes of the component
        self._set_nuclide()
        self._set_mineral()
        self._set_production_pathway()  
        self._set_production_model()    # check compatibility with production pathway   
        self._set_init_condition()
        self._set_path_prefix()
                        
        # Initialize the dictionaries of parameters and data
        self._set_geochemical_setting()
        self._set_geo_setting()
        self._set_cosmogenic_parameter()
        
        # Continue to initialize attributes of the component 
        self._set_decay_constant_y_1() # from _cosmogenic_parameter
        
        self._set_cosmogenic_production_scaler = cosmogenic_production_scaler
        
        # Initialize privated grid field values 
        self._set_private_grid_field_dict() # including with scaling factors from cosmogenic production scaler
        self._set_concentration_contribution_top_bedrock_before_first_run() # init concentrations if steady state cosmogenic production
        if self._verbose:
            print(landlab.core.messages.format_message("prepare_first_run() executed"))

    
    # job methods
              
    # Methods to calculate the contribution of each production pathway to the
    # concentration of nuclides, obtained from an initial (inherited) 
    # concentration which evolved during a time step
    ############################################################################
    
    @property
    def concentration_contribution_top_bedrock_atom_g(self):
        """Get the dictionary of the contributions to nuclide concentration at bedrock surface
        obtained from the different 
        production pathways for current run.
        """
        d = dict()
        pg = self.private_grid_field_dict
        for p in self.production_pathway:
            d[p] = pg["concentration_contribution_top_bedrock_from_" + p + "_atom_g"]
        return d
        
    def _set_concentration_contribution_top_bedrock_atom_g(self, dt):
        """Calculates the contribution of each production pathway to
        the nuclide concentration at bedrock surface,
        obtained after dt passed, taking into account radioactive decay
        
        We suppose an exponential production depth-function as in (Braucher et al., 2011), Equation (1) , p. 4
        
        For spallation, we suppose an exponential production depth-function as in (Lal, 1991), equation (6), p. 430
        For muons, we split between slow muons and fast muons, andwe suppose an exponential production 
        depth-function as in (Braucher et al., 2011), Equation (1) , p. 4
        which is identic as the (Lal, 1991) equation.
        
        production in bedrock only.
        Yields a positive value

        Parameters
        ----------
        dt: float (time) in [y]
            The imposed timestep.
        """
        self._check_var_list_type([dt], [float])
        pg = self.private_grid_field_dict 
        ps = self.production_pathway
        g = self._grid
        
        # Update of values of dictionary, unfortunately dictionary of references is not possible in Python !!!
        pg["denudation_rate_bedrock_mm_y"] = g.at_node["denudation_rate_bedrock_mm_y"]
        pg["thickness_cover_m"] = g.at_node["thickness_cover_m"]
            
        # cython use
        n = self.nuclide
        m = self.mineral
        pathways_n = len(ps)
        nodes_n = self._grid.number_of_nodes
        absorption_coefficient = np.empty((pathways_n, nodes_n)) # depend on density (which can be set on the grid nodes)
        slhl_prod_rate = np.empty(pathways_n)
        concentration_contribution_previous = np.empty((pathways_n, nodes_n))
        scaling_factor = np.empty((pathways_n, nodes_n));
        for i in range(pathways_n): # This LOOP is EXPENSIVE (I don't know why)
            p = ps[i]
            if p in ["spallation"] or \
                (p in ["muon_fast", "muon_slow"] and self.production_model["muon"] == "Braucher_2011"):
                
                absorption_coefficient[i, :] = pg["absorption_coefficient_" + p + "_cm_1"] # BEWARE: coefficient for cover
                slhl_prod_rate[i] = self.cosmogenic_parameter["slhl_prod_rate_" + n + "_" + m + "_" + p + "_atom_g_y"]
                concentration_contribution_previous[i, :] = \
                    pg["concentration_contribution_top_bedrock_previous_from_" + p + "_atom_g"]
                scaling_factor[i, :] = self.cosmogenic_production_scaler.cosmogenic_scaling_factor[p]  
                     
        # During dt, we assume a constant bedrock denudation rate and constant cosmogenic production rate,
        # with depth z = cover thickness = 0 when no cover.
        thickness_cover_cm = pg["thickness_cover_m"] * 100  
        denudation_rate_bedrock_cm_y = pg["denudation_rate_bedrock_mm_y"] * 0.1
        _return = np.asarray(self._cfuncs._get_concentration_contribution_top_bedrock_cosmogenic_exponential(\
            denudation_rate_bedrock_cm_y, self.decay_constant_y_1, absorption_coefficient,
            concentration_contribution_previous, slhl_prod_rate,
            scaling_factor, dt, pg['target_mineral_presence_bedrock'],
            thickness_cover_cm))
                 
        for i in range(pathways_n):
            pg["concentration_contribution_top_bedrock_from_" + ps[i] + "_atom_g"] = _return[i, :]
        pg["concentration_" + self._nuclide + "_" + self.mineral + "_top_bedrock_atom_g"] = \
            self.grid.at_node["concentration_" + n + "_" + m + "_top_bedrock_atom_g"] = _return[-1, :]
        # end of cython use   
        return
    
        c = 0
        
        for p in ps:
            if p in ["spallation"]:
                cp = self._calc_concentration_contribution_top_bedrock_cosmogenic_exponential_model_atom_g(dt, production_pathway=p)
            elif p in ["muon_fast", "muon_slow"] and self.production_model["muon"] == "Braucher_2011":
                cp = self._calc_concentration_contribution_top_bedrock_cosmogenic_exponential_model_atom_g(dt, production_pathway=p)
            pg["concentration_contribution_top_bedrock_from_" + p + "_atom_g"] = cp
            c += cp  
        pg["concentration_" + self._nuclide + "_" + self.mineral + "_top_bedrock_atom_g"] = \
            self.grid.at_node["concentration_" + self._nuclide + "_" + self.mineral + "_top_bedrock_atom_g"] = c
            
    def _calc_concentration_contribution_top_bedrock_cosmogenic_exponential_model_atom_g(self, dt, production_pathway="spallation"):
        """Calculates the contribution of the input cosmogenic production pathway and radioactive decay 
        on this contribution, to the concentration of nuclides at bedrock surface, produced by cosmic rays 
        through  during one time step of size dt.
        
        We assume that exponential production depth-function, as in (Lal, 1991), Equation (6) p. 430
        
        production in bedrock only.
        Yields a positive value

        Parameters
        ----------
        dt: float (time) in [y]
            The imposed timestep.
        production_pathway: str
            Pathway for which the calculation is done.
            Choice: "spallation", "muon_slow", "muon_fast"
        """
        
        self._check_var_list_type([dt, production_pathway], [float, str])
        g = self._grid
        pg = self.private_grid_field_dict
        p = production_pathway
        if p not in self.production_pathway:
            val = np.zeros(self.grid.number_of_nodes)
            return val
        
        # During dt, we assume a constant bedrock denudation rate and constant cosmogenic production rate,
        # with depth z = cover thickness = 0 when no cover.
        eps = denudation_rate_bedrock_cm_y = pg["denudation_rate_bedrock_mm_y"] * 0.1 if pg["denudation_rate_bedrock_mm_y"]> 0 else 1e-9
        lam = self.decay_constant_y_1
        mu = pg["absorption_coefficient_" + p + "_cm_1"] # BEWARE: coefficient for cover
        n = self.nuclide
        m = self.mineral
        N_0 = pg["concentration_contribution_top_bedrock_previous_from_" + p + "_atom_g"]
        P_0 = self.cosmogenic_parameter["slhl_prod_rate_" + n + "_" + m + "_" + p + "_atom_g_y"]
        scaling = self.cosmogenic_production_scaler.cosmogenic_scaling_factor[p]
        t = dt
        target = pg['target_mineral_presence_bedrock']
        z = thickness_cover_cm = pg["thickness_cover_m"] * 100
        
        val = N_0 * np.exp(-mu * eps * t ) * (np.exp(- lam * t))  # radioactive decay-corrected inherited concentration related to production pathway
                                                       # applied to the slice of rock advected up to surface
        
        val += scaling * target * P_0 / ( -mu * eps) * (np.exp(-mu * eps * t) - 1) \
                + P_0 * lam / (mu * eps) * (t * np.exp(-mu * eps * t) + 1 / (mu * eps) * (np.exp(-mu * eps * t) - 1)) 
                # decay-corrected production by the pathway within 
                # the slice of rock advected up to surface
        #val += scaling * target * P_0 \
        #     / (lam +  mu * eps) * np.exp(- mu * z) * (1 - np.exp(-(lam + mu * eps) * t)) 
        
        return val
          
    # Methods to calculate and set steady-state concentrations related to 
    # spallation, slow and fast muons
    ############################################################################
    
    @property
    def concentration_contribution_steady_state_top_bedrock_atom_g(self):
        """Get the dictionary of the contributions to nuclide concentration at bedrock surface
        obtained from the different 
        production pathways once steady state denudation was reached, after long irradiation.
        """
        d = dict()
        pg = self.private_grid_field_dict
        for p in self.production_pathway:
            d[p] = pg["concentration_contribution_steady_state_top_bedrock_from_" + p + "_atom_g"]
        return d
        
    def _set_concentration_contribution_steady_state_top_bedrock_atom_g(self):
        """Calculate and set the contributions to nuclide concentration at bedrock surface
        obtained from the different 
        production pathways once steady state denudation was reached, after long irradiation.
        
        We suppose that spallation production follows an exponential depth-function as in (Lal, 1991),
         equation (8) p. 431.
        
        We split fast and slow muons, and we suppose that production follows 
        an exponential depth-function as in (Braucher et al., 2011),equation (1) p. 4
        
        production in bedrock only.
        Yields a positive value
        """
        
        pg = self.private_grid_field_dict 
        ps = self.production_pathway
        
        for p in ps:
            if p in ["spallation"]:
                val = self._calc_concentration_contribution_steady_state_top_bedrock_cosmogenic_exponential_model_atom_g(production_pathway=p)
            elif p in ["muon_fast", "muon_slow"] and self.production_model["muon"] == "Braucher_2011":
                val = self._calc_concentration_contribution_steady_state_top_bedrock_cosmogenic_exponential_model_atom_g(production_pathway=p)
            pg["concentration_contribution_steady_state_top_bedrock_from_" + p + "_atom_g"] = val
                     
    def _calc_concentration_contribution_steady_state_top_bedrock_cosmogenic_exponential_model_atom_g(self, production_pathway="spallation"):
        """Calculates the concentration derived from the production pathway 
        when steady-state production is reached
        (means that denudation rates have been steady since t >> lambda + mu * epsilon
        Here we choose t >= 5 * (lambda + mu * epsilon)
        
        Note that for the scaling factor, we assume that elevation remained steady during the period
        We also assume that rock density and cover thickness/density remained steady (which is probably incorrect, at least in valleys
        and in landsliding areas.
        See (Lal, 1991), equation (8) p. 431.
        production in bedrock only.
        Yields a positive value
        Parameters
        ----------
        production_pathway: str
            Pathway for which the calculation is done.
            Choice: "spallation", "muon_slow", "muon_fast"
        """
        self._check_var_list_type([production_pathway], [str])
        pg = self.private_grid_field_dict
        p = production_pathway
        if p not in self.production_pathway:
            val = np.zeros(self.grid.number_of_nodes)
            return val
        
        # we assume a constant bedrock denudation rate and constant cosmogenic production rate,
        # with depth z = cover thickness = 0 when no cover. 
        # Using equation 8 in Lal, 1991:   
        eps = denudation_rate_bedrock_cm_y = pg["denudation_rate_bedrock_mm_y"] * 0.1
        lam = self.decay_constant_y_1
        mu = pg["absorption_coefficient_" + p + "_cm_1"] # BEWARE: coefficient for cover
        P_0 = self.cosmogenic_parameter["slhl_prod_rate_" + self.nuclide + "_" + self.mineral + "_" + p + "_atom_g_y"]
        scaling = self.cosmogenic_production_scaler.cosmogenic_scaling_factor[p]
        target = pg['target_mineral_presence_bedrock']             
        z = thickness_cover_cm = pg["thickness_cover_m"] * 100
        
        val = scaling * target * P_0 / (lam +  mu * eps) * np.exp(- mu * z)        
        return val
    
    def _set_concentration_contribution_top_bedrock_before_first_run(self):
        """ If initial steady-state cosmogenic production set to false,
         the initial concentration contribution are set to 0.
         
        If initial steady-state cosmogenic production set to true, 
        the component considers that denudation rate and exposure to cosmic rays remain steady for sufficient time
        to equilibrate (radioactive) loss and production of all cosmogenic pathways. In that case, it 
        will calculate steady-state concentrations
        and will add them to the initial top bedrock concentrations (considered here as inherited).
        Note that in THIS version, intial top bedrock concentrations should be parametered to 0.
        
        That means that denudation rates and exposure have been steady since t >> lambda + mu * epsilon
        Here we choose t >= factor * (lambda + mu * epsilon)
        
        BEWARE: all production pathways are considered steady-state, which might be probably untrue for muons

        See (Lal, 1991), equation (8) p. 431.
        """
        pg = self.private_grid_field_dict 
        ps = self.production_pathway
        g = self.grid
        c = 0
        if self.init_condition["steady_state_cosmogenic_production"] == False:   
            for p in ps:
                pg["concentration_contribution_top_bedrock_from_" + p + "_atom_g"] = np.zeros(g.number_of_nodes)
        else:
            self._set_concentration_contribution_steady_state_top_bedrock_atom_g()
            for p in ps:
                cp = pg["concentration_contribution_steady_state_top_bedrock_from_" + p + "_atom_g"]
                pg["concentration_contribution_top_bedrock_from_" + p + "_atom_g"] = cp
                c += cp
                
        pg["concentration_" + self.nuclide + "_" + self.mineral + "_top_bedrock_atom_g"] = \
            self.grid.at_node["concentration_" + self.nuclide + "_" + self.mineral + "_top_bedrock_atom_g"] = c
        
    # Methods to calculate and set time to reach steady-state                        
    ############################################################################
    
    @property
    def steady_state_time_to_reach_y(self):
        """Get the dictionary of the times to reach steady state production for each production
        pathway,
        once steady state denudation was reached, after long irradiation.
        """
        d = dict()
        pg = self.private_grid_field_dict
        for p in self.production_pathway:
            d[p] = pg["steady_state_time_to_reach_from_" + p + "_y"]
        return d
             
    def _set_steady_state_time_to_reach_y(self):
        """Calculates the time required for each production pathway to reach steady state when
        steady-state denudation was reached, after long irradiation.   
        
        We suppose that spallation production follows an exponential depth-function as in (Lal, 1991), p. 431.
        
        We suppose that fast muon production follows an exponential depth-function as in (Braucher et al., 2011),
        equation (1), p. 4.
        As in Braucher et al., 2011), we also suppose this function is independent 
        of the nuclide (see discussion Balco et al., 2017)
        
        production in bedrock only.
        Yields a positive value
        """
        self._check_var_list_type([dt], [float])
        pg = self.private_grid_field_dict 
        ps = self.production_pathway
        
        for p in ps:
            if p in ["spallation"]:
                val = self._calc_steady_state_time_to_reach_exponential_model_y(production_pathway=p)
            elif p in ["muon_fast", "muon_slow"] and self.production_model["muon"] == "Braucher_2011":
                val = self._calc_steady_state_time_to_reach_exponential_model_y(production_pathway=p)
            pg["steady_state_time_to_reach_from_" + p + "_y"] = val
             
        pg = self.private_grid_field_dict
        p = "muon_fast"
        val = self._calc_steady_state_time_to_reach_exponential_model_y(production_pathway=p)
        
        pg["steady_state_time_to_reach" + p] = val
     
    
    def _calc_steady_state_time_to_reach_exponential_model_y(self, production_pathway="spallation"):
        """Calculates the time required for the production pathway to reach steady state when
        steady-state denudation is achieved, after long irradiation, supposing that the pathway 
        follows an exponential depth-function
        as for spallation in (Lal, 1991).
        
        See (Lal, 1991), p. 431.
        production in bedrock only.
        Yields a positive value
        
        Parameters
        ----------
        production_pathway: str
            Pathway for which the calculation is done.
            Choice: "spallation", "muon_slow", "muon_fast"
        
        Returns
        -------
        """
        self._check_var_list_type([production_pathway], [str])
        p = production_pathway
        pg = self.private_grid_field_dict
        
        if p not in self.production_pathway:
            val = np.zeros(self.grid.number_of_nodes)
            return val
        
        # we assume a constant bedrock denudation rate and constant cosmogenic production rate
        eps = denudation_rate_bedrock_cm_y = pg["denudation_rate_bedrock_mm_y"] * 0.1
        lam = self.decay_constant_y_1
        mu = pg["absorption_coefficient_" + p + "_cm_1"] # BEWARE: coefficient for cover
                
        val = self.cosmogenic_parameter["steady_state_time_to_reach_factor"] * (lam +  mu * eps)
        
        return val
        
        
    # Run the component
    ############################################################################
    
    def run_one_step(self, dt):
        """Advance BedrockCosmogenicProducer component by one time step of size dt.
        Compute and update the bedrock concentration of nuclides of the grid, at surface of bedrock
        Parameters
        ----------
        dt: float (time) in [y]
            The imposed timestep.
        """
        dt = float(dt)
        
        self._set_concentration_contribution_top_bedrock_previous_atom_g()
        self._set_concentration_contribution_top_bedrock_atom_g(dt)
                
        """ NB: We suppose that the addition or subtraction always yield a result 
        different from the initial concentration. This might be not the case
        (if for each dt, the budget is very small. This happens for Be10 and 
        concentrations < 0.01 atom/g) and then a totalizer of all budgets 
        done for each dt might be necessary
        """
        
        """ TODO: Find a way to monitor concentration and budget evolution over time
        """