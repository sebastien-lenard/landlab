"""
flow_router.py contains the FlowRouter component. It calculates the flow
depending on gradients (FlowDirector), overcome depressions
(DepressionFinderAndRouter), and accumulates flow and calculates
drainage areas (FlowAccumulator).

Related components: PriorityFloodFlowRouter (wrapper of the RichDEM package),
FlowDirectorSteepest, FlowDirectorD8, FlowAccumulator,
DepressionFinderAndRouter, LakeMapperBarnes, SinkFillerBarnes


@author: Sebastien Lenard sebastien.lenard@gmail.com
@date: 2022, July
"""
import numpy as np

from landlab import Component, RasterModelGrid, NetworkModelGrid
from landlab.components.depression_finder.floodstatus import FloodStatus


class FlowRouter(Component):
    """
    The FlowRouter carries out 2 operations: WORK TO DO: check for active
    links (when links between two core nodes or a core node and boundary
    node are inactive)

    (1) Public method run_flow_directions() calculates flow directions over a
    surface (usually the elevation, node field "topographic__elevation")
    from the base-level nodes downstream to upstream. Depressions are resolved
    and the flow bypasses them simultaneously.

    . - The method digs channels from the pit nodes to the nearest nodes (in
    .   the priority-flood meaning, Barnes et al., 2014) outside of the
    .   depression containing the pit. This resolves the depression as all
    .   nodes in the depression now have a drainage path to a base-level node

    . - The method ensures that the highest peaks of the surface and the
    .   base-level, perimeter, and closed nodes don't receive flow. This
    .   method also supplies a depression-free elevation surface (node field
    .   "depression_free__elevation"), which is constructed by filling
    .   depressions. This filling is done based on a minimum relative
    .   difference in elevation between two nodes, so that a donor is an
    .   epsilon higher than its receiver, downstream to the outlet of
    .   the depression. This surface can be yielded as an input to other
    .   FlowDirector components to generate flow directions using flow metrics
    .   not restricted to the ones implemented by FlowRouter (e.g.
    .   FlowDirectorMFD).

    (2) Public method run_flow_accumulations() determines the downstream to
    upstream order of nodes and calculates drainage areas and flow
    accumulations. The algorithm is similar to the FlowAccumulator.

    The public method run_one_step() can carry out the 2 operations.

    Tutorials
    = = = = =
    flow_direction_and_accumulation/the_FlowRouter

    Remarks on usage
    = = = = = = = =
    - FlowRouter is compatible with all grids, including the
    .  VoronoiDelaunayGrid and the NetworkModelGrid.
    - Flow directions are one-to-one (single flow firection), similarly to the
    .  FlowDirectorSteepest (D4 or D8)(O'Callaghan and Mark, 1984).
    - We don't have a sink mode. Thus we can't accept a manual supply of sinks
    .  (=pits). To calculate directions and keeping the sinks, use the
    .  FlowDirector components.
    - Once instantiated, the component will not consider any modification of
    .  the basic components of the grid: nodes, links, cells, faces, boundary,
    .  base-level (neither shift of position, change of status, or change of
    .  length).
    - If no base-level, we take the perimeter node having the lowest value
    .  of the surface.
    - For NetworkModelGrid, we take a cell area = 1.
    - This component has a run limit of 1e9 nodes.

    Remarks on implementation
    = = = = = = = = = = = = =
    - For optimization, the code of the component is partly delegated to
    .  Cython.
    - For flow direction and depression overcoming, we constructed and adapted
    .  an algorithm  based on the priority flood algorithm description #4 in
    .  Barnes et al., 2014. The major differences with the Barnes algorithm is
    .  that flow outlets are base-level nodes (and not perimeter nodes), the
    .  calculations of fields generated by FlowDirector and FlowAccumulator
    .  components, and some performance tweaks linked to the use of Cython.
    - The component is not a wrapper of the RichDEM package.
    - For flow accumulation, we slightly adapted the algorithm implemented
    in the FlowAccumulator,
    based on the upstream/downstream O(n) algorithm of
    Braun and Willett, 2013.

    References
    ----------
    Required Software Citation(s) Specific to this Component
    --------------------------------------------------------
    Additional References
    ---------------------
    Barnes, R., Lehman, C., Mulla, D. (2014). An efficient assignment of
    drainage direction over flat surfaces in raster digital elevation models.
    Computers & Geosciences, 62, 128-135.
    https://dx.doi.org/10.1016/j.cageo.2013.01.009

    O’Callaghan, J.F., Mark, D.M. (1984). The Extraction of Drainage Networks
    from Digital Elevation Data. Computer Vision, Graphics and Image
    Processing, 28, 328-344. https://dx.doi.org/10.1016/S0734-189X(84)80011-0
    """

    from ext.single_flow.priority_routing import init_tools as _init_tools_funcs
    from ext.single_flow.priority_routing import breach as _breach_funcs
    from ext.single_flow.accumulation import accumulation as _accumulation_funcs

    _name = "FlowRouter"
    _unit_agnostic = True

    _info = {
        """Note: if another component use the fields, the field description
        must be the same (except for intent and optional).
        To be sure that the intent=in fields are created if non existent,
        intent is set to inout."""
        # Input grid-fields
        "cell_area_at_node": {
            "dtype": float,
            "intent": "inout",
            "optional": True,
            "units": "m**2",
            "mapping": "node",
            "doc": "Area of the cell surrounding the node",
        },
        "topographic__elevation": {
            "dtype": float,
            "intent": "inout",
            "optional": True,
            "units": "m",
            "mapping": "node",
            "doc": "Land surface topographic elevation",
        },
        "water__unit_flux_in": {
            "dtype": float,
            "intent": "inout",
            "optional": True,
            "units": "m/s",
            "mapping": "node",
            "doc": "External volume water per area per time input to each"
            + " node (e.g., rainfall rate)",
        },
        # Output grid-fields
        "depression_free__elevation": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Filled land surface topographic elevation.",
        },
        "depression__depth": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Depth of depression below its spillway point",
        },
        "depression__outlet_node": {
            "dtype": int,
            "intent": "out",
            "optional": False,
            "units": "-",
            "mapping": "node",
            "doc": "If a depression, the id of the outlet node for that"
            + " depression, otherwise grid.BAD_INDEX",
        },
        "drainage_area": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m**2",
            "mapping": "node",
            "doc": "Upstream accumulated surface area contributing to"
            + " the node's discharge",
        },
        "flood_status_code": {
            "dtype": int,
            "intent": "out",
            "optional": False,
            "units": "-",
            "mapping": "node",
            "doc": "Map of flood status (_PIT, _CURRENT_LAKE, _UNFLOODED,"
            + " or _FLOODED).",
        },
        "flow__link_to_receiver_node": {
            "dtype": int,
            "intent": "out",
            "optional": False,
            "units": "-",
            "mapping": "node",
            "doc": "ID of link downstream of each node, which carries the"
            + " discharge",
        },
        "flow__receiver_node": {
            "dtype": int,
            "intent": "out",
            "optional": False,
            "units": "-",
            "mapping": "node",
            "doc": "Node array of receivers (node that receives flow from"
            + " current node)",
        },
        "flow__upstream_node_order": {
            "dtype": int,
            "intent": "out",
            "optional": False,
            "units": "-",
            "mapping": "node",
            "doc": "Node array containing downstream-to-upstream ordered list"
            + " of node IDs",
        },
        "outlet_node": {
            "dtype": int,
            "intent": "out",
            "optional": False,
            "units": "-",
            "mapping": "node",
            "doc": "Base-level outlet of the flux coming from the node.",
        },
        "surface_water__discharge": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m**3/s",
            "mapping": "node",
            "doc": "Volumetric discharge of surface water",
        },
        "topographic__steepest_slope": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "-",
            "mapping": "node",
            "doc": "The steepest *downhill* slope",
        },
    }

    def __init__(
        self,
        grid,
        surface="topographic__elevation",
        diagonals=[True, False][0],
        runoff_rate=[None, 1, "water__unit_flux_in"][0],
        single_flow=[True, False][0],
    ):
        """Initialize the component and its attributes.
        If all boundary nodes are closed, force the 1st one to be open.

        Parameters
        ----------
        grid : ModelGrid.
            A Landlab grid (all classes accepted).
        surface : str, optional.
            The surface to direct an accumulate flow across. An at-node field
            name.
            > If the field doesn't exist, creates a field of this name with
            . random values.
            > If not a str, force to "topographic__elevation". Contrary to
            . other components, surface cannot be a nd.array.
        diagonals: bool, optional.
            If False, component excludes diagonal links for RasterModelGrid.
            > Automatically set to False for other grid classes.
        runoff_rate: None, int, float, nd.array, or str, optional, [m/y].
            Controls the constant or variable values of the node field
            "water__unit_flux_in", which is the influx of water to nodes
            that originates from sources or sinks external to the grid.
            > Positive e.g. rainfall (including evapotranspiration)
            > or negative e.g. karstic losses.
            x x x x
            Options:
            x x x x
            > None: The values of the field "water__unit_flux_in" are not
            .  modified.
            > float: A spatially constant value overwrites the values of the
            .  "water__unit_flux_in".
            > nd.array or str : Spatially variable values permanently
            .  overwrites the values of the "water__unit_flux_in". The values
            .  are given either by an array or by another existing field "str".
        single_flow: bool
            True, the component can calculate all flows as single-to-single
            flow and update the grid fields accordingly.
            False, the component calculate single flow directions but only
            update depression_free__elevation, depression__depth,
            depression__outlet_node and flood_status_code. It doesn't
            calculate flow accumulation. This is because the component cannot
            deal with multiple flows and need to be combined with a
            FlowDirectorMFD, which will setup the other output fields with the
            right dimensions and will work on the depression_free__elevation
            surface (rather than the topographic__elevation surface).

        Examples
        --------
        1. RasterModelGrid

        >>> # Libraries
        >>> import numpy as np
        >>> from landlab import RasterModelGrid
        >>> #from landlab.components import FlowRouter
        >>> # Creation of the grid
        >>> params = {"shape": (5, 5), "xy_spacing": (10, 10)}
        >>> g = RasterModelGrid(**params)
        >>> # Generation ot the topography
        >>> random_generator = np.random.Generator(np.random.PCG64(seed=500))
        >>> z = g.add_field("topographic__elevation", 10. *
        ...     random_generator.random(25))
        >>> # Creation of the router
        >>> router_params = {"grid": g}
        >>> router = FlowRouter(**router_params)
        >>> g.at_node["depression_free__elevation"]
        array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,
                0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])

        Set up a external water influx if none
        >>> g.at_node["water__unit_flux_in"]
        array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1., 1.,
        1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.])

        2. NetworkModelGrid
        >>> from landlab import NetworkModelGrid
        >>> # Creation of the grid
        >>> params = {"yx_of_node": ((0, 100, 200, 200, 300, 400, 400, 125),
        ...     (0, 0, 100, -50, -100, 50, -150, -100)),
        ...     "links": ((1, 0), (2, 1), (1, 7), (3, 1), (3, 4), (4, 5),
        ...     (4, 6))}
        >>> g = NetworkModelGrid(**params)
        >>> # Generation ot the topography
        >>> z = g.add_field("topographic__elevation", [0.0, 0.08, 0.25, 0.15,
        ...     0.25, 0.4, 0.8, 0.8])
        >>> # Creation of the router
        >>> router_params = {"grid": g}
        >>> router = FlowRouter(**router_params)

        Includes creation of an outlet (status_at_node = 1)
        >>> g.status_at_node
        array([1, 0, 0, 0, 0, 0, 0, 0], dtype=uint8)

        Set up of an cell area of 1
        >>> g.at_node["cell_area_at_node"]
        array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.])
        """
        super(FlowRouter, self).__init__(grid)
        g = self._grid

        self.initialize_optional_output_fields()
        self._single_flow = single_flow
        if single_flow:
            self.initialize_output_fields()
        else:
            for field in [
                "depression_free__elevation",
                "depression__depth",
                "depression__outlet_node",
                "flooded_status_code",
            ]:
                if field not in g.at_node.keys():
                    g.add_zeros(field, at="node")

        nodes_n = g.number_of_nodes
        self._nodes = nodes = (
            g.nodes.reshape(nodes_n) if (isinstance(g, RasterModelGrid)) else g.nodes
        )

        # 1. Initialization of the input parameters
        ###########################################
        # 1.1. Surface where the flow is directed
        s = self._surface = surface
        if type(s) != str:
            s = "topographic__elevation"
        if s not in g.at_node:
            g.add_field(s, np.zeros(nodes_n, dtype=float))
        z = g.at_node[s]

        # 1.2. Water influx external to grid
        s = "water__unit_flux_in"
        v = runoff_rate
        alph = v is None or type(v) not in ([np.array, str])
        self._uniform_water_external_influx = True if alph else False
        v = float(v) if type(v) == int else v
        if v is None or type(v) not in ([float, np.array, str]):
            v = 1.0
        if s in g.at_node and v is not None:
            g.at_node["water__unit_flux_in"] = np.full(
                nodes_n, v if v is not None else 1.0
            )

        # 1.3. Options
        self._diagonals = (
            diagonals if isinstance(g, RasterModelGrid) else False
        )  # noqa: E501

        # 2. Boundary conditions
        ########################
        # 2.1. Boundary settings: guarantee of one
        # base-level node (at least).

        if not np.any(
            g.status_at_node == g.BC_NODE_IS_FIXED_VALUE
        ) and not np.any(  # noqa: E501
            g.status_at_node == g.BC_NODE_IS_FIXED_GRADIENT
        ):
            node = np.where(z[g.perimeter_nodes] == np.min(z[g.perimeter_nodes]))[0][0]
            g.status_at_node[
                g.perimeter_nodes[node]
            ] = g.BC_NODE_IS_FIXED_VALUE  # noqa: E501

        self._closed_nodes = nodes[
            np.where(g.status_at_node == g.BC_NODE_IS_CLOSED)
        ]  # noqa: E501
        self._base_level_nodes = nodes[
            (
                np.where(g.status_at_node == g.BC_NODE_IS_FIXED_VALUE)
                or np.where(g.status_at_node == g.BC_NODE_IS_FIXED_GRADIENT)
            )[0]
        ]
        self._base_level_and_closed_nodes = np.concatenate(
            (self._base_level_nodes, self._closed_nodes)
        )

        # 2.2 Cell area at boundary nodes = 0, NetworkModelGrid cell area = 1
        # used in run_flow_accumulations()
        g.at_node["cell_area_at_node"] = (
            np.full(nodes_n, 1.0)
            if isinstance(g, NetworkModelGrid)
            else g.cell_area_at_node.copy()
        )
        self._cell_area_at_nodes = g.at_node["cell_area_at_node"]

        # 2.2. Max number of nodes (for sort head/tails/links)
        self._max_number_of_nodes = 1e9

        # 2.3. Minimum relative difference in elevation required in the
        # construction of the depression_free_elevations surface
        self._min_elevation_relative_diff = 1e-8

        # 3. Determination of stable input grid data necessary to calculate
        # run_flow_directions
        ###################################################################
        diagonals = self._diagonals
        z = g.at_node[self._surface]
        nodes_n = g.number_of_nodes

        if diagonals:
            # d8 include classic nodes and nodes at diagonal
            # tails and heads sorted by link ids
            head_nodes = g.nodes_at_d8[:, 1]
            tail_nodes = g.nodes_at_d8[:, 0]
            links_n = g.number_of_d8
        else:
            head_nodes = g.node_at_link_head
            tail_nodes = g.node_at_link_tail
            links_n = g.number_of_links

        self._neighbors_max_number = (
            8
            if isinstance(g, RasterModelGrid)
            else len(g.adjacent_nodes_at_node[0])  # noqa: E501
        )

        """ Link infos(tail, head, link id, gradient) sorted by head id.
        These infos are voluntarily duplicate (tails are appended to heads
        and heads appended to tails,
        which lead to pseudo heads and pseudo tails) to be sure to access to
        all links connecting a node just by filtering head id = node id.
        This is to use numpy.where on small arrays and limit data handling
        during the flow direction process.
        > Note that in landlab tail ids < head ids."""
        self._dupli_links = dupli_links = np.concatenate(
            (np.arange(links_n), np.arange(links_n))
        )
        self._pseudo_head_nodes = pseudo_head_nodes = np.concatenate(
            (head_nodes, tail_nodes)
        )
        self._pseudo_tail_nodes = pseudo_tail_nodes = np.concatenate(
            (tail_nodes, head_nodes)
        )
        idx = np.argsort(pseudo_head_nodes)
        self._sorted_pseudo_heads = sorted_pseudo_heads = pseudo_head_nodes[
            idx
        ]  # noqa: E501
        self._sorted_pseudo_tails = pseudo_tail_nodes[idx]
        self._sorted_dupli_links = dupli_links[idx]

        # array of start and end indexes of the pseudo head id in the array of
        # link infos.
        # start is at array[0, :] and end at array[1, :]
        self._head_start_end_indexes = (
            self._init_tools_funcs._get_start_end_indexes_in_sorted_array(
                sorted_pseudo_heads, nodes_n, self._max_number_of_nodes
            )
        )

        self._link_idx_sorted_by_heads = idx

        # 4. Precalculation of grid info
        ################################
        # Grid is assumed stable (constant areas and distances over time)
        # used in run_flow_directions()
        g.length_of_link
        if diagonals == "True":
            g.length_of_diagonal

    def run_flow_directions(self):
        """
        Calculates flow directions performing the priority-flood algorithms.
        Most of the code is delegated to cython functions in files of the ext
        directory. We implement Barnes et al., 2014's algorithm #4, adapted to
        the landlab framework and grids and to cython optimization:

        Algorithm
        = = = = =
        Numbers are steps described in Barnes et al., 2014's algorithm. Some
        minor adaptations are not indicated here, refer to the code.

        With Elevations, Directions:
        1: Let To_do be a total order priority queue
        2: Let Done have the same dimensions as Elevations
        3: Let Done be initialized to FALSE
        4: for all Nodes on the open boundaries of Elevations do:
        5:   Push Node onto To_do with priority Elevations(Node)
        .       (i.e. member with the lowest elevation should be popped first)
        6:   Done(Node) <- true
        7-9: Not implemented
        10:  Directions(Node) point to themselves
        11: while To_do is not empty do:
        12:   Node <- pop(To_do)
        13:   for all neighbors Neighbors of Node do:
        14:     if Done(Neighbor) then repeat loop
        15-17:  Not implemented
        18:     Directions(Neighbor) points towards Node
        19:     Closed(Neighbor) <- true
        20:     Push Neighbor onto To_do with priority Elevations(Neighbors)

        Remarks
        -------
        - strictly speaking, and for optimization reasons, this algorithm is
        .  not always steepest descent and and our adaptation favors flow to
        . the base-level nodes (open boundary nodes and one of the perimeter
        . nodes for NetworkModelGrid)ies of the grid. E.g.
        .   3	2	3	4	5
        .   3	2	4	5	3
        .   2	0	3	4	2
        .   2	2	0	2	4
        .   3	2	5	4	2
        .   top left : 4 and 2 send flow to 2, but the steepest slope for 4 is
        .   within the grid, to zero.
        - contrary to Barnes et al., 2014 and similar to the FlowDirectorD8
        .   component, we don't prioritize non-diagonal links: indeed, slopes
        .   can be higher on diagonals than on non-diagonals.
        - "depression_free__elevation", "depression__depth",
        .   "depression__outlet_node", "flood_status_code",
        .   "flow__link_to_receiver_node", "flow__receiver_node",
        .   "outlet_node", "topographic__steepest_slope" are updated here.
        - Following FlowDirectorD8, "flow__link_direction" is not updated.
        - steepest_slope is not corrected in case of a depression (in theory
        .   should be lower because we flood the depression to pass the flow).
        - following the FlowDirectorSteepest and D8 (and
        .   DepressionFinderAndRouter), we set closed boundary nodes as their
        .   own receivers.
        - base-level nodes = open boundary nodes, except for NetworkModelGrid.
        - no flux from and into closed boundary nodes, no flux from open
        .   boundary nodes.

        Examples
        --------
        HexModelGrid

        >>> # Libraries
        >>> import numpy as np
        >>> from landlab import HexModelGrid
        >>> #from landlab.components import FlowRouter
        >>> # Creation of the grid
        >>> params = {"shape": (7, 4), "spacing": 10, "node_layout": "hex"}
        >>> g = HexModelGrid(**params)
        >>> # Closure of the bottom edge
        >>> g.status_at_node[g.nodes_at_bottom_edge] = g.BC_NODE_IS_CLOSED
        >>> # Generation ot the topography
        >>> random_generator = np.random.Generator(np.random.PCG64(seed=500))
        >>> z = g.add_field("topographic__elevation",
        ...     10. * random_generator.random(g.number_of_nodes))
        >>> # Creation of the router
        >>> router_params = {"grid": g}
        >>> router = FlowRouter(**router_params)
        >>> # Run of the router
        >>> router.run_flow_directions()

        Depression-free surface that can be used by a multiple flow director
        component
        >>> g.at_node["depression_free__elevation"]
        array([ 5.66743143,  8.53977988,  6.45357199,  4.11156813, 4.68031945,
            8.21361221,  5.57024205,  7.8622079 ,  4.35550157,  2.47630211,
            4.42145537,  8.96951584,  5.57024205,  5.28644224,  2.75982042,
            8.74862188,  2.59199246,  8.05920356,  4.31320515,  4.31320515,
            3.49732191,  5.16016994,  3.69042619,  4.72153783,  4.31320515,
            4.31320515,  4.9459002 ,  2.71301982,  4.8779288 ,  6.39422443,
            9.37963598,  8.29564356,  6.35885287,  6.11800132,  9.41483585,
            8.8676027 ,  0.46845509])

        Receivers ordered by the node id of the donors, used to calculate flow
        accumulation
        >>> g.at_node["flow__receiver_node"]
        array([ 0,  1,  2,  3,  4, 10, 12,  8,  8,  9,  9, 18, 19, 14, 14, 15,
            9,
           16, 19, 20, 27, 21, 22, 16, 18, 19, 27, 27, 28, 24, 24, 36, 32, 33,
           34, 35, 36])

        Flooded nodes that can be used to prevent river incision in lakes (by
        the Space component, for instance)
        Flooded nodes are coded 3
        >>> g.at_node["flood_status_code"]
        array([0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0,
            0, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        Depths of depressions
        >>> g.at_node["depression__depth"]
        array([ 0.        ,  0.        ,  0.        ,  0.        ,  0.      ,
            0.        ,  2.44501033,  0.        ,  0.        ,  0.        ,
            0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
            0.        ,  0.        ,  0.        ,  3.1045881 ,  0.        ,
            0.        ,  0.        ,  0.        ,  0.        ,  2.37721033,
            1.8170932 ,  0.        ,  0.        ,  0.        ,  0.        ,
            0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
            0.        ,  0.        ])

        Set up a external water influx if none
        >>> g.at_node["water__unit_flux_in"]
        array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1., 1.,
            1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
            1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.])

        Steepest slopes at nodes, which can be used by a landslider component
        >>> g.at_node["topographic__steepest_slope"]
        array([ 0.        ,  0.        ,  0.        ,  0.        ,  0.     ,
             0.37921568,  0.        ,  0.35067063,  0.        ,  0.        ,
             0.19451533,  0.77608988,  0.12570369,  0.25266218,  0.        ,
             0.        ,  0.01156903,  0.54672111,  0.        ,  0.08158832,
             0.07843021,  0.        ,  0.        ,  0.21295454,  0.07273778,
             0.        ,  0.22328804,  0.        ,  0.        ,  0.44582296,
             0.74436412,  0.78271885,  0.        ,  0.        ,  0.        ,
             0.        ,  0.        ])
        """

        # 1. Get the input grid data (steps #4 and #11)
        ##############################################
        g = self._grid
        z = g.at_node[self._surface]
        diagonals = self._diagonals
        idx = self._link_idx_sorted_by_heads

        nodes_n = g.number_of_nodes
        base_level_nodes = self._base_level_nodes
        closed_nodes = self._closed_nodes
        neighbors_max_number = self._neighbors_max_number
        min_elevation_relative_diff = self._min_elevation_relative_diff

        gradients = (
            g.calc_grad_at_d8(z) if diagonals else g.calc_grad_at_link(z)
        )  # slopes
        sorted_dupli_gradients = np.abs(np.concatenate((gradients, gradients)))[idx]

        """start and end indexes of the pseudo head id in the array of link
        infos. Remember, the sorted_pseudo_tails are ordered by the
        _sorted_pseudo_heads of the link"""
        sorted_pseudo_tails = self._sorted_pseudo_tails
        sorted_dupli_links = self._sorted_dupli_links
        head_start_end_indexes = self._head_start_end_indexes
        # adjacent = self._adjacent

        # 2. Instantiate the output grid data (Steps #2-3)
        ##################################################
        if self._single_flow:
            receivers = g.at_node["flow__receiver_node"]
            receivers[:] = g.BAD_INDEX

            steepest_slopes = g.at_node["topographic__steepest_slope"]
            steepest_slopes[:] = 0.0
            links_to_receivers = g.at_node["flow__link_to_receiver_node"]
            links_to_receivers[:] = g.BAD_INDEX

            outlet_nodes = g.at_node["outlet_node"]
            outlet_nodes[:] = g.BAD_INDEX
        else:
            receivers = g.BAD_INDEX * np.ones(nodes_n, dtype=int)
            steepest_slopes = np.zeros(nodes_n, dtype=float)
            links_to_receivers = g.BAD_INDEX * np.ones(nodes_n, dtype=int)
            outlet_nodes = g.BAD_INDEX * np.ones(nodes_n, dtype=int)

        flooded_nodes = g.at_node["flood_status_code"]
        flooded_nodes[:] = FloodStatus._UNFLOODED
        depression_depths = g.at_node["depression__depth"]
        depression_depths[:] = 0.0
        depression_free_elevations = g.at_node["depression_free__elevation"]
        depression_free_elevations[:] = z.copy()
        depression_outlet_nodes = g.at_node["depression__outlet_node"]
        depression_outlet_nodes[:] = g.BAD_INDEX

        # 3. Flow direction process (Steps #11 - 20)
        ############################################

        self._breach_funcs._direct_flow(
            nodes_n,
            base_level_nodes,
            closed_nodes,
            sorted_pseudo_tails,
            sorted_dupli_gradients,
            sorted_dupli_links,
            head_start_end_indexes,
            outlet_nodes,
            depression_outlet_nodes,
            flooded_nodes,
            depression_depths,
            depression_free_elevations,
            links_to_receivers,
            receivers,
            steepest_slopes,
            z,
            FloodStatus._FLOODED.value,
            g.BAD_INDEX,
            neighbors_max_number=neighbors_max_number,
            min_elevation_relative_diff=min_elevation_relative_diff,
        )

    def run_flow_accumulations(self):
        """
        Flow accumulation: Calculation of drainage area and water discharge
        Following Braun and Willett, 2013 algorithm.

        Hypothesis: base level nodes are nodes that are their own receiver.
        In the following, we consider open boundary nodes are base level of
        each watershed.
        0 - association nodes (i) to receivers r(i) previously done with flow
        .   direction algorithm
        1 - tranformation to get the donors D(i,j)
        - Calculate the number of donors d_i for each node
        - Calculate the index delta_i where donor list begins for each node

        "drainage_area", "flow__upstream_node_order",
        "surface_water__discharge" are updated here.

        Examples
        --------
        HexModelGrid

        >>> # Libraries
        >>> import numpy as np
        >>> from landlab import HexModelGrid
        >>> #from landlab.components import FlowRouter
        >>> # Creation of the grid
        >>> params = {"shape": (7, 4), "spacing": 10, "node_layout": "hex"}
        >>> g = HexModelGrid(**params)
        >>> # Closure of the bottom edge
        >>> g.status_at_node[g.nodes_at_bottom_edge] = g.BC_NODE_IS_CLOSED
        >>> # Generation ot the topography
        >>> random_generator = np.random.Generator(np.random.PCG64(seed=500))
        >>> z = g.add_field("topographic__elevation",
        ...     10. * random_generator.random(g.number_of_nodes))
        >>> # Creation of the router
        >>> router_params = {"grid": g, "runoff_rate": 2.}
        >>> router = FlowRouter(**router_params)
        >>> # Run of the router
        >>> router.run_flow_directions()
        >>> router.run_flow_accumulations()

        Generates an array of nodes ordered from downstream to upstream:
        >>> g.at_node["flow__upstream_node_order"]
        array([ 4,  8,  7,  9, 16, 17, 23, 10,  5, 14, 13, 15, 21, 22, 27,
            26, 20,
            19, 25, 18, 11, 24, 29, 30, 12,  6, 28, 32, 33, 34, 35, 36, 31,
            0, 1,  2,  3])

        Calculates drainage areas
        >>> g.at_node["drainage_area"]
        array([   0.      ,    0.      ,    0.      ,    0.      ,    0.   ,
              86.60254 ,   86.60254 ,   86.60254 ,   86.60254 ,  433.0127  ,
             173.20508 ,   86.60254 ,  173.20508 ,   86.60254 ,   86.60254 ,
               0.      ,  259.80762 ,   86.60254 ,  433.01271 ,  779.42287 ,
             866.02541 ,    0.      ,    0.      ,   86.60254 ,  259.80763 ,
              86.60254 ,   86.60254 ,  952.62795 ,    0.      ,   86.602545,
              86.602545,   86.602545,    0.      ,    0.      ,
              0.      ,
               0.      ,   86.602545])

        Calculates discharges
        >>> g.at_node["surface_water__discharge"]
        array([    0.     ,     0.     ,     0.     ,     0.     ,     0.  ,
             173.20508,   173.20508,   173.20508,   173.20508,   866.0254 ,
             346.41016,   173.20508,   346.41016,   173.20508,   173.20508,
               0.     ,   519.61524,   173.20508,   866.02542,  1558.84574,
            1732.05082,     0.     ,     0.     ,   173.20508,   519.61526,
             173.20508,   173.20508,  1905.2559 ,     0.     ,   173.20509,
             173.20509,   173.20509,     0.     ,     0.     ,     0.     ,
               0.     ,   173.20509])
        """

        if not self._single_flow:
            return

        g = self._grid
        nodes_n = g.number_of_nodes
        nodes = self._nodes
        base_level_and_closed_nodes = self._base_level_and_closed_nodes
        receivers = g.at_node["flow__receiver_node"]
        upstream_ordered_nodes = g.at_node["flow__upstream_node_order"]

        # number of donors for each node d_i
        donors_n_by_node = np.bincount(receivers, minlength=nodes_n)

        """start index of the list of donors for each node delta_i.
        Note that donors_start_indexes has a size of nodes_n + 1 to avoid out
        of boundary error in _add_to_upstream_ordered_nodes when accessing
        donors_start_indexes[receiver_id + 1]."""
        donors_start_indexes = np.cumsum(
            np.concatenate(([0], donors_n_by_node))
        )  # noqa: E501

        """D, in python, this way is 10xtimes faster than the way of updating
        with index handling array of nodes sorted from downstream to upstream
        (construction of a stack)."""
        donors = nodes[receivers.argsort()]

        """ the stack. Beware to set values use [:] otherwise
        upstream_ordered_nodes will point to a new array and not
        g.at_node["flow__upstream_node_order"].
        Comment: I don't know why
        g.at_node["flow__upstream_node_order"][:] = -1 triggers an error in
        _calc_upstream_order_for_nodes."""
        upstream_ordered_nodes[:] = np.full(
            nodes_n, g.BAD_INDEX, dtype=np.int_
        )  # noqa: E501

        # Call to the algorithm.
        self._accumulation_funcs._calc_upstream_order_for_nodes(
            base_level_and_closed_nodes,
            upstream_ordered_nodes,
            donors_start_indexes,
            donors,
        )

        # Calculation of drainage areas and water discharges
        cell_area_at_nodes = (
            self._cell_area_at_nodes
        )  # areas = 0 in boundary because there are no cells
        water_external_influxes = g.at_node["water__unit_flux_in"]
        downstream_ordered_nodes = upstream_ordered_nodes[::-1]
        drainage_areas = cell_area_at_nodes.copy()
        discharges = np.full(
            shape=nodes_n,
            fill_value=cell_area_at_nodes * water_external_influxes,  # noqa: E501
        )

        self._accumulation_funcs._calc_drainage_areas(
            downstream_ordered_nodes, receivers, drainage_areas
        )
        if not self._uniform_water_external_influx:
            discharges = drainage_areas.copy() * water_external_influxes
        else:
            discharges = drainage_areas.copy() * water_external_influxes[0]

        g.at_node["drainage_area"] = drainage_areas
        g.at_node["surface_water__discharge"] = discharges
        # At boundary, drainage and discharge are not forced to 0 but can be 0
        # (because there cell area = 0)

    def run_one_step(self):
        """Calculates flow directions and accumulations
        (if the component is set to accumulate_flow=True)

        Examples
        --------
        HexModelGrid

        >>> # Libraries
        >>> import numpy as np
        >>> from landlab import HexModelGrid
        >>> #from landlab.components import FlowRouter
        >>> # Creation of the grid
        >>> params = {"shape": (7, 4), "spacing": 10, "node_layout": "hex"}
        >>> g = HexModelGrid(**params)
        >>> # Closure of the bottom edge
        >>> g.status_at_node[g.nodes_at_bottom_edge] = g.BC_NODE_IS_CLOSED
        >>> # Generation ot the topography
        >>> random_generator = np.random.Generator(np.random.PCG64(seed=500))
        >>> z = g.add_field("topographic__elevation",
        ...     10. * random_generator.random(g.number_of_nodes))
        >>> # Creation of the router
        >>> router_params = {"grid": g, "runoff_rate": 2.}
        >>> router = FlowRouter(**router_params)
        >>> # Run of the router
        >>> router.run_one_step()

        Calculates discharges (and all the fields described in
        run_flow_directions() and run_flow_accumulations()
        >>> g.at_node["surface_water__discharge"]
        array([    0.     ,     0.     ,     0.     ,     0.     ,     0.   ,
             173.20508,   173.20508,   173.20508,   173.20508,   866.0254 ,
             346.41016,   173.20508,   346.41016,   173.20508,   173.20508,
               0.     ,   519.61524,   173.20508,   866.02542,  1558.84574,
            1732.05082,     0.     ,     0.     ,   173.20508,   519.61526,
             173.20508,   173.20508,  1905.2559 ,     0.     ,   173.20509,
             173.20509,   173.20509,     0.     ,     0.     ,     0.     ,
               0.     ,   173.20509])
        """
        self.run_flow_directions()
        self.run_flow_accumulations()
