
COLORADO RIVER NETWORK PROJECT
==============================

Contents
--------

graphs/
    master_compressed_river_graph.pkl
        Main compressed NetworkX river graph used for analysis.

    largest_river_component.pkl
        Detailed river-network graph before compression.
        This file is optional but is useful for plotting the
        actual river geometry.

data/
    master_wq_station_table.csv
        Water-quality stations and associated metadata.

    usgs_cleaning_audit.csv
        Audit of downloaded/cleaned USGS series.

    usgs_all_stations_hourly.csv
        Combined cleaned hourly USGS observations.

    usgs_station_variable_completeness.csv
        Data completeness by station and variable.

exports/
    graph_nodes.csv
        Portable table of graph nodes.

    graph_edges.csv
        Portable table of directed graph edges.

notebooks/
    Interactive_River_Network_Map.ipynb
        Interactive Folium map.

    USGS_Data_Cleaning.ipynb
        USGS data-cleaning workflow.


LOADING THE GRAPH
-----------------

Python example:

    import pickle

    with open(
        "graphs/master_compressed_river_graph.pkl",
        "rb"
    ) as f:
        G = pickle.load(f)

    print(G.number_of_nodes())
    print(G.number_of_edges())
    print(G.is_directed())


IMPORTANT
---------

Only load pickle files from trusted sources.

The interactive-map notebook may request a CARTO
Basemaps API key. API keys are intentionally NOT
included in this package.

The graph is directed according to river flow.

Water-quality station information is stored primarily
in the "usgs_clusters" node attribute.

Hydrology-support gauges are stored in the
"support_gauges" node attribute.
