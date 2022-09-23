#!/bin/zsh

cd extracts;
panoptes_aggregation extract ../circle-the-vortex-classifications.csv \
	../configs/Extractor_config_workflow_21832_V1.1.yaml;

cd ../reductions;
panoptes_aggregation reduce ../extracts/shape_extractor_ellipse_extractions.csv \
	../configs/Reducer_config_workflow_21832_V1.1_shape_extractor_ellipse.yaml -c 12;

cd ..
# aggregate and convert to JSON
python3 scripts/create_JSON_output.py

# create vortex clusters
python3 scripts/find_vortex_clusters.py
