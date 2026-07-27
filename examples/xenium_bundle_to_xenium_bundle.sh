python assign_cells.py --xenium_bundle_path '/path/to/output-XETG00055__0032785__AX5-SKI-0-FFPE-1-S10__20241115__150230' \
                        --image_path '/path/to/resegmented_xenium_bundle_cellpose/output-XETG00055__0032785__AX5-SKI-0-FFPE-1-S10__20241115__150230/morphology.ome.tif' \
                        --polygons_parquet_path '/path/to/resegmented_xenium_bundle_cellpose/output-XETG00055__0032785__AX5-SKI-0-FFPE-1-S10__20241115__150230/cell_boundaries.parquet' \
                        --output_matches_csv "output_test/matched_cells_test2.csv" \
                        --he_coordinate_units "auto" \
                        --invert_matrix True \
                        --max_distance_um 10
