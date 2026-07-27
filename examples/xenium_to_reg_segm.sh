python assign_cells.py --xenium_bundle_path "/lustre/scratch127/cellgen/cellgeni/public_datasets/Atera/WTA_Preview_FFPE_Breast_Cancer_outs/" \
                        --image_path "/lustre/scratch127/cellgen/cellgeni/public_datasets/Atera//WTA_Preview_FFPE_Breast_Cancer_outs_supplemental_files/WTA_Preview_FFPE_Breast_Cancer_he_image.ome.tif" \
                        --polygons_parquet_path "/lustre/scratch127/cellgen/cellgeni/tickets/REQ-70619/actions/stardist_he_human_breast/WTA_Preview_FFPE_Breast_Cancer_stardist_he_human_breast_polygons_fullcoords.parquet" \
                        --transform_csv_path "/lustre/scratch127/cellgen/cellgeni/tickets/REQ-70619/actions/WTA_Preview_FFPE_Breast_Cancer_he_alignment.csv" \
                        --output_matches_csv "output_test/matched_cells_test1.csv" \
                        --he_coordinate_units "auto" \
                        --invert_matrix True \
                        --max_distance_um 10
