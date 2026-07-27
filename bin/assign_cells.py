
from __future__ import annotations

import colorsys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from shapely import wkb, wkt
from shapely.geometry import Polygon, MultiPolygon
from shapely.geometry.base import BaseGeometry

from matplotlib.collections import LineCollection
from scipy.spatial import cKDTree
import fire


PathLike = Union[str, Path]
PixelSize = Union[float, Tuple[float, float]]


# =====================================================================
# Main function 1
# =====================================================================

def assign_cells(
    xenium_bundle_path: PathLike,
    polygons_parquet_path: PathLike,
    image_path: PathLike,
    transform_csv_path: PathLike = None,
    *,
    max_distance_um: float = 10.0,
    xenium_boundary_type: str = "cell",
    invert_matrix: bool = True,
    coordinate_convention: str = "column",
    he_coordinate_units: str = "auto",
    he_cell_id_col: Optional[str] = None,
    he_x_col: Optional[str] = None,
    he_y_col: Optional[str] = None,
    output_matches_csv: Optional[PathLike] = None,
    output_unmatched_xenium_txt: Optional[PathLike] = None,
    output_unmatched_he_txt: Optional[PathLike] = None,
    print_diagnostics: bool = True,
    return_results: bool = False,
):
    """
    Match Xenium cells to cells segmented directly from an H&E image.

    Matching is based on transformed polygon centroids.

    Xenium coordinates are transformed into full-resolution H&E pixel
    coordinates using:

        1. the supplied affine matrix;
        2. automatic inversion, if requested;
        3. the pixel size encoded in the original matrix;
        4. the full-resolution H&E pixel size from OME metadata.

    A global greedy one-to-one assignment is then performed. Candidate
    pairs within `max_distance_um` are sorted by distance, and the closest
    available pairs are selected first.

    Parameters
    ----------
    xenium_bundle_path
        Path to the Xenium output bundle containing cell boundaries.

    polygons_parquet_path
        Parquet file containing H&E segmentation polygon vertices.

        Expected structure is one row per polygon vertex, with columns
        similar to:

            cell_id
            vertex_x
            vertex_y

        Other common column names are detected automatically.

    image_path
        Path to the full-resolution OME-TIFF H&E image.

        This is used to read:
            - image dimensions;
            - PhysicalSizeX;
            - PhysicalSizeY.

    transform_csv_path
        CSV containing the 3x3 affine transformation matrix.

    max_distance_um
        Maximum allowed centroid-to-centroid distance in micrometres.

        Default: 10 µm.

    xenium_boundary_type
        Either "cell" or "nucleus".

    invert_matrix
        Usually True for matrices mapping H&E pixels to Xenium
        micrometres, because the required plotting direction is:

            Xenium micrometres -> H&E pixels

    coordinate_convention
        "column" means:

            transformed = matrix @ [x, y, 1].T

        "row" means:

            transformed = [x, y, 1] @ matrix

    he_coordinate_units
        Units of the H&E segmentation polygon coordinates:

            "auto"
            "pixels"
            "microns"

        With "auto", the coordinate range is compared with the H&E image
        pixel dimensions and physical dimensions.

    he_cell_id_col, he_x_col, he_y_col
        Optional explicit H&E polygon column names.

    output_matches_csv
        Optional CSV path for the matched cell table.

    output_unmatched_xenium_txt
        Optional text path containing one unmatched Xenium cell ID per line.

    output_unmatched_he_txt
        Optional text path containing one unmatched H&E cell ID per line.

    Returns
    -------
    matches_df
        DataFrame with:

            cell_id_xenium
            cell_id_he
            distance_um

        The first two columns are the requested mapping. Distance is included
        because it is useful for quality control.

    unmatched_xenium_ids
        List of Xenium cell IDs without an assigned H&E cell.

    unmatched_he_ids
        List of H&E cell IDs without an assigned Xenium cell.

    diagnostics
        Dictionary containing metadata and transformed centroid tables.
    """

    xenium_bundle_path = Path(xenium_bundle_path)
    polygons_parquet_path = Path(polygons_parquet_path)
    image_path = Path(image_path)

    if transform_csv_path: transform_csv_path = Path(transform_csv_path)

    _require_existing_path(
        xenium_bundle_path,
        "Xenium bundle",
    )
    _require_existing_path(
        polygons_parquet_path,
        "H&E polygon Parquet file",
    )
    _require_existing_path(
        image_path,
        "H&E image",
    )
    

    if max_distance_um <= 0:
        raise ValueError("max_distance_um must be greater than zero.")

    coordinate_convention = coordinate_convention.lower()

    if coordinate_convention not in {"column", "row"}:
        raise ValueError(
            "coordinate_convention must be 'column' or 'row'."
        )

    # ---------------------------------------------------------------
    # H&E metadata
    # ---------------------------------------------------------------
    he_metadata = read_ome_image_metadata(image_path)

    display_pixel_size_um = (
        he_metadata["pixel_size_x_um"],
        he_metadata["pixel_size_y_um"],
    )

    image_width = he_metadata["width_px"]
    image_height = he_metadata["height_px"]

    # ---------------------------------------------------------------
    # Transformation matrix
    # ---------------------------------------------------------------
    if transform_csv_path:
        original_matrix = read_3x3_matrix(transform_csv_path)
    else:
        original_matrix = np.array([[1,0,0], [0,1,0], [0,0,1]]) 
    matrix_pixel_size_um = estimate_matrix_pixel_size_um(
        original_matrix,
        coordinate_convention=coordinate_convention,
    )

    matrix_used = original_matrix.copy()

    if invert_matrix:
        try:
            matrix_used = np.linalg.inv(matrix_used)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "The supplied transformation matrix is singular and "
                "cannot be inverted."
            ) from exc

    # ---------------------------------------------------------------
    # Xenium polygons
    # ---------------------------------------------------------------
    xenium_boundary_path = find_xenium_boundary_file(
        xenium_bundle_path,
        boundary_type=xenium_boundary_type,
    )

    print(f"Reading Xenium polygons: {xenium_boundary_path}")

    xenium_polygons = read_polygon_table(xenium_boundary_path)

    (
        xenium_id_col,
        xenium_x_col,
        xenium_y_col,
    ) = detect_polygon_columns(
        xenium_polygons,
        id_candidates=[
            "cell_id",
            "nucleus_id",
            "cellid",
            "barcode",
            "cell",
        ],
        x_candidates=[
            "vertex_x",
            "x",
            "x_location",
        ],
        y_candidates=[
            "vertex_y",
            "y",
            "y_location",
        ],
    )

    xenium_polygons = clean_polygon_table(
        xenium_polygons,
        id_col=xenium_id_col,
        x_col=xenium_x_col,
        y_col=xenium_y_col,
    )

    xenium_xy_um = xenium_polygons[
        [xenium_x_col, xenium_y_col]
    ].to_numpy(dtype=np.float64)

    xenium_xy_matrix_pixels = apply_affine_transform(
        xenium_xy_um,
        matrix_used,
        convention=coordinate_convention,
    )

    xenium_xy_he_pixels = correct_matrix_resolution(
        transformed_xy=xenium_xy_matrix_pixels,
        matrix_used=matrix_used,
        matrix_image_pixel_size_um=matrix_pixel_size_um,
        display_image_pixel_size_um=display_pixel_size_um,
        coordinate_convention=coordinate_convention,
    )

    xenium_polygons["x_he_pixel"] = xenium_xy_he_pixels[:, 0]
    xenium_polygons["y_he_pixel"] = xenium_xy_he_pixels[:, 1]

    xenium_polygons = xenium_polygons.loc[
        np.isfinite(xenium_polygons["x_he_pixel"])
        & np.isfinite(xenium_polygons["y_he_pixel"])
    ].copy()

    xenium_centroids = calculate_polygon_centroids(
        polygons=xenium_polygons,
        id_col=xenium_id_col,
        x_col="x_he_pixel",
        y_col="y_he_pixel",
        output_id_col="cell_id_xenium",
    )

    # ---------------------------------------------------------------
    # H&E polygons
    # ---------------------------------------------------------------
    print(f"Reading H&E polygons: {polygons_parquet_path}")

    he_polygons_raw = pd.read_parquet(
        polygons_parquet_path
    )

    (
        he_polygons,
        he_cell_id_col,
        he_x_col,
        he_y_col,
    ) = prepare_he_polygon_vertices(
        he_table=he_polygons_raw,
        cell_id_col=he_cell_id_col,
        x_col=he_x_col,
        y_col=he_y_col,
    )

    resolved_he_units = infer_he_coordinate_units(
        he_polygons=he_polygons,
        x_col=he_x_col,
        y_col=he_y_col,
        image_width_px=image_width,
        image_height_px=image_height,
        pixel_size_x_um=display_pixel_size_um[0],
        pixel_size_y_um=display_pixel_size_um[1],
        requested_units=he_coordinate_units,
    )

    if resolved_he_units == "microns":
        he_polygons["x_he_pixel"] = (
            he_polygons[he_x_col] / display_pixel_size_um[0]
        )
        he_polygons["y_he_pixel"] = (
            he_polygons[he_y_col] / display_pixel_size_um[1]
        )
    else:
        he_polygons["x_he_pixel"] = he_polygons[he_x_col]
        he_polygons["y_he_pixel"] = he_polygons[he_y_col]

    he_centroids = calculate_polygon_centroids(
        polygons=he_polygons,
        id_col=he_cell_id_col,
        x_col="x_he_pixel",
        y_col="y_he_pixel",
        output_id_col="cell_id_he",
    )

    # ---------------------------------------------------------------
    # Remove clearly out-of-image centroids
    # ---------------------------------------------------------------
    xenium_centroids_in_image = xenium_centroids.loc[
        (xenium_centroids["centroid_x_px"] >= 0)
        & (xenium_centroids["centroid_x_px"] < image_width)
        & (xenium_centroids["centroid_y_px"] >= 0)
        & (xenium_centroids["centroid_y_px"] < image_height)
    ].copy()

    he_centroids_in_image = he_centroids.loc[
        (he_centroids["centroid_x_px"] >= 0)
        & (he_centroids["centroid_x_px"] < image_width)
        & (he_centroids["centroid_y_px"] >= 0)
        & (he_centroids["centroid_y_px"] < image_height)
    ].copy()

    # ---------------------------------------------------------------
    # Match one-to-one in physical coordinates
    # ---------------------------------------------------------------
    matches_df = greedy_one_to_one_centroid_matching(
        xenium_centroids=xenium_centroids_in_image,
        he_centroids=he_centroids_in_image,
        pixel_size_um=display_pixel_size_um,
        max_distance_um=max_distance_um,
    )

    all_xenium_ids = xenium_centroids["cell_id_xenium"].tolist()
    all_he_ids = he_centroids["cell_id_he"].tolist()

    matched_xenium_ids = set(matches_df["cell_id_xenium"])
    matched_he_ids = set(matches_df["cell_id_he"])

    unmatched_xenium_ids = [
        cell_id
        for cell_id in all_xenium_ids
        if cell_id not in matched_xenium_ids
    ]

    unmatched_he_ids = [
        cell_id
        for cell_id in all_he_ids
        if cell_id not in matched_he_ids
    ]

    # ---------------------------------------------------------------
    # Save outputs
    # ---------------------------------------------------------------
    if output_matches_csv is not None:
        output_matches_csv = Path(output_matches_csv)
        output_matches_csv.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        matches_df.to_csv(
            output_matches_csv,
            index=False,
        )

        print(f"Saved matches: {output_matches_csv}")

    if output_unmatched_xenium_txt is not None:
        write_id_list(
            unmatched_xenium_ids,
            output_unmatched_xenium_txt,
        )

    if output_unmatched_he_txt is not None:
        write_id_list(
            unmatched_he_ids,
            output_unmatched_he_txt,
        )

    if print_diagnostics:
        print("\nMatching summary")
        print("=" * 70)
        print(
            f"H&E image size: "
            f"{image_width:,} x {image_height:,} pixels"
        )
        print(
            f"H&E pixel size: "
            f"x={display_pixel_size_um[0]:.6f}, "
            f"y={display_pixel_size_um[1]:.6f} µm/pixel"
        )
        print(
            f"Matrix-reference pixel size: "
            f"x={matrix_pixel_size_um[0]:.6f}, "
            f"y={matrix_pixel_size_um[1]:.6f} µm/pixel"
        )
        print(f"H&E polygon units: {resolved_he_units}")
        print(
            f"Xenium cells: {len(all_xenium_ids):,}"
        )
        print(
            f"H&E cells: {len(all_he_ids):,}"
        )
        print(
            f"Matched pairs: {len(matches_df):,}"
        )
        print(
            f"Unmatched Xenium cells: "
            f"{len(unmatched_xenium_ids):,}"
        )
        print(
            f"Unmatched H&E cells: "
            f"{len(unmatched_he_ids):,}"
        )

        if not matches_df.empty:
            print(
                f"Median matched distance: "
                f"{matches_df['distance_um'].median():.3f} µm"
            )
            print(
                f"Maximum matched distance: "
                f"{matches_df['distance_um'].max():.3f} µm"
            )

        print("=" * 70)

    diagnostics = {
        "he_metadata": he_metadata,
        "display_pixel_size_um": display_pixel_size_um,
        "matrix_pixel_size_um": matrix_pixel_size_um,
        "original_matrix": original_matrix,
        "matrix_used": matrix_used,
        "he_coordinate_units": resolved_he_units,
        "xenium_boundary_path": xenium_boundary_path,
        "xenium_polygons_transformed": xenium_polygons,
        "he_polygons": he_polygons,
        "xenium_centroids": xenium_centroids,
        "he_centroids": he_centroids,
    }

    if return_results:
        return (
            matches_df,
            unmatched_xenium_ids,
            unmatched_he_ids,
            diagnostics,
        )

    return None


# =====================================================================
# Main function 2
# =====================================================================

def plot_matched_xenium_he_polygons(
    image: np.ndarray,
    matches_df: pd.DataFrame,
    xenium_bundle_path: PathLike,
    polygons_parquet_path: PathLike,
    image_path: PathLike,
    transform_csv_path: PathLike,
    x_bounds: Tuple[float, float],
    y_bounds: Tuple[float, float],
    *,
    xenium_boundary_type: str = "cell",
    invert_matrix: bool = True,
    coordinate_convention: str = "column",
    he_coordinate_units: str = "auto",
    he_cell_id_col: Optional[str] = None,
    he_x_col: Optional[str] = None,
    he_y_col: Optional[str] = None,
    channels: Optional[Union[int, Sequence[int]]] = None,
    channel_axis: Optional[int] = None,
    channel_percentiles: Tuple[float, float] = (1.0, 99.8),
    channel_weights: Optional[Sequence[float]] = None,
    he_linewidth: float = 1.5,
    xenium_linewidth: float = 1.5,
    unmatched_linewidth: float = 0.8,
    matched_alpha: float = 0.95,
    unmatched_alpha: float = 0.7,
    he_linestyle: Union[str, Tuple] = "solid",
    xenium_linestyle: Union[str, Tuple] = "dashed",
    unmatched_color: str = "white",
    random_seed: int = 0,
    figsize: Tuple[float, float] = (12, 12),
    image_alpha: float = 1.0,
    background: str = "black",
    title: Optional[str] = None,
    save_path: Optional[PathLike] = None,
    dpi: int = 300,
):
    """
    Plot a crop of the H&E image with paired segmentation outlines.

    Matched H&E and Xenium polygons receive the same random colour.

    H&E outlines:
        solid by default.

    Xenium outlines:
        dashed by default.

    Unmatched polygons:
        white by default.

    Parameters
    ----------
    image
        Already-loaded H&E image array.

        Supported forms:

            (Y, X)
            (C, Y, X)
            (Y, X, C)

    matches_df
        Output matching table from `match_xenium_to_he_cells()`.

        Required columns:

            cell_id_xenium
            cell_id_he

    x_bounds, y_bounds
        Full-resolution H&E pixel coordinates defining the crop.

    save_path
        Optional output path, such as:

            "matched_overlay.png"
            "matched_overlay.pdf"

    Returns
    -------
    fig
        Matplotlib Figure.

    ax
        Matplotlib Axes.

    plotted_he_polygons
        H&E polygon vertices intersecting the crop.

    plotted_xenium_polygons
        Transformed Xenium polygon vertices intersecting the crop.
    """

    required_match_columns = {
        "cell_id_xenium",
        "cell_id_he",
    }

    missing_match_columns = (
        required_match_columns - set(matches_df.columns)
    )

    if missing_match_columns:
        raise ValueError(
            "matches_df is missing columns: "
            f"{sorted(missing_match_columns)}"
        )

    image = np.asarray(image)

    xenium_bundle_path = Path(xenium_bundle_path)
    polygons_parquet_path = Path(polygons_parquet_path)
    image_path = Path(image_path)
    transform_csv_path = Path(transform_csv_path)

    # ---------------------------------------------------------------
    # Image dimensions and crop
    # ---------------------------------------------------------------
    (
        image_height,
        image_width,
        resolved_channel_axis,
    ) = get_image_spatial_shape(
        image,
        channel_axis=channel_axis,
    )

    x_min, x_max = map(float, x_bounds)
    y_min, y_max = map(float, y_bounds)

    if x_min >= x_max:
        raise ValueError("x_bounds must satisfy x_min < x_max.")

    if y_min >= y_max:
        raise ValueError("y_bounds must satisfy y_min < y_max.")

    crop_x0 = max(0, int(np.floor(x_min)))
    crop_x1 = min(image_width, int(np.ceil(x_max)))

    crop_y0 = max(0, int(np.floor(y_min)))
    crop_y1 = min(image_height, int(np.ceil(y_max)))

    if crop_x0 >= crop_x1 or crop_y0 >= crop_y1:
        raise ValueError(
            "The requested crop does not overlap the image."
        )

    raw_crop = crop_image(
        image=image,
        crop_x0=crop_x0,
        crop_x1=crop_x1,
        crop_y0=crop_y0,
        crop_y1=crop_y1,
        channel_axis=resolved_channel_axis,
    )

    display_crop = prepare_cropped_image_for_display(
        image_crop=raw_crop,
        channels=channels,
        channel_axis=resolved_channel_axis,
        percentiles=channel_percentiles,
        channel_weights=channel_weights,
    )

    # ---------------------------------------------------------------
    # Metadata and transformation
    # ---------------------------------------------------------------
    he_metadata = read_ome_image_metadata(he_image_path)

    display_pixel_size_um = (
        he_metadata["pixel_size_x_um"],
        he_metadata["pixel_size_y_um"],
    )

    
    original_matrix = read_3x3_matrix(transform_csv_path)

    matrix_pixel_size_um = estimate_matrix_pixel_size_um(
        original_matrix,
        coordinate_convention=coordinate_convention,
    )

    matrix_used = original_matrix.copy()

    if invert_matrix:
        matrix_used = np.linalg.inv(matrix_used)

    # ---------------------------------------------------------------
    # Xenium polygons and transformation
    # ---------------------------------------------------------------
    xenium_boundary_path = find_xenium_boundary_file(
        xenium_bundle_path,
        boundary_type=xenium_boundary_type,
    )

    xenium_polygons = read_polygon_table(
        xenium_boundary_path
    )

    (
        xenium_id_col,
        xenium_x_col,
        xenium_y_col,
    ) = detect_polygon_columns(
        xenium_polygons,
        id_candidates=[
            "cell_id",
            "nucleus_id",
            "cellid",
            "barcode",
            "cell",
        ],
        x_candidates=[
            "vertex_x",
            "x",
            "x_location",
        ],
        y_candidates=[
            "vertex_y",
            "y",
            "y_location",
        ],
    )

    xenium_polygons = clean_polygon_table(
        xenium_polygons,
        id_col=xenium_id_col,
        x_col=xenium_x_col,
        y_col=xenium_y_col,
    )

    xenium_xy_um = xenium_polygons[
        [xenium_x_col, xenium_y_col]
    ].to_numpy(dtype=np.float64)

    xenium_xy_matrix_pixels = apply_affine_transform(
        xenium_xy_um,
        matrix_used,
        convention=coordinate_convention,
    )

    xenium_xy_he_pixels = correct_matrix_resolution(
        transformed_xy=xenium_xy_matrix_pixels,
        matrix_used=matrix_used,
        matrix_image_pixel_size_um=matrix_pixel_size_um,
        display_image_pixel_size_um=display_pixel_size_um,
        coordinate_convention=coordinate_convention,
    )

    xenium_polygons["x_he_pixel"] = xenium_xy_he_pixels[:, 0]
    xenium_polygons["y_he_pixel"] = xenium_xy_he_pixels[:, 1]

    # ---------------------------------------------------------------
    # H&E polygons
    # ---------------------------------------------------------------
    he_polygons_raw = pd.read_parquet(polygons_parquet_path)

    (
        he_polygons,
        he_cell_id_col,
        he_x_col,
        he_y_col,
    ) = prepare_he_polygon_vertices(
        he_polygons_raw,
        cell_id_col=he_cell_id_col,
        x_col=he_x_col,
        y_col=he_y_col,
    )

    resolved_he_units = infer_he_coordinate_units(
        he_polygons=he_polygons,
        x_col=he_x_col,
        y_col=he_y_col,
        image_width_px=image_width,
        image_height_px=image_height,
        pixel_size_x_um=display_pixel_size_um[0],
        pixel_size_y_um=display_pixel_size_um[1],
        requested_units=he_coordinate_units,
    )

    if resolved_he_units == "microns":
        he_polygons["x_he_pixel"] = (
            he_polygons[he_x_col] / display_pixel_size_um[0]
        )
        he_polygons["y_he_pixel"] = (
            he_polygons[he_y_col] / display_pixel_size_um[1]
        )
    else:
        he_polygons["x_he_pixel"] = he_polygons[he_x_col]
        he_polygons["y_he_pixel"] = he_polygons[he_y_col]

    # ---------------------------------------------------------------
    # Filter polygon tables to the crop
    # ---------------------------------------------------------------
    he_polygons_crop = filter_polygons_intersecting_crop(
        polygons=he_polygons,
        id_col=he_cell_id_col,
        x_col="x_he_pixel",
        y_col="y_he_pixel",
        x_bounds=x_bounds,
        y_bounds=y_bounds,
    )

    xenium_polygons_crop = filter_polygons_intersecting_crop(
        polygons=xenium_polygons,
        id_col=xenium_id_col,
        x_col="x_he_pixel",
        y_col="y_he_pixel",
        x_bounds=x_bounds,
        y_bounds=y_bounds,
    )

    # ---------------------------------------------------------------
    # Assign pair colours
    # ---------------------------------------------------------------
    matches_unique = matches_df[
        ["cell_id_xenium", "cell_id_he"]
    ].drop_duplicates()

    pair_colors = create_distinct_random_colors(
        n_colors=len(matches_unique),
        random_seed=random_seed,
    )

    xenium_color_map = {}
    he_color_map = {}

    for row, color in zip(
        matches_unique.itertuples(index=False),
        pair_colors,
    ):
        xenium_color_map[row.cell_id_xenium] = color
        he_color_map[row.cell_id_he] = color

    # ---------------------------------------------------------------
    # Create segments
    # ---------------------------------------------------------------
    (
        he_matched_segments,
        he_matched_colors,
        he_unmatched_segments,
    ) = build_colored_polygon_segments(
        polygons=he_polygons_crop,
        id_col=he_cell_id_col,
        x_col="x_he_pixel",
        y_col="y_he_pixel",
        color_map=he_color_map,
    )

    (
        xenium_matched_segments,
        xenium_matched_colors,
        xenium_unmatched_segments,
    ) = build_colored_polygon_segments(
        polygons=xenium_polygons_crop,
        id_col=xenium_id_col,
        x_col="x_he_pixel",
        y_col="y_he_pixel",
        color_map=xenium_color_map,
    )

    # ---------------------------------------------------------------
    # Plot
    # ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)

    ax.set_facecolor(background)

    if display_crop.ndim == 2:
        ax.imshow(
            display_crop,
            cmap="gray",
            extent=(
                crop_x0,
                crop_x1,
                crop_y1,
                crop_y0,
            ),
            interpolation="nearest",
            alpha=image_alpha,
        )
    else:
        ax.imshow(
            display_crop,
            extent=(
                crop_x0,
                crop_x1,
                crop_y1,
                crop_y0,
            ),
            interpolation="nearest",
            alpha=image_alpha,
        )

    # Unmatched polygons first, so matched colours remain visible.
    if he_unmatched_segments:
        ax.add_collection(
            LineCollection(
                he_unmatched_segments,
                colors=unmatched_color,
                linewidths=unmatched_linewidth,
                linestyles=he_linestyle,
                alpha=unmatched_alpha,
                zorder=2,
            )
        )

    if xenium_unmatched_segments:
        ax.add_collection(
            LineCollection(
                xenium_unmatched_segments,
                colors=unmatched_color,
                linewidths=unmatched_linewidth,
                linestyles=xenium_linestyle,
                alpha=unmatched_alpha,
                zorder=3,
            )
        )

    if he_matched_segments:
        ax.add_collection(
            LineCollection(
                he_matched_segments,
                colors=he_matched_colors,
                linewidths=he_linewidth,
                linestyles=he_linestyle,
                alpha=matched_alpha,
                zorder=4,
            )
        )

    if xenium_matched_segments:
        ax.add_collection(
            LineCollection(
                xenium_matched_segments,
                colors=xenium_matched_colors,
                linewidths=xenium_linewidth,
                linestyles=xenium_linestyle,
                alpha=matched_alpha,
                zorder=5,
            )
        )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_max, y_min)
    ax.set_aspect("equal")

    ax.set_xlabel("H&E x coordinate [pixels]")
    ax.set_ylabel("H&E y coordinate [pixels]")

    if title is None:
        title = (
            "Matched H&E and Xenium cell boundaries\n"
            "H&E: solid; Xenium: dashed; unmatched: white"
        )

    ax.set_title(title)

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)

        save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig.savefig(
            save_path,
            dpi=dpi,
            bbox_inches="tight",
        )

        print(f"Saved overlay: {save_path}")

    print(
        f"H&E polygons in crop: "
        f"{he_polygons_crop[he_cell_id_col].nunique():,}"
    )
    print(
        f"Xenium polygons in crop: "
        f"{xenium_polygons_crop[xenium_id_col].nunique():,}"
    )
    print(
        f"Matched H&E polygons plotted: "
        f"{len(he_matched_segments):,}"
    )
    print(
        f"Matched Xenium polygons plotted: "
        f"{len(xenium_matched_segments):,}"
    )

    return (
        fig,
        ax,
        he_polygons_crop,
        xenium_polygons_crop,
    )


# =====================================================================
# Matching helpers
# =====================================================================
def prepare_he_polygon_vertices(
    he_table: pd.DataFrame,
    cell_id_col: Optional[str] = None,
    x_col: Optional[str] = None,
    y_col: Optional[str] = None,
    geometry_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, str, str, str]:
    """
    Convert an H&E segmentation table into one row per polygon vertex.

    Supports either:

        1. Explicit vertex columns:
               cell_id, vertex_x, vertex_y

        2. A geometry column containing:
               Shapely Polygon
               Shapely MultiPolygon
               WKB bytes
               hexadecimal WKB strings
               WKT strings

    Returns
    -------
    vertices
        Long-form DataFrame with one row per polygon vertex.

    cell_id_col
        Cell identifier column name.

    x_col
        Output x-coordinate column name, always "vertex_x".

    y_col
        Output y-coordinate column name, always "vertex_y".
    """
    he_table = he_table.copy()

    lower_to_original = {
        str(column).lower(): column
        for column in he_table.columns
    }

    if cell_id_col is None:
        id_candidates = [
            "cell_id",
            "cellid",
            "object_id",
            "segment_id",
            "label_id",
            "instance_id",
            "polygon_id",
            "id",
        ]

        cell_id_col = next(
            (
                lower_to_original[name]
                for name in id_candidates
                if name in lower_to_original
            ),
            None,
        )

    if cell_id_col is None:
        raise ValueError(
            "Could not identify the H&E cell ID column.\n"
            f"Columns found: {list(he_table.columns)}"
        )

    # First check whether explicit x/y columns already exist.
    if x_col is None:
        x_candidates = [
            "vertex_x",
            "x",
            "x_coord",
            "x_coordinate",
            "pixel_x",
            "x_pixel",
        ]

        x_col = next(
            (
                lower_to_original[name]
                for name in x_candidates
                if name in lower_to_original
            ),
            None,
        )

    if y_col is None:
        y_candidates = [
            "vertex_y",
            "y",
            "y_coord",
            "y_coordinate",
            "pixel_y",
            "y_pixel",
        ]

        y_col = next(
            (
                lower_to_original[name]
                for name in y_candidates
                if name in lower_to_original
            ),
            None,
        )

    if x_col is not None and y_col is not None:
        vertices = clean_polygon_table(
            he_table,
            id_col=cell_id_col,
            x_col=x_col,
            y_col=y_col,
        )

        return vertices, cell_id_col, x_col, y_col

    # Otherwise use the geometry column.
    if geometry_col is None:
        geometry_candidates = [
            "geometry",
            "geom",
            "polygon",
            "shape",
        ]

        geometry_col = next(
            (
                lower_to_original[name]
                for name in geometry_candidates
                if name in lower_to_original
            ),
            None,
        )

    if geometry_col is None:
        raise ValueError(
            "The H&E table has neither explicit x/y coordinates nor "
            "a recognised geometry column.\n"
            f"Columns found: {list(he_table.columns)}"
        )

    print(
        f"Extracting H&E polygon vertices from geometry column: "
        f"{geometry_col}"
    )

    vertices = geometry_table_to_vertices(
        geometry_table=he_table,
        cell_id_col=cell_id_col,
        geometry_col=geometry_col,
    )

    return (
        vertices,
        cell_id_col,
        "vertex_x",
        "vertex_y",
    )


def geometry_table_to_vertices(
    geometry_table: pd.DataFrame,
    cell_id_col: str,
    geometry_col: str = "geometry",
) -> pd.DataFrame:
    """
    Convert Polygon or MultiPolygon geometries to a vertex table.

    For MultiPolygon geometries, all component polygons are retained.
    A `polygon_part` column identifies the component.

    Interior rings are ignored because cell segmentations normally use
    only the exterior boundary.
    """
    rows = []

    for row in geometry_table[
        [cell_id_col, geometry_col]
    ].itertuples(index=False, name=None):

        cell_id, raw_geometry = row

        geometry = parse_geometry(raw_geometry)

        if geometry is None or geometry.is_empty:
            continue

        if isinstance(geometry, Polygon):
            polygon_parts = [geometry]

        elif isinstance(geometry, MultiPolygon):
            polygon_parts = list(geometry.geoms)

        else:
            # Some GeoParquet readers may return a GeometryCollection.
            polygon_parts = [
                item
                for item in getattr(geometry, "geoms", [])
                if isinstance(item, Polygon)
            ]

        for part_index, polygon in enumerate(polygon_parts):
            coordinates = np.asarray(
                polygon.exterior.coords,
                dtype=np.float64,
            )

            if coordinates.ndim != 2 or coordinates.shape[1] < 2:
                continue

            for vertex_index, coordinate in enumerate(coordinates):
                rows.append(
                    {
                        cell_id_col: cell_id,
                        "polygon_part": part_index,
                        "vertex_index": vertex_index,
                        "vertex_x": float(coordinate[0]),
                        "vertex_y": float(coordinate[1]),
                    }
                )

    vertices = pd.DataFrame(rows)

    if vertices.empty:
        raise ValueError(
            "No polygon vertices could be extracted from the geometry "
            "column. Inspect the first geometry value and its type."
        )

    vertices["_vertex_order"] = np.arange(
        len(vertices),
        dtype=np.int64,
    )

    return vertices


def parse_geometry(value) -> Optional[BaseGeometry]:
    """
    Parse common geometry representations into a Shapely geometry.
    """
    if value is None:
        return None

    if isinstance(value, BaseGeometry):
        return value

    # WKB stored as Python bytes, bytearray or memoryview.
    if isinstance(value, (bytes, bytearray, memoryview)):
        return wkb.loads(bytes(value))

    if isinstance(value, str):
        stripped = value.strip()

        if not stripped:
            return None

        # Try WKT first.
        if stripped.upper().startswith(
            (
                "POLYGON",
                "MULTIPOLYGON",
                "GEOMETRYCOLLECTION",
            )
        ):
            return wkt.loads(stripped)

        # Otherwise try hexadecimal WKB.
        try:
            return wkb.loads(stripped, hex=True)
        except Exception as exc:
            raise ValueError(
                "Could not parse geometry string as WKT or hexadecimal WKB."
            ) from exc

    # PyArrow scalar-like values.
    if hasattr(value, "as_py"):
        return parse_geometry(value.as_py())

    raise TypeError(
        f"Unsupported geometry representation: {type(value)}"
    )

def greedy_one_to_one_centroid_matching(
    xenium_centroids: pd.DataFrame,
    he_centroids: pd.DataFrame,
    pixel_size_um: Tuple[float, float],
    max_distance_um: float,
) -> pd.DataFrame:
    """
    Perform global greedy one-to-one matching.

    All candidate pairs within max_distance_um are generated. They are
    sorted from shortest to longest distance. A pair is accepted only if
    neither cell has already been assigned.

    This avoids constructing a full dense distance matrix.
    """

    if xenium_centroids.empty or he_centroids.empty:
        return pd.DataFrame(
            columns=[
                "cell_id_xenium",
                "cell_id_he",
                "distance_um",
            ]
        )

    pixel_size_x_um, pixel_size_y_um = pixel_size_um

    xenium_points_um = np.column_stack(
        [
            xenium_centroids["centroid_x_px"].to_numpy(
                dtype=np.float64
            ) * pixel_size_x_um,
            xenium_centroids["centroid_y_px"].to_numpy(
                dtype=np.float64
            ) * pixel_size_y_um,
        ]
    )

    he_points_um = np.column_stack(
        [
            he_centroids["centroid_x_px"].to_numpy(
                dtype=np.float64
            ) * pixel_size_x_um,
            he_centroids["centroid_y_px"].to_numpy(
                dtype=np.float64
            ) * pixel_size_y_um,
        ]
    )

    he_tree = cKDTree(he_points_um)

    nearby_he_indices = he_tree.query_ball_point(
        xenium_points_um,
        r=max_distance_um,
    )

    candidate_xenium_indices = []
    candidate_he_indices = []
    candidate_distances = []

    for xenium_index, he_indices in enumerate(
        nearby_he_indices
    ):
        if not he_indices:
            continue

        candidate_he_points = he_points_um[he_indices]

        differences = (
            candidate_he_points
            - xenium_points_um[xenium_index]
        )

        distances = np.sqrt(
            np.sum(differences * differences, axis=1)
        )

        for he_index, distance in zip(
            he_indices,
            distances,
        ):
            candidate_xenium_indices.append(xenium_index)
            candidate_he_indices.append(he_index)
            candidate_distances.append(float(distance))

    if not candidate_distances:
        return pd.DataFrame(
            columns=[
                "cell_id_xenium",
                "cell_id_he",
                "distance_um",
            ]
        )

    candidate_xenium_indices = np.asarray(
        candidate_xenium_indices,
        dtype=np.int64,
    )
    candidate_he_indices = np.asarray(
        candidate_he_indices,
        dtype=np.int64,
    )
    candidate_distances = np.asarray(
        candidate_distances,
        dtype=np.float64,
    )

    order = np.argsort(
        candidate_distances,
        kind="stable",
    )

    assigned_xenium = set()
    assigned_he = set()
    accepted_rows = []

    xenium_ids = xenium_centroids[
        "cell_id_xenium"
    ].to_numpy()

    he_ids = he_centroids[
        "cell_id_he"
    ].to_numpy()

    for candidate_position in order:
        xenium_index = int(
            candidate_xenium_indices[candidate_position]
        )
        he_index = int(
            candidate_he_indices[candidate_position]
        )

        if xenium_index in assigned_xenium:
            continue

        if he_index in assigned_he:
            continue

        assigned_xenium.add(xenium_index)
        assigned_he.add(he_index)

        accepted_rows.append(
            {
                "cell_id_xenium": xenium_ids[xenium_index],
                "cell_id_he": he_ids[he_index],
                "distance_um": candidate_distances[
                    candidate_position
                ],
            }
        )

    matches_df = pd.DataFrame(accepted_rows)

    if not matches_df.empty:
        matches_df = matches_df.sort_values(
            "distance_um"
        ).reset_index(drop=True)

    return matches_df


def calculate_polygon_centroids(
    polygons: pd.DataFrame,
    id_col: str,
    x_col: str,
    y_col: str,
    output_id_col: str,
) -> pd.DataFrame:
    """
    Calculate one centroid per cell.

    For ordinary polygons, an area-weighted polygon centroid is used.
    For cells with multiple polygon parts, component centroids are weighted
    by component area.
    """
    rows = []

    for cell_id, cell_df in polygons.groupby(
        id_col,
        sort=False,
    ):
        component_centroids = []
        component_areas = []

        if "polygon_part" in cell_df.columns:
            component_iterator = cell_df.groupby(
                "polygon_part",
                sort=False,
            )
        else:
            component_iterator = [(0, cell_df)]

        for _, polygon_df in component_iterator:
            xy = polygon_df[
                [x_col, y_col]
            ].to_numpy(dtype=np.float64)

            xy = xy[
                np.isfinite(xy[:, 0])
                & np.isfinite(xy[:, 1])
            ]

            if len(xy) == 0:
                continue

            if len(xy) < 3:
                centroid_x = float(np.mean(xy[:, 0]))
                centroid_y = float(np.mean(xy[:, 1]))
                area = 0.0
            else:
                centroid_x, centroid_y = polygon_centroid(xy)
                area = polygon_area(xy)

            component_centroids.append(
                (centroid_x, centroid_y)
            )
            component_areas.append(area)

        if not component_centroids:
            continue

        component_centroids = np.asarray(
            component_centroids,
            dtype=np.float64,
        )

        component_areas = np.asarray(
            component_areas,
            dtype=np.float64,
        )

        if np.sum(component_areas) > 0:
            centroid_x = float(
                np.average(
                    component_centroids[:, 0],
                    weights=component_areas,
                )
            )

            centroid_y = float(
                np.average(
                    component_centroids[:, 1],
                    weights=component_areas,
                )
            )
        else:
            centroid_x = float(
                np.mean(component_centroids[:, 0])
            )

            centroid_y = float(
                np.mean(component_centroids[:, 1])
            )

        rows.append(
            {
                output_id_col: cell_id,
                "centroid_x_px": centroid_x,
                "centroid_y_px": centroid_y,
            }
        )

    return pd.DataFrame(rows)


def polygon_area(
    xy: np.ndarray,
) -> float:
    """
    Calculate absolute polygon area using the shoelace formula.
    """
    xy = np.asarray(xy, dtype=np.float64)

    if len(xy) < 3:
        return 0.0

    if not np.allclose(
        xy[0],
        xy[-1],
        rtol=0,
        atol=1e-10,
    ):
        xy = np.vstack([xy, xy[0]])

    x0 = xy[:-1, 0]
    y0 = xy[:-1, 1]
    x1 = xy[1:, 0]
    y1 = xy[1:, 1]

    return float(
        0.5 * abs(
            np.sum(x0 * y1 - x1 * y0)
        )
    )


def polygon_centroid(
    xy: np.ndarray,
) -> Tuple[float, float]:
    """
    Calculate an area-weighted polygon centroid.
    """

    if not np.allclose(
        xy[0],
        xy[-1],
        rtol=0,
        atol=1e-10,
    ):
        xy = np.vstack([xy, xy[0]])

    x0 = xy[:-1, 0]
    y0 = xy[:-1, 1]
    x1 = xy[1:, 0]
    y1 = xy[1:, 1]

    cross = x0 * y1 - x1 * y0
    area_twice = np.sum(cross)

    if np.isclose(area_twice, 0):
        return (
            float(np.mean(xy[:-1, 0])),
            float(np.mean(xy[:-1, 1])),
        )

    centroid_x = np.sum(
        (x0 + x1) * cross
    ) / (3.0 * area_twice)

    centroid_y = np.sum(
        (y0 + y1) * cross
    ) / (3.0 * area_twice)

    return float(centroid_x), float(centroid_y)


# =====================================================================
# Transformation helpers
# =====================================================================

def correct_matrix_resolution(
    transformed_xy: np.ndarray,
    matrix_used: np.ndarray,
    matrix_image_pixel_size_um: PixelSize,
    display_image_pixel_size_um: PixelSize,
    coordinate_convention: str = "column",
    anchor: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """
    Correct coordinates from the matrix-reference image resolution to
    the displayed full-resolution H&E image.

    This reproduces the transformation behaviour established for the
    current dataset: scaling around the affine translation anchor.
    """

    transformed_xy = np.asarray(
        transformed_xy,
        dtype=np.float64,
    ).copy()

    matrix_mpp_x, matrix_mpp_y = parse_pixel_size(
        matrix_image_pixel_size_um,
        "matrix_image_pixel_size_um",
    )

    display_mpp_x, display_mpp_y = parse_pixel_size(
        display_image_pixel_size_um,
        "display_image_pixel_size_um",
    )

    scale_x = matrix_mpp_x / display_mpp_x
    scale_y = matrix_mpp_y / display_mpp_y

    if anchor is None:
        if coordinate_convention == "column":
            anchor_x = float(matrix_used[0, 2])
            anchor_y = float(matrix_used[1, 2])
        else:
            anchor_x = float(matrix_used[2, 0])
            anchor_y = float(matrix_used[2, 1])
    else:
        anchor_x, anchor_y = map(float, anchor)

    transformed_xy[:, 0] = (
        anchor_x
        + (
            transformed_xy[:, 0]
            - anchor_x
        ) * scale_x
    )

    transformed_xy[:, 1] = (
        anchor_y
        + (
            transformed_xy[:, 1]
            - anchor_y
        ) * scale_y
    )

    return transformed_xy


def apply_affine_transform(
    xy: np.ndarray,
    matrix: np.ndarray,
    convention: str = "column",
) -> np.ndarray:
    """
    Apply a homogeneous 3x3 transformation.
    """

    xy = np.asarray(xy, dtype=np.float64)

    homogeneous_xy = np.column_stack(
        [
            xy,
            np.ones(len(xy), dtype=np.float64),
        ]
    )

    if convention == "column":
        transformed = (
            matrix @ homogeneous_xy.T
        ).T
    elif convention == "row":
        transformed = homogeneous_xy @ matrix
    else:
        raise ValueError(
            "convention must be 'column' or 'row'."
        )

    denominator = transformed[:, 2]

    if np.any(np.isclose(denominator, 0)):
        raise ValueError(
            "The transformation produced homogeneous w=0."
        )

    return (
        transformed[:, :2]
        / denominator[:, None]
    )


def estimate_matrix_pixel_size_um(
    original_matrix: np.ndarray,
    coordinate_convention: str = "column",
) -> Tuple[float, float]:
    """
    Estimate the image pixel size encoded in the original matrix.

    The original matrix is expected to map image pixels to physical
    coordinates in micrometres.
    """

    matrix = np.asarray(
        original_matrix,
        dtype=np.float64,
    )

    if matrix.shape != (3, 3):
        raise ValueError(
            "original_matrix must have shape (3, 3)."
        )

    if coordinate_convention == "column":
        pixel_size_x = np.linalg.norm(matrix[:2, 0])
        pixel_size_y = np.linalg.norm(matrix[:2, 1])
    elif coordinate_convention == "row":
        pixel_size_x = np.linalg.norm(matrix[0, :2])
        pixel_size_y = np.linalg.norm(matrix[1, :2])
    else:
        raise ValueError(
            "coordinate_convention must be 'column' or 'row'."
        )

    return float(pixel_size_x), float(pixel_size_y)


# =====================================================================
# Polygon table helpers
# =====================================================================

def find_xenium_boundary_file(
    xenium_bundle_path: PathLike,
    boundary_type: str = "cell",
) -> Path:
    """
    Find Xenium cell or nucleus boundary polygons.
    """

    xenium_bundle_path = Path(xenium_bundle_path)

    boundary_type = boundary_type.lower()

    if boundary_type not in {"cell", "nucleus"}:
        raise ValueError(
            "boundary_type must be 'cell' or 'nucleus'."
        )

    filenames = [
        f"{boundary_type}_boundaries.parquet",
        f"{boundary_type}_boundaries.csv.gz",
        f"{boundary_type}_boundaries.csv",
    ]

    for filename in filenames:
        direct_path = xenium_bundle_path / filename

        if direct_path.exists():
            return direct_path

    for filename in filenames:
        matches = list(
            xenium_bundle_path.rglob(filename)
        )

        if matches:
            return matches[0]

    raise FileNotFoundError(
        f"Could not find {boundary_type} boundaries in "
        f"{xenium_bundle_path}."
    )


def read_polygon_table(
    polygon_path: PathLike,
) -> pd.DataFrame:
    """
    Read polygon vertices from Parquet or CSV.
    """

    polygon_path = Path(polygon_path)
    filename = polygon_path.name.lower()

    if filename.endswith(".parquet"):
        return pd.read_parquet(polygon_path)

    if filename.endswith(".csv.gz"):
        return pd.read_csv(polygon_path)

    if filename.endswith(".csv"):
        return pd.read_csv(polygon_path)

    raise ValueError(
        f"Unsupported polygon file type: {polygon_path}"
    )


def detect_polygon_columns(
    polygons: pd.DataFrame,
    id_candidates: Sequence[str],
    x_candidates: Sequence[str],
    y_candidates: Sequence[str],
) -> Tuple[str, str, str]:
    """
    Detect ID, x and y columns case-insensitively.
    """

    lower_to_original = {
        str(column).lower(): column
        for column in polygons.columns
    }

    id_col = next(
        (
            lower_to_original[name.lower()]
            for name in id_candidates
            if name.lower() in lower_to_original
        ),
        None,
    )

    x_col = next(
        (
            lower_to_original[name.lower()]
            for name in x_candidates
            if name.lower() in lower_to_original
        ),
        None,
    )

    y_col = next(
        (
            lower_to_original[name.lower()]
            for name in y_candidates
            if name.lower() in lower_to_original
        ),
        None,
    )

    if id_col is None or x_col is None or y_col is None:
        raise ValueError(
            "Could not detect polygon columns.\n"
            f"Columns found: {list(polygons.columns)}\n"
            f"Detected ID: {id_col}\n"
            f"Detected x: {x_col}\n"
            f"Detected y: {y_col}"
        )

    return id_col, x_col, y_col


def clean_polygon_table(
    polygons: pd.DataFrame,
    id_col: str,
    x_col: str,
    y_col: str,
) -> pd.DataFrame:
    """
    Convert coordinates to numeric and remove invalid rows.
    """

    polygons = polygons.copy()

    polygons[x_col] = pd.to_numeric(
        polygons[x_col],
        errors="coerce",
    )

    polygons[y_col] = pd.to_numeric(
        polygons[y_col],
        errors="coerce",
    )

    polygons = polygons.dropna(
        subset=[id_col, x_col, y_col]
    ).copy()

    polygons["_vertex_order"] = np.arange(
        len(polygons),
        dtype=np.int64,
    )

    return polygons


def infer_he_coordinate_units(
    he_polygons: pd.DataFrame,
    x_col: str,
    y_col: str,
    image_width_px: int,
    image_height_px: int,
    pixel_size_x_um: float,
    pixel_size_y_um: float,
    requested_units: str = "auto",
) -> str:
    """
    Infer whether H&E polygon coordinates are pixels or micrometres.
    """

    requested_units = requested_units.lower()

    if requested_units not in {
        "auto",
        "pixels",
        "microns",
    }:
        raise ValueError(
            "he_coordinate_units must be 'auto', 'pixels', "
            "or 'microns'."
        )

    if requested_units != "auto":
        return requested_units

    x_max = float(he_polygons[x_col].max())
    y_max = float(he_polygons[y_col].max())

    image_width_um = image_width_px * pixel_size_x_um
    image_height_um = image_height_px * pixel_size_y_um

    fits_pixels = (
        x_max <= image_width_px * 1.1
        and y_max <= image_height_px * 1.1
    )

    fits_microns = (
        x_max <= image_width_um * 1.1
        and y_max <= image_height_um * 1.1
    )

    # Full-resolution pixel coordinates generally have much larger ranges
    # than physical micrometre coordinates.
    if fits_pixels and not fits_microns:
        return "pixels"

    if fits_microns and not fits_pixels:
        return "microns"

    if fits_pixels and fits_microns:
        # Prefer pixels if the coordinate range occupies a substantial
        # fraction of the image pixel dimensions.
        pixel_fraction = max(
            x_max / max(image_width_px, 1),
            y_max / max(image_height_px, 1),
        )

        micron_fraction = max(
            x_max / max(image_width_um, 1),
            y_max / max(image_height_um, 1),
        )

        if pixel_fraction >= micron_fraction:
            return "pixels"

        return "microns"

    raise ValueError(
        "Could not infer whether the H&E polygon coordinates are "
        "pixels or micrometres.\n"
        f"Polygon maximum x/y: {x_max:.3f}, {y_max:.3f}\n"
        f"Image pixel dimensions: "
        f"{image_width_px}, {image_height_px}\n"
        f"Image physical dimensions: "
        f"{image_width_um:.3f}, {image_height_um:.3f} µm\n"
        "Set he_coordinate_units explicitly."
    )


# =====================================================================
# Plotting helpers
# =====================================================================

def filter_polygons_intersecting_crop(
    polygons: pd.DataFrame,
    id_col: str,
    x_col: str,
    y_col: str,
    x_bounds: Tuple[float, float],
    y_bounds: Tuple[float, float],
) -> pd.DataFrame:
    """
    Keep complete polygons whose bounding boxes intersect the crop.
    """

    x_min, x_max = x_bounds
    y_min, y_max = y_bounds

    polygon_bounds = (
        polygons.groupby(id_col, sort=False)
        .agg(
            x_min=(x_col, "min"),
            x_max=(x_col, "max"),
            y_min=(y_col, "min"),
            y_max=(y_col, "max"),
        )
    )

    intersects = (
        (polygon_bounds["x_max"] >= x_min)
        & (polygon_bounds["x_min"] <= x_max)
        & (polygon_bounds["y_max"] >= y_min)
        & (polygon_bounds["y_min"] <= y_max)
    )

    selected_ids = polygon_bounds.index[intersects]

    result = polygons.loc[
        polygons[id_col].isin(selected_ids)
    ].copy()

    if "_vertex_order" in result.columns:
        result = result.sort_values(
            [id_col, "_vertex_order"]
        )

    return result


def build_colored_polygon_segments(
    polygons: pd.DataFrame,
    id_col: str,
    x_col: str,
    y_col: str,
    color_map: dict,
):
    """
    Build matched coloured segments and unmatched segments.
    """

    matched_segments = []
    matched_colors = []
    unmatched_segments = []

    for cell_id, polygon_df in polygons.groupby(
        id_col,
        sort=False,
    ):
        xy = polygon_df[
            [x_col, y_col]
        ].to_numpy(dtype=np.float64)

        xy = xy[
            np.isfinite(xy[:, 0])
            & np.isfinite(xy[:, 1])
        ]

        if len(xy) < 2:
            continue

        if not np.allclose(
            xy[0],
            xy[-1],
            rtol=0,
            atol=1e-8,
        ):
            xy = np.vstack([xy, xy[0]])

        if cell_id in color_map:
            matched_segments.append(xy)
            matched_colors.append(color_map[cell_id])
        else:
            unmatched_segments.append(xy)

    return (
        matched_segments,
        matched_colors,
        unmatched_segments,
    )


def create_distinct_random_colors(
    n_colors: int,
    random_seed: int = 0,
):
    """
    Create reproducible bright colours.

    Hue order is randomised, while saturation and value remain high enough
    to remain visible over H&E.
    """

    if n_colors == 0:
        return []

    rng = np.random.default_rng(random_seed)

    hues = (
        np.arange(n_colors, dtype=float)
        / max(n_colors, 1)
    )

    rng.shuffle(hues)

    colors = []

    for hue in hues:
        saturation = rng.uniform(0.65, 0.95)
        value = rng.uniform(0.75, 1.0)

        colors.append(
            colorsys.hsv_to_rgb(
                float(hue),
                float(saturation),
                float(value),
            )
        )

    return colors


# =====================================================================
# Image and OME metadata helpers
# =====================================================================

def read_ome_image_metadata(
    image_path: PathLike,
) -> dict:
    """
    Read full-resolution dimensions and physical pixel size from OME-TIFF.
    """

    image_path = Path(image_path)

    with tifffile.TiffFile(image_path) as tif:
        series = tif.series[0]

        level_zero = series.levels[0]

        axes = level_zero.axes
        shape = level_zero.shape

        ome_xml = tif.ome_metadata

    if "X" not in axes or "Y" not in axes:
        raise ValueError(
            f"Could not identify X and Y axes from TIFF axes: {axes}"
        )

    width_px = int(shape[axes.index("X")])
    height_px = int(shape[axes.index("Y")])

    if ome_xml is None:
        raise ValueError(
            "The H&E image does not contain OME metadata."
        )

    root = ET.fromstring(ome_xml)

    namespace_uri = root.tag.split("}")[0].strip("{")

    namespace = {"ome": namespace_uri}

    pixels_element = root.find(
        ".//ome:Pixels",
        namespace,
    )

    if pixels_element is None:
        raise ValueError(
            "Could not find the OME Pixels element."
        )

    physical_size_x = pixels_element.attrib.get(
        "PhysicalSizeX"
    )
    physical_size_y = pixels_element.attrib.get(
        "PhysicalSizeY"
    )

    if physical_size_x is None or physical_size_y is None:
        raise ValueError(
            "The OME metadata does not contain PhysicalSizeX and "
            "PhysicalSizeY."
        )

    unit_x = pixels_element.attrib.get(
        "PhysicalSizeXUnit",
        "µm",
    )
    unit_y = pixels_element.attrib.get(
        "PhysicalSizeYUnit",
        "µm",
    )

    pixel_size_x_um = convert_length_to_microns(
        float(physical_size_x),
        unit_x,
    )

    pixel_size_y_um = convert_length_to_microns(
        float(physical_size_y),
        unit_y,
    )

    return {
        "width_px": width_px,
        "height_px": height_px,
        "pixel_size_x_um": pixel_size_x_um,
        "pixel_size_y_um": pixel_size_y_um,
        "axes": axes,
        "shape": shape,
    }


def convert_length_to_microns(
    value: float,
    unit: str,
) -> float:
    """
    Convert common OME length units to micrometres.
    """

    normalized_unit = (
        str(unit)
        .strip()
        .lower()
        .replace("μ", "µ")
    )

    if normalized_unit in {
        "µm",
        "um",
        "micrometer",
        "micrometre",
        "micrometers",
        "micrometres",
    }:
        return float(value)

    if normalized_unit in {
        "nm",
        "nanometer",
        "nanometre",
        "nanometers",
        "nanometres",
    }:
        return float(value) / 1000.0

    if normalized_unit in {
        "mm",
        "millimeter",
        "millimetre",
        "millimeters",
        "millimetres",
    }:
        return float(value) * 1000.0

    raise ValueError(
        f"Unsupported OME length unit: {unit}"
    )


def get_image_spatial_shape(
    image: np.ndarray,
    channel_axis: Optional[int],
) -> Tuple[int, int, Optional[int]]:
    """
    Return height, width and resolved channel axis.
    """

    image = np.asarray(image)

    if image.ndim == 2:
        height, width = image.shape
        return height, width, None

    if image.ndim != 3:
        raise ValueError(
            "image must have shape (Y, X), (C, Y, X), or "
            f"(Y, X, C). Received {image.shape}."
        )

    if channel_axis is None:
        channel_axis = infer_channel_axis(image.shape)

    channel_axis = channel_axis % image.ndim

    if channel_axis == 0:
        _, height, width = image.shape
    elif channel_axis == 2:
        height, width, _ = image.shape
    else:
        raise ValueError(
            "channel_axis must be 0 or -1/2."
        )

    return height, width, channel_axis


def crop_image(
    image: np.ndarray,
    crop_x0: int,
    crop_x1: int,
    crop_y0: int,
    crop_y1: int,
    channel_axis: Optional[int],
) -> np.ndarray:
    """
    Crop before conversion to float.
    """

    if image.ndim == 2:
        return image[
            crop_y0:crop_y1,
            crop_x0:crop_x1,
        ]

    if channel_axis == 0:
        return image[
            :,
            crop_y0:crop_y1,
            crop_x0:crop_x1,
        ]

    if channel_axis == 2:
        return image[
            crop_y0:crop_y1,
            crop_x0:crop_x1,
            :,
        ]

    raise ValueError(
        f"Unsupported channel axis: {channel_axis}"
    )


def prepare_cropped_image_for_display(
    image_crop: np.ndarray,
    channels: Optional[Union[int, Sequence[int]]],
    channel_axis: Optional[int],
    percentiles: Tuple[float, float],
    channel_weights: Optional[Sequence[float]],
) -> np.ndarray:
    """
    Normalize only the selected image crop.
    """

    image_crop = np.asarray(image_crop)

    if image_crop.ndim == 2:
        return percentile_normalize(
            image_crop,
            lower=percentiles[0],
            upper=percentiles[1],
        )

    if channel_axis == 0:
        number_of_channels = image_crop.shape[0]
    elif channel_axis == 2:
        number_of_channels = image_crop.shape[2]
    else:
        raise ValueError(
            "channel_axis must be 0 or -1/2."
        )

    if channels is None:
        selected_channels = list(
            range(min(number_of_channels, 3))
        )
    elif isinstance(channels, (int, np.integer)):
        selected_channels = [int(channels)]
    else:
        selected_channels = [
            int(channel)
            for channel in channels
        ]

    if not 1 <= len(selected_channels) <= 3:
        raise ValueError(
            "Select between one and three channels."
        )

    normalized_channels = []

    for channel in selected_channels:
        if not 0 <= channel < number_of_channels:
            raise IndexError(
                f"Channel {channel} is invalid for an image with "
                f"{number_of_channels} channels."
            )

        if channel_axis == 0:
            channel_crop = image_crop[channel]
        else:
            channel_crop = image_crop[..., channel]

        normalized_channels.append(
            percentile_normalize(
                channel_crop,
                lower=percentiles[0],
                upper=percentiles[1],
            )
        )

    if channel_weights is not None:
        weights = np.asarray(
            channel_weights,
            dtype=np.float32,
        )

        if len(weights) != len(normalized_channels):
            raise ValueError(
                "channel_weights must contain one value per "
                "selected channel."
            )

        normalized_channels = [
            np.clip(channel * weight, 0, 1)
            for channel, weight in zip(
                normalized_channels,
                weights,
            )
        ]

    if len(normalized_channels) == 1:
        return normalized_channels[0]

    if len(normalized_channels) == 2:
        empty_channel = np.zeros_like(
            normalized_channels[0]
        )

        return np.stack(
            [
                normalized_channels[0],
                normalized_channels[1],
                empty_channel,
            ],
            axis=-1,
        )

    return np.stack(
        normalized_channels,
        axis=-1,
    )


def percentile_normalize(
    image: np.ndarray,
    lower: float,
    upper: float,
) -> np.ndarray:
    """
    Normalize a cropped image channel to 0-1 using float32.
    """

    if not 0 <= lower < upper <= 100:
        raise ValueError(
            "Percentiles must satisfy "
            "0 <= lower < upper <= 100."
        )

    result = np.asarray(
        image,
        dtype=np.float32,
    ).copy()

    finite_mask = np.isfinite(result)

    if not finite_mask.any():
        return np.zeros_like(
            result,
            dtype=np.float32,
        )

    vmin, vmax = np.percentile(
        result[finite_mask],
        [lower, upper],
    )

    if np.isclose(vmin, vmax):
        return np.zeros_like(
            result,
            dtype=np.float32,
        )

    result -= np.float32(vmin)
    result /= np.float32(vmax - vmin)

    np.clip(
        result,
        0,
        1,
        out=result,
    )

    result[~finite_mask] = 0

    return result


def infer_channel_axis(
    shape: Tuple[int, int, int],
) -> int:
    """
    Infer channel-first or channel-last organisation.
    """

    first_is_small = shape[0] <= 10
    last_is_small = shape[-1] <= 10

    if first_is_small and not last_is_small:
        return 0

    if last_is_small and not first_is_small:
        return 2

    if first_is_small and last_is_small:
        return 0 if shape[0] <= shape[-1] else 2

    raise ValueError(
        f"Could not infer channel axis from shape {shape}. "
        "Specify channel_axis explicitly."
    )


# =====================================================================
# General helpers
# =====================================================================

def read_3x3_matrix(
    csv_path: PathLike,
) -> np.ndarray:
    """
    Read a numeric 3x3 matrix from CSV.
    """

    csv_path = Path(csv_path)

    candidates = []

    no_header = pd.read_csv(
        csv_path,
        header=None,
    )

    candidates.append(
        no_header.apply(
            pd.to_numeric,
            errors="coerce",
        )
        .dropna(axis=0, how="all")
        .dropna(axis=1, how="all")
        .to_numpy()
    )

    with_header = pd.read_csv(csv_path)

    candidates.append(
        with_header.apply(
            pd.to_numeric,
            errors="coerce",
        )
        .dropna(axis=0, how="all")
        .dropna(axis=1, how="all")
        .to_numpy()
    )

    for candidate in candidates:
        if candidate.shape == (3, 3):
            candidate = candidate.astype(
                np.float64
            )

            if np.all(np.isfinite(candidate)):
                return candidate

    raise ValueError(
        f"Could not extract a numeric 3x3 matrix from {csv_path}."
    )


def parse_pixel_size(
    pixel_size: PixelSize,
    argument_name: str,
) -> Tuple[float, float]:
    """
    Convert scalar or x/y tuple to two values.
    """

    if np.isscalar(pixel_size):
        pixel_size_x = float(pixel_size)
        pixel_size_y = float(pixel_size)
    else:
        if len(pixel_size) != 2:
            raise ValueError(
                f"{argument_name} must be a scalar or (x, y)."
            )

        pixel_size_x, pixel_size_y = map(
            float,
            pixel_size,
        )

    if pixel_size_x <= 0 or pixel_size_y <= 0:
        raise ValueError(
            f"{argument_name} must be greater than zero."
        )

    return pixel_size_x, pixel_size_y


def write_id_list(
    ids: Sequence,
    output_path: PathLike,
):
    """
    Write one ID per line.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for cell_id in ids:
            handle.write(f"{cell_id}\n")

    print(f"Saved ID list: {output_path}")


def _require_existing_path(
    path: Path,
    description: str,
):
    if not path.exists():
        raise FileNotFoundError(
            f"{description} was not found: {path}"
        )

if __name__ == "__main__":
    fire.Fire(assign_cells) 
