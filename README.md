# Cell Assignment

Assign cells detected by Xenium to cells segmented from any other image segmentation pipeline using polygon boundaries.

The repository performs one-to-one matching between Xenium cells and external segmentation results based on transformed polygon centroids.

---

## Features

- Supports Xenium cell or nucleus boundaries
- Accepts external segmentation polygons stored as Parquet files
- Automatic detection of polygon column names
- Supports both vertex tables and geometry (Polygon/MultiPolygon) formats
- Automatic inference of H&E coordinate units (pixels or microns)
- Greedy one-to-one centroid matching
- Optional affine coordinate transformation
- Outputs matched cells and unmatched cell lists
- Includes utilities for visualising matched cell boundaries

---

## Repository structure

```
.
├── bin/
│   └── assign_cells.py
│
├── examples/
│   ├── test1/
│   └── test2/
│
└── README.md
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/<username>/cell-assignment.git
cd cell-assignment
```

Create a Conda environment

```bash
conda create -n cell_assignment python=3.12
conda activate cell_assignment
```

Install required packages

```bash
conda install -c conda-forge \
    numpy \
    pandas \
    scipy \
    matplotlib \
    tifffile \
    shapely \
    pyarrow \
    fire
```

---

## Input

The program requires

- Xenium output bundle
- H&E segmentation polygons (Parquet)
- H&E OME-TIFF image
- Optional affine transformation matrix

External segmentation polygons may originate from any segmentation software, provided that polygon boundaries are available as a Parquet table.

Supported polygon formats include

- explicit vertex tables
- Shapely Polygon
- Shapely MultiPolygon
- WKB
- WKT

---

## Usage

Basic usage

```bash
python bin/assign_cells.py \
    --xenium_bundle_path <xenium_bundle> \
    --image_path <he_image.ome.tif> \
    --polygons_parquet_path <he_polygons.parquet> \
    --output_matches_csv matches.csv
```

---

## Examples

Two example workflows are included.

### Test 1

Demonstrates matching Xenium cells to StarDist segmentation after affine registration.

```
examples/test1/run_test1.sh
```

### Test 2

Demonstrates matching Xenium cells to an alternative segmentation generated directly in Xenium space.

```
examples/test2/run_test2.sh
```

---

## Outputs

The program generates

- matched cell table (.csv)
- unmatched Xenium cell IDs (.txt, optional)
- unmatched H&E cell IDs (.txt, optional)

The matching table contains

| Column | Description |
|---------|-------------|
| cell_id_xenium | Xenium cell identifier |
| cell_id_he | External segmentation cell identifier |
| distance_um | Centroid distance after matching |

---

## Matching algorithm

1. Read Xenium polygons.
2. Read external segmentation polygons.
3. Compute polygon centroids.
4. Apply the affine transformation (if supplied).
5. Convert coordinates into H&E full-resolution pixels.
6. Build a KD-tree for efficient neighbour search.
7. Perform greedy one-to-one centroid matching within a user-defined maximum distance.

---

## Requirements

Python ≥3.10

Main dependencies

- numpy
- pandas
- scipy
- shapely
- tifffile
- matplotlib
- pyarrow
- fire

---

## License

Specify your preferred license here.
