# Cell Assignment

Assign cells from Xenium bundle to cells segmented from any other image segmentation pipeline using polygon parquet file. It can be used for datasets which are aligned with global affine transformation if transformation matrix is provided.

The repository performs one-to-one matching between Xenium cells and external segmentation results based on transformed polygon centroids, simply by finding the closest cells and assigning them with upper limit that can be set.


---

## Installation

Clone the repository

```bash
git clone https://github.com/<username>/cell-assignment.git
cd cell-assignment
```

Create a Conda environment

```bash
conda env create -f environment.yml
conda activate cell_assignment
```

---

## Input

The program requires

- Xenium output bundle
- H&E segmentation polygons (Parquet)
- H&E OME-TIFF image
- Affine transformation matrix (optional). Has to be simply csv with matrix 3x3
- 
- 
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

### Xenium to registered segmentation

Demonstrates matching Xenium cells to StarDist segmentation after affine registration. Please nto that depending on direction of reigstration you may use True or False values for `invert_matrix`

```
examples/test1/xenium_to_reg_segm.sh
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
7. Perform greedy one-to-one centroid matching within a defined maximum distance.

---

## License

This project is distributed under the MIT License. See the `LICENSE` file for details.
