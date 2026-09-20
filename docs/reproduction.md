# Reproduction notes

The repository includes a relationship-preserving anonymized release dataset under `data/public/processed_en/`. The source operator data remain private and are not included. The release copy preserves the row-level temporal and inter-well structure; target values are retained to make the published benchmark comparable.

## Recommended workflow

1. Create a local Python environment and install `requirements.txt`.
2. The default command uses the included anonymized dynamic, static, and water-invasion CSV files. Authorized source data can be supplied with the corresponding path arguments if needed.
3. Confirm that dates are parsed as daily observations and that the well identifier is consistent across dynamic and static tables.
4. Convert sentinel values such as `-999` to missing values before feature normalization. Preserve validity masks for targets.
5. Run the appropriate script in `scripts/` after checking its path and hyperparameter arguments.
6. Store checkpoints and metrics outside the repository or in ignored output directories.

## Leakage control

Normalization statistics are fitted from the training portion in the loaders. Sequence windows are built with an explicit forecast horizon, and target values are kept separate from the input features. When reproducing a chronological experiment, keep the split rule and window-boundary policy fixed across all models.

## Reproducibility checklist

- Record Python, PyTorch, CUDA, and GPU versions.
- Record the random seed and the exact command line.
- Record the data snapshot, preprocessing version, and column mapping.
- Record sequence length, forecast horizon, batch size, graph-neighbor setting, and optimization settings.
- Report the test samples and evaluation metrics for the STG-MT release. Comparative baseline implementations and benchmark outputs are maintained outside this public code-only release.
