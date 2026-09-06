# Reproduction notes

The field experiments require authorized access to the project data and the same preprocessing configuration used to derive the manuscript results. This repository intentionally contains no field data and no synthetic replacement data.

## Recommended workflow

1. Create a local Python environment and install `requirements.txt`.
2. Place authorized dynamic and static CSV files outside the repository, or under paths ignored by `.gitignore`.
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
- Report the same test samples and evaluation metrics for every baseline.
