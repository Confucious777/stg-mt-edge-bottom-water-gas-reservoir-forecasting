# STG-MT: Edge-Bottom Water Gas Reservoir Forecasting

This repository contains the core PyTorch implementation of the STG-MT spatiotemporal forecasting framework for joint gas-production and water-invasion-rate prediction in edge-bottom water gas reservoirs.

## Scope

The release contains the STG-MT model definition, a relationship-preserving anonymized dataset, data-loader interfaces, training entry points, metrics, and a small unit-test suite. Baseline model implementations and benchmark result artifacts are not included in this public release.

The public dataset anonymizes well identifiers, dates, coordinates, layer labels, measure labels, and static non-target attributes. Gas-production and water-invasion target values are retained to support comparison with the manuscript metrics. The release dataset is not an operator-identifiable copy of the source files.

## Repository layout

```text
dataloader/   CSV readers and sequence-window construction
models/       STG-MT temporal, spatial, and regression modules
pygcn/        graph-neural-network components
scripts/      training entry points
utils/        I/O, metrics, and random-seed helpers
tests/        lightweight unit tests
docs/         data interface and reproduction notes
data/public/  anonymized release dataset (dynamic production is gzip-compressed)
```

## Environment

Python 3.10 or newer is recommended. Install the dependencies with:

```bash
python -m pip install -r requirements.txt
```

Run the available tests with:

```bash
python -m pytest -q
```

The current test suite exercises time-modeling utilities and does not require field data. Training requires PyTorch and authorized input files.

## Training

The default entry point is:

```bash
python main.py
```

The training scripts default to the anonymized release dataset under `data/public/processed_en/`. Run the STG-MT training entry point with:

```bash
python main.py
```

Do not commit raw source data, checkpoints, or experiment outputs.

## Data and reproducibility

See [docs/data_schema.md](docs/data_schema.md) for the required file interface and [docs/reproduction.md](docs/reproduction.md) for the recommended private-data workflow. The implementation uses chronological data handling and computes normalization statistics from the training portion.

### Manuscript experiment defaults

The defaults exposed by the main spatiotemporal training entry point use a chronological 60%/20%/20% train/validation/test split, a 1-step forecast horizon, batch size 16, 8 graph neighbors per node, static graph weights $(\alpha_d,\alpha_p,\alpha_l)=(0.5,0.4,0.1)$, dynamic edge coefficients $(\beta_L,\beta_D)=(0.4,0.4)$, and an influx-loss weight of 0.05. The short-term and long-term temporal ranges remain configurable. The default run does not enable held-out-well masking; use `--holdout_well` explicitly for a separate held-out-well experiment.

## Citation

If you use this code, cite the associated manuscript. Machine-readable metadata are provided in [CITATION.cff](CITATION.cff).

## License

This project is released under the [MIT License](LICENSE).
