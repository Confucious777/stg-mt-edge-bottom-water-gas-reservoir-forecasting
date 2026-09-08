# STG-MT: Edge-Bottom Water Gas Reservoir Forecasting

This repository contains the core PyTorch implementation of the STG-MT spatiotemporal forecasting framework for joint gas-production and water-invasion-rate prediction in edge-bottom water gas reservoirs.

## Scope

The release contains model definitions, data-loader interfaces, training entry points, metrics, and a small unit-test suite. Field production records, well coordinates, reservoir properties, derived water-invasion labels, trained checkpoints, and experiment outputs are not included because they are project-specific and may be proprietary. No synthetic-data generator or synthetic dataset is included.

The public code is therefore an implementation release rather than a ready-to-run copy of the field experiment. Reproduction requires authorized access to the data files described in [docs/data_schema.md](docs/data_schema.md).

## Repository layout

```text
dataloader/   CSV readers and sequence-window construction
models/       temporal, spatial, benchmark, and regression models
pygcn/        graph-neural-network components
scripts/      training entry points
utils/        I/O, metrics, and random-seed helpers
tests/        lightweight unit tests
docs/         data interface and reproduction notes
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

The training scripts expose the data paths and model settings used by the implementation. Before training, inspect the argument defaults and provide paths to local authorized data. Do not commit raw data, processed field data, checkpoints, or experiment outputs.

## Data and reproducibility

See [docs/data_schema.md](docs/data_schema.md) for the required file interface and [docs/reproduction.md](docs/reproduction.md) for the recommended private-data workflow. The implementation uses chronological data handling and computes normalization statistics from the training portion.

### Manuscript experiment defaults

The defaults exposed by the main spatiotemporal training entry point correspond to the manuscript experiment: a chronological 60%/20%/20% train/validation/test split, a short-term input length of 5 time steps, a 1-step forecast horizon, batch size 16, 8 graph neighbors per node, a 120-day long-term aggregation scale, and an influx-loss weight of 0.05. The default run does not enable held-out-well masking; use `--holdout_well` explicitly for a separate held-out-well experiment.

## Citation

If you use this code, cite the associated manuscript. Machine-readable metadata are provided in [CITATION.cff](CITATION.cff).

## License

This project is released under the [MIT License](LICENSE).
