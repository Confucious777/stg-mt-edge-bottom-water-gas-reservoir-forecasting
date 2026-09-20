# Data schema

The public repository does not contain field records or derived labels. The following interface documents the files expected by the loaders when authorized project data are supplied locally.

## Dynamic production table

The release file is `data/public/processed_en/production_dynamic.csv.gz`; the
loader reads it directly as a gzip-compressed CSV.

The dynamic CSV must contain at least these columns:

| Column | Meaning |
| --- | --- |
| `well_id` | Well identifier |
| `date` | Observation date |
| `gas_production` | Daily gas production |
| `water_production` | Daily water production, when available |
| `water_invasion_rate` | Water-invasion-rate label, when the TKG loader is used |

Additional numeric columns are treated as dynamic model features by the production loader. The TKG loader also recognizes measure-type fields and builds one-hot indicators when they are present.

## Static well table

The static table may contain the following well-level fields:

| Column | Meaning |
| --- | --- |
| `well_id` | Well identifier |
| `layer_group` | Development layer group |
| `x_coordinate`, `y_coordinate` | Well coordinates |
| `mean_perforation_depth` | Mean perforation depth |
| `porosity` | Porosity |
| `permeability` | Permeability |
| `water_saturation` | Water saturation |

The exact column names can be overridden in the training entry points where applicable. Static values are used for graph construction; dynamic production variables are used for sequence windows and edge-weight updates.

## Missing values and exclusions

The loaders coerce numeric fields, fill feature gaps using forward/backward values followed by training-data medians, and maintain target-validity masks. The model code excludes the known temperature fields from dynamic inputs and treats zero porosity/permeability entries as missing static values before graph normalization. Sentinel values such as `-999` must be converted to missing values during project preprocessing and must not be interpreted as physical measurements.

## Privacy

The repository dataset is a relationship-preserving anonymized release. Well
identifiers, dates, coordinates, layer labels, measure labels, and static
non-target attributes are transformed before publication. Raw field data,
operator-identifying well names, source coordinates, checkpoints, and
generated experiment files must not be added to the repository.
