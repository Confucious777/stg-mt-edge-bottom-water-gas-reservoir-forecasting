# Data schema

The public repository does not contain field records or derived labels. The following interface documents the files expected by the loaders when authorized project data are supplied locally.

## Dynamic production table

The dynamic CSV must contain at least these columns:

| Column | Meaning |
| --- | --- |
| `井号` | Well identifier |
| `日期` | Observation date |
| `日产气量` | Daily gas production |
| `日产水量` | Daily water production, when available |
| `V(m3/d)` | Water-invasion-rate label, when the TKG loader is used |

Additional numeric columns are treated as dynamic model features by the production loader. The TKG loader also recognizes measure-type fields and builds one-hot indicators when they are present.

## Static well table

The static table may contain the following well-level fields:

| Column | Meaning |
| --- | --- |
| `井号` | Well identifier |
| `开发层组` | Development layer group |
| `横坐标`, `纵坐标` | Well coordinates |
| `平均射孔深度` | Mean perforation depth |
| `孔隙度` | Porosity |
| `渗透率` | Permeability |
| `含水饱和度` | Water saturation |

The exact column names can be overridden in the training entry points where applicable. Static values are used for graph construction; dynamic production variables are used for sequence windows and edge-weight updates.

## Missing values and exclusions

The loaders coerce numeric fields, fill feature gaps using forward/backward values followed by training-data medians, and maintain target-validity masks. The model code excludes the known temperature fields from dynamic inputs and treats zero porosity/permeability entries as missing static values before graph normalization. Sentinel values such as `-999` must be converted to missing values during project preprocessing and must not be interpreted as physical measurements.

## Privacy

Do not place raw field data, well names that identify an operator's asset, private coordinates, production histories, labels, model checkpoints, or generated experiment files in this repository.
