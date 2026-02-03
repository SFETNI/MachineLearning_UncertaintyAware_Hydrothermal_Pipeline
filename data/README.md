# Data Directory

## Overview

Place the hydrothermal processing dataset CSV file (derived ML ready dataset) in this directory.

## Required Data Format

Your dataset should be a CSV file with the following structure:

### Input Features (Feedstock & Process Conditions)

**Feedstock Composition:**
- `C`, `H`, `O`, `N`, `S` - Elemental composition (wt%, dry basis)
- `Ash` - Ash content (wt%, dry basis)
- `Lignin`, `cellulose`, `hemicellulose` - Biochemical composition (wt%, dry basis)
- `H/C`, `O/C` - Elemental ratios
- `LRI` - Lignin Richness Index

**Process Conditions:**
- `T` - Temperature (°C)
- `t` - Residence time (min)
- `pressure_effective_mpa` - Effective pressure (MPa)
- `IC` - Initial concentration (wt%)
- `water_biomass_ratio` - Water to biomass ratio
- `catalyst_biomass_ratio` - Catalyst to biomass ratio (if applicable)

**Optional Categorical Features:**
- `catalyst_description` - Catalyst type
- `solvent_name` - Solvent used
- `Family` - Feedstock family/category
- `process_subtype` - Process variant (HTL, HTC, etc.)

### Output Targets (Product Yields & Properties)

**Yields (wt%, dry ash-free basis):**
- `biochar_Y_daf` - Biochar yield
- `biooil_Y_daf` - Bio-oil yield  
- `water_Y_daf` - Aqueous phase yield
- `gas_Y_daf` - Gas yield

**Biochar Properties:**
- `C_biochar`, `H_biochar`, `O_biochar` - Elemental composition
- `HHV_biochar` - Higher heating value (MJ/kg)

**Bio-oil Properties:**
- `C_biooil`, `H_biooil`, `O_biooil` - Elemental composition
- `HHV_biooil` - Higher heating value (MJ/kg)

### Metadata (Optional but Recommended)

- `DOI` - Digital Object Identifier of source paper
- `Paper_Title` - Title of source publication
- `Feedstock` - Specific feedstock name
- `Ref` - Reference identifier

## Example Data Structure

```csv
C,H,O,N,Ash,T,t,IC,biochar_Y_daf,biooil_Y_daf,C_biochar,HHV_biooil,DOI
47.2,6.1,45.8,0.5,0.4,300,30,10,35.2,48.5,68.4,28.5,10.1234/example
...
```

## Data Placement

1. **Raw Data**: Place your original CSV file here as `HTT_normalized_data.csv` or similar
2. **Processed Data**: The notebook will generate intermediate files in `outputs/tables/`


## Sample Dataset

A minimal sample dataset is provided for testing purposes. Replace it with your actual data before running the pipeline.

To use your own dataset, update the path in the notebook:

```python
csv_path = Path("../data/your_dataset.csv")
htt_data = pd.read_csv(csv_path)
```

## Data Quality Checks

The pipeline performs automatic checks for:
- Missing values
- Data types
- Value ranges
- Outliers

Review the data exploration cells in the notebook for detailed diagnostics.
