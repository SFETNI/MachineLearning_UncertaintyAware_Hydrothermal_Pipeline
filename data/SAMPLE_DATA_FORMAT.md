# Sample Data Format

This is a minimal example showing the expected CSV format for the hydrothermal processing dataset (derived dataset)

## Instructions

The dataset should contain:
- Feedstock composition (elemental & biochemical)
- Process conditions (temperature, time, pressure, etc.)
- Product yields and properties

## Minimal Example

```csv
C,H,O,N,S,Ash,Lignin,cellulose,hemicellulose,T,t,pressure_effective_mpa,IC,water_biomass_ratio,catalyst_biomass_ratio,biochar_Y_daf,biooil_Y_daf,water_Y_daf,gas_Y_daf,C_biochar,H_biochar,O_biochar,HHV_biochar,C_biooil,H_biooil,O_biooil,HHV_biooil,DOI,Feedstock,Family,process_subtype
47.2,6.1,45.8,0.5,0.4,2.5,78.3,17.2,28.5,42.1,29.4,300,30,10.0,10.0,10.0,0.0,35.2,48.5,12.3,4.0,68.4,4.2,26.8,26.5,72.1,7.8,19.5,28.3,10.1234/example1,Corn Stover,Agricultural Residues,HTL
51.3,5.9,42.1,0.6,0.1,3.2,76.5,20.3,32.1,38.5,29.4,320,45,12.0,15.0,8.0,0.0,32.8,52.1,10.5,4.6,70.2,4.5,24.8,27.8,73.5,8.1,17.9,29.1,10.1234/example2,Wheat Straw,Agricultural Residues,HTL
```

## Column Descriptions

### Feedstock Composition (wt%, dry basis)
- C, H, O, N, S: Elemental composition
- Ash: Ash content
- Lignin, cellulose, hemicellulose: Biochemical components etc. 

### Process Conditions
- T: Temperature (°C)
- t: Residence time (min)
- pressure_effective_mpa: Effective pressure (MPa)
- IC: Initial concentration (wt%)
- water_biomass_ratio: Water to biomass ratio
- catalyst_biomass_ratio: Catalyst to biomass ratio (0 if none)

### Product Yields (wt%, dry ash-free basis)
- biochar_Y_daf: Biochar yield
- biooil_Y_daf: Bio-oil yield
- water_Y_daf: Aqueous phase yield
- gas_Y_daf: Gas yield

### Product Properties
- C_biochar, H_biochar, O_biochar: Biochar elemental composition (wt%, daf)
- HHV_biochar: Biochar higher heating value (MJ/kg)
- C_biooil, H_biooil, O_biooil: Bio-oil elemental composition (wt%, daf)
- HHV_biooil: Bio-oil higher heating value (MJ/kg)

### Metadata
- DOI: Publication identifier
- Feedstock: Specific feedstock name
- Family: Feedstock category
- process_subtype: Process variant (HTL, HTC, etc.)

## Data Quality

Ensure your data:
- Has no duplicate rows (unless intentional replicates)
- Uses consistent units
- Handles missing values appropriately
- Is normalized/standardized if needed
- Has balanced train/test representation

