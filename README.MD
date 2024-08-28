# Polymer Solar Cells Machine learning

This repo contains code for the paper 'Accelerating materials discovery for polymer solar cells: Data-driven insights enabled by natural language processing' [[1]](https://pubs.acs.org/doi/10.1021/acs.chemmater.4c00709). This paper releases a data set of polymer solar cell (PSC) device characteristics that are extracted using the NLP pipeline described in Ref. [[2]](https://www.nature.com/articles/s41524-023-01003-w) and curated manually. In the paper, we trained machine learning models to predict power conversion efficiency (PCE) of PSCs. We also simulated several active learning strategies to pick out optimal donor/acceptor pairs with high PCE. We compared that against against how the field actually evolved. This exercise gave an estimate of how much faster the field could have evolved and generated several insights on the advantages of different active learning strategies. The code and data to reproduce our experiments is provided in this repo.

## Requirements and Setup

- Python 3.10
- RDKit (version 2023.9.4)
- Scikit-learn (version 1.3.2)

You can install all required Python packages using the provided environment.yml file using the below commands:
```bash
conda env create -f environment.yml
pip install -e .
```

## Repository structure
```
.
├── PolymerSolarCellsML                     # All source code for the current project
│ 
├── dataset                                 # Data set files used by the code
│ ├── polymer_solar_cell_curated_data.xlsx  # Contains curated PSC data
│ └── polymer_solar_cell_extracted_data.csv # Contains PSC data extracted by the NLP pipeline described in Ref. [2]
└── metadata                                # Additional files required by the code
  ├── fp_dict.pkl                           # Pickle file of features for each donor and acceptor which is used as input to the ML model
  ├── normalized_polymer_dictionary.json    # Dictionary of polymer names and common variations in their names found in the literature obtained using the pipeline described in Ref. [3]
  └── property_metadata.json                # Dictionary of common variations in name for several material properties and associated metadata like commonly used units.
  
 ```

## Usage

The ML model for training a model for power conversion efficiency for polymer solar cells can be run using the below commands. The output by default is saved in ./output.

```bash
python ./PolymerSolarCellsML/property_prediction/PCE_models.py \
        --use_median
```

The ML model for simulating active learning over polymer solar cell donor/acceptor pairs can be run using the below command. Other possible input configurations are documented in ./sequential_selection/parse_args.py. The flag --run_parallel_paths will run contextual bandits in a separate process from all gaussian process regression based methods. Use if you have a large enough CPU (>=8 cores) else execute without that flag.

```bash
python ./PolymerSolarCellsML/sequential_selection/PSC_iterative_selection.py \
        --run_parallel_paths
```

The NLP extracted data can be compared against the curated data to compute metrics using the below command. The precision, recall and F1 score for 2-tuple, 3-tuple and 4-tuple overlap are printed to the console.

```bash
python ./PolymerSolarCellsML/nlp_eval.py
```

Please cite our paper if you use the code or data in this repo.

```latex
@article{shetty2024accelerating,
author = {Shetty, Pranav and Adeboye, Aishat and Gupta, Sonakshi and Zhang, Chao and Ramprasad, Rampi},
title = {Accelerating Materials Discovery for Polymer Solar Cells: Data-Driven Insights Enabled by Natural Language Processing},
journal = {Chemistry of Materials},
volume = {36},
number = {16},
pages = {7676-7689},
year = {2024},
doi = {10.1021/acs.chemmater.4c00709},
URL = {https://doi.org/10.1021/acs.chemmater.4c00709},
}
```

## References

[1] Shetty, P., Adeboye, A., Gupta, S., Zhang, C., & Ramprasad, R.. Accelerating Materials Discovery for Polymer Solar Cells: Data-Driven Insights Enabled by Natural Language Processing. Chemistry of Materials, 36(16), 7676-7689 (2024)

[2] Shetty, P., Rajan, A., Kuenneth, C., Gupta, S., Panchumarti, L., Holm, L., Zhang, C. & Ramprasad, R.  A general-purpose material property data extraction pipeline from large polymer corpora using natural language processing. npj Computational Materials 9, 52 (2023) 

[3] Shetty, P., and Ramprasad R.. "Machine-Guided Polymer Knowledge Extraction Using Natural Language Processing: The Example of Named Entity Normalization." Journal of Chemical Information and Modeling (2021)
