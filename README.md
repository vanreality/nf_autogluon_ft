# Autogluon read classifier training pipeline

This Nextflow pipeline automates the process of training machine learning models using `AutoGluon`, which make use of DNA sequences and methylation information for read classification tasks. The pipeline processes multiple datasets in parallel, performs model training, and outputs the results along with log files for each training run.

The workflow is designed to be scalable, reproducible, and efficient, leveraging Nextflow’s ability to handle job scheduling and parallelization.

## Requirements
To use the scripts, ensure the following dependencies are installed:

- [Nextflow](https://www.nextflow.io/docs/latest/index.html)
- [Singularity](https://docs.sylabs.io/guides/4.2/user-guide/) (Optional – for containerized environments)
- [AutoGluon](https://github.com/autogluon/autogluon) (Optional – if not using Singularity)

Install Nextflow
```bash
curl -s https://get.nextflow.io | bash
```

Build AutoGluon Singularity container
```bash
singularity build autogluon.sif autogluon.def
```

Install AutoGluon and dependencies (Optional)
```bash
pip install pandas numpy click scikit-learn autogluon
```

## Pipeline Structure
```
.                            # Project root directory
├── assets                   # Assets files directory
│   └── meta.tsv             # Metadata file containing task parameters (e.g., paths for train/validation/test data)
│
├── data                     # Data directory
│   └── test_data            # Test data directory
│       ├── data_1           
│       │   ├── test.bed     # Test set for the first data group (BED format)
│       │   ├── train.bed    # Training set for the first data group (BED format)
│       │   └── val.bed      # Validation set for the first data group (BED format)
│       └── data_2          
│           ├── test.bed     # Test set for the second data group (BED format)
│           ├── train.bed    # Training set for the second data group (BED format)
│           └── val.bed      # Validation set for the second data group (BED format)
│
├── images                   # Container image directory
│   └── autogluon.sif        # Singularity image for Autogluon
│
├── main.nf                  # Main Nextflow pipeline script, defines the entire bioinformatics workflow
│
├── nextflow.config          # Nextflow configuration file, sets pipeline parameters, execution environment, and container paths
│
├── README.md                # Project documentation, including usage instructions and workflow descriptions
│
├── results                  # Directory for output results, generated after workflow execution
│
└── scripts                  # Custom scripts directory
    └── at.py                # Python script for handling autogluon training tasks
```

## Usage
### 1. Prepare Data Files
Ensure your raw methylation data is formatted according to the expected format. The format should contain **six columns** with the following headers:
```
chr    start    end    seq    tag    label
```
- **chr** – Chromosome  
- **start** – Start position  
- **end** – End position  
- **seq** – DNA sequence with or without base 'M'
- **tag** – Additional metadata or identifier  
- **label** – Class or label for the sequence  

Place the prepared data files in the appropriate directory for processing.

### 2. Prepare Metadata File
Create a meta.tsv file according to the format specified in `assets/meta.tsv`, including the necessary datasets and parameters for each run.

### 3. Execute the Pipeline
```bash
nextflow run main.nf \
    -profile singularity \
    --input meta.tsv \
    --outdir <OUTDIR> \
    -bg 
```

## Cleaning Up Temporary Files
After execution, clean up intermediate files to free disk space:
```bash
nextflow clean -f
```

## Notes
- Ensure that Nextflow and Singularity are properly installed and configured before running the pipeline.
- Review the logs in the output directory for troubleshooting and performance evaluation.
