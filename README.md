# scAnnoAKM

This repository contains the code and curated paper assets for scAnnoAKM, an agent-driven knowledge-enhanced framework for fine-grained single-cell annotation.

## Main Workflow

- `DPA/`: data processing agent for ranked transcriptomic evidence sequences.
- `KRA/`: knowledge retrieval agent for structured marker, tissue, and role knowledge.
- `QVA/`: quality validation agent for structured knowledge checking.
- `load/`: knowledge and sequence construction utilities.
- `scAnnoModel/`: main scAnnoAKM model, training, pre-training, and prediction code.
- `driver.py`: main experiment entry point.
- `dataset.py`: dataset loading and reference-target split utilities.

## Datasets
Datasets. The single-cell transcriptomic datasets used in this study were collected from publicly available cell-by-gene expression resources. For each dataset, the original expression matrix, cell-type annotations, and available metadata were organized under the datasets/ directory. Dataset-specific source information is provided in the corresponding info.txt files when available.

## Geneformer
Geneformer. The Geneformer backbone used in scAnnoAKM follows the publicly available implementation from the official Geneformer repository: https://github.com/jkobject/geneformer. The pretrained Geneformer model is used to encode ranked gene sequences into transcriptomic representations for downstream annotation.

## BioBERT
BioBERT. The biomedical text encoder used in scAnnoAKM is based on BioBERT, which is used to encode tissue-context and functional-role descriptions into semantic embeddings. The BioBERT pretrained model files should be placed under scAnnoModel/biobert/biobert_model/ before running scAnnoAKM. The original BioBERT project is available at https://github.com/dmis-lab/biobert.
