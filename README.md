# scAnnoAKM

Code for agent-driven, knowledge-enhanced single-cell annotation.

## Main Workflow

- `DPA/`: ranked transcriptomic evidence sequences.
- `KRA/`: structured marker, tissue, and role knowledge.
- `QVA/`: knowledge validation.
- `scAnnoModel/`: model, training, and prediction.
- `driver.py`: main entry point.
- `dataset.py`: data loading and reference-target splits.

Set `QWEN_API_KEY` before running `driver.py`. Under `datasets/<name>/`, provide `data.h5ad`, `knowledge/<cell_type>/{marker,description}.txt`, and `knowledge/gene_knowledge.json`.

## Demo

Python 3.11 or later.

```bash
cd demo
python -m pip install -r requirements.txt
bash run_demo.sh
```

The Retina demo is an exploratory example.

## Pretrained Encoders

Place `Geneformer-V2-104M/` and `token_dictionary_gc104M.pkl` under `scAnnoModel/geneformer/`. Place BioBERT model and tokenizer files under `scAnnoModel/biobert/biobert_model/` and `scAnnoModel/biobert/biobert_tokenizer/`.
