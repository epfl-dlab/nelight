# NELight

Source code and evaluation for
**[Strong Heuristics for Named Entity Linking](https://aclanthology.org/2022.naacl-srw.30/)**
(Čuljak et al., NAACL SRW 2022).

## Quick start (all paper tables)

```bash
git lfs install && git lfs pull
pip install -r requirements-repro.txt
bash scripts/reproduce_all.sh
```

This reproduces **Tables 1–11** (main + appendix), including the **type-based**
AIDA analysis (Table 3), from shipped score dumps. No GPU required.

## From-scratch methods

| Method family | Command | Deps |
|---|---|---|
| Popularity, I/NI/EEIScore, UIScore | `PYTHONPATH=runlib python3 scripts/run_heuristics.py --dataset both` | `requirements-from-scratch.txt` |
| CSE / NCSE / CSSVE / UCSE | add `--with-embeddings` | same + LFS embedding caches |
| mGENRE (exact dumps) | `python3 scripts/convert_mgenre_raw.py` | `requirements-repro.txt` |
| mGENRE (live GPU) | `python scripts/run_mgenre.py --dataset quotebank --context 128` | §5 in REPRODUCIBILITY.md |
| Eigenthemes | `python3 scripts/run_eigenthemes.py --dataset both --reuse-raw` | §6 in REPRODUCIBILITY.md |

Full install notes, asset download URLs, protocols, and **known paper errors**:
**[REPRODUCIBILITY.md](REPRODUCIBILITY.md)**.

| Resource | Path |
|---|---|
| Parsed paper tables | `paper/tables/paper_tables.json` |
| Reproduction output | `artifacts/all_paper_tables.json` |
| Caches (Git LFS) | `caches/{quotebank,aida}/` |
| Score dumps | `score_cache/raw/` |
| Cache builders | `cache_building/` |
