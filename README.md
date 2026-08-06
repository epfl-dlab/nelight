# NELight

Code and evaluation for
**[Strong Heuristics for Named Entity Linking](https://aclanthology.org/2022.naacl-srw.30/)**
(Čuljak et al., NAACL SRW 2022).

## Rebuild the paper tables

The repo includes the score files from the original experiments. To recompute
Tables 1–11 (including AIDA entity types and the appendix):

```bash
git lfs install && git lfs pull
pip install -r requirements-repro.txt
bash scripts/reproduce_all.sh
```

No GPU. Details, how to re-run each method yourself, installs, and known typos
in the PDF: **[REPRODUCIBILITY.md](REPRODUCIBILITY.md)**.

## Re-running methods (optional)

| Method | Command | Install |
|---|---|---|
| Popularity, I/NI/EEIScore, UIScore | `PYTHONPATH=runlib python3 scripts/run_heuristics.py --dataset both` | `requirements-from-scratch.txt` |
| CSE / NCSE / CSSVE / UCSE | same, with `--with-embeddings` | + embedding caches via LFS |
| mGENRE (use saved scores) | `python3 scripts/convert_mgenre_raw.py` | `requirements-repro.txt` |
| mGENRE (run the model again) | `python scripts/run_mgenre.py --dataset quotebank --context 128` | see REPRODUCIBILITY.md §5 |
| Eigenthemes | `python3 scripts/run_eigenthemes.py --dataset both --reuse-raw` | see REPRODUCIBILITY.md §6 |

| Path | Contents |
|---|---|
| `paper/tables/paper_tables.json` | Numbers as printed in the paper |
| `artifacts/all_paper_tables.json` | What the scripts produce |
| `caches/{quotebank,aida}/` | Entity KB and BART embeddings (Git LFS) |
| `score_cache/raw/` | Saved per-method scores |
| `cache_building/` | Rebuild caches from a Wikidata dump |
