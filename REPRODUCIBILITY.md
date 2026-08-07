# Reproducibility

[Čuljak et al., NAACL SRW 2022](https://aclanthology.org/2022.naacl-srw.30/)

```bash
git clone https://github.com/epfl-dlab/nelight.git && cd nelight
git lfs install && git lfs pull
bash scripts/reproduce_all.sh
```

Needs [uv](https://docs.astral.sh/uv/) + Git LFS. No GPU. Writes `artifacts/all_paper_tables.json`.

## What the script recomputes (in-repo only)

| Step | Source in repo |
|---|---|
| Popularity + I/NI/EEI/UI + CSE family | `caches/` (LFS) via `run_heuristics.py` |
| Table 6 IScore ablation | `caches/quotebank/entity_kb{,_aliases}.pkl` |
| Eigen / Eigen(IScore) | shipped `artifacts/from_scratch/*/Eigen*_live_weigen.pkl` |
| mGENRE (Tables 2, 8) | `score_cache/raw/genre_context_scores_{qb,aida}.pkl` |
| Table 5 coarse buckets | `data/Quotebank/gt_annotation.json` |
| Splits / types | `data/` |

## Cannot be reproduced exactly (documented)

| Item | Why |
|---|---|
| **Table 10** timings | Paper hardware (GTX TITAN X / Xeon E5-2680); reported as-is |
| **Table 4** error *labels* | Manual qualitative categories; we only recompute the count (14) |
| **Table 5** fine null split (151/37/24/22) | Full checked annotation MDs were never archived; null total (234) and gold/unambiguous are reconstructed (±2 vs paper) |
| **Live Eigenthemes** | Needs the original DeepWalk tree (~2 GB, not in this repo). Use shipped Eigen pickles |
| **Live mGENRE** | Optional GPU path (`setup_mgenre.sh`); tables use shipped beam dumps |
| **AIDA CSSVE/UCSE exact floats** | Live rebuild from pooled BART caches drifts ~1 pp; tables use in-repo `score_cache/raw/AIDA/{cssve,ncse}_scores.pkl` for those two. CSE/NCSE still come from the live recompute |

## PDF typos (targets use corrected values)

| Printed | Correct |
|---|---|
| QB NIScore overall 0.851 | **0.898** |
| AIDA NIScore overall 0.562 | **0.589** |
| AIDA Eigen easy 0.859 | **0.858** (overall exact) |
| Several Table 11 AIDA MRR rows | Same scores as Table 2; PDF scrambled |
| Table 8/11 AIDA mGENRE MRR 0.720/0.730 | **0.743/0.736** |

## Optional re-runs

```bash
# Live Eigenthemes (external research tree)
ln -sfn /path/to/eigenthemes workspace/eigenthemes
uv sync --extra eigenthemes
uv run --extra eigenthemes python scripts/run_eigenthemes.py --dataset both

# Live mGENRE (GPU)
bash scripts/setup_mgenre.sh && source scripts/mgenre_env.sh
python scripts/run_mgenre.py --dataset quotebank --context 128
```

Own data / Wikidata dump rebuild: `cache_building/README.md`.
