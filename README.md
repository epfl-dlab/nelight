# NELight

Source code and evaluation for
**[Strong Heuristics for Named Entity Linking](https://aclanthology.org/2022.naacl-srw.30/)**
(Čuljak et al., NAACL SRW 2022).

## Reproduce paper tables

```bash
bash scripts/reproduce_all.sh
```

See **[REPRODUCIBILITY.md](REPRODUCIBILITY.md)** for protocol notes and
**known paper errors** (NIScore overall typos; Table 11 AIDA MRR scramble).

| Resource | Path |
|---|---|
| Parsed paper tables | `paper/tables/paper_tables.json` |
| Reproduction output | `artifacts/all_paper_tables.json` |
| Caches (Git LFS) | `caches/{quotebank,aida}/` |
| Score dumps | `score_cache/raw/` |
| Cache builders | `cache_building/` |

Large `*.pkl` under `caches/` use [Git LFS](https://git-lfs.com). After clone:

```bash
git lfs install
git lfs pull
```
