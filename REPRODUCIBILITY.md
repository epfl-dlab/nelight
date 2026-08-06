# Reproducibility

Paper: [Strong Heuristics for Named Entity Linking](https://aclanthology.org/2022.naacl-srw.30/)
(Čuljak et al., NAACL SRW 2022).

## What you need

| Goal | Command | Also download |
|---|---|---|
| Rebuild Tables 1–11 | `uv sync` then `bash scripts/reproduce_all.sh` | nothing (use Git LFS files in the repo) |
| Re-run popularity / text-overlap heuristics | `uv sync --extra from-scratch` | NLTK data (command below) |
| Re-run embedding heuristics | same, plus `--with-embeddings` | nothing extra (embeddings are in `caches/` via LFS) |
| Re-run mGENRE on a GPU | `bash scripts/setup_mgenre.sh` | handled by the setup script (~6 GB) |
| Rebuild entity caches from scratch | `uv sync --extra from-scratch` | [Wikidata dump](https://dumps.wikimedia.org/wikidatawiki/entities/); optional [PageRank files](#pagerank) |

Install [uv](https://docs.astral.sh/uv/) once. This project uses Python **3.12**
(`.python-version`) except live mGENRE, which uses its own Python **3.10** env.

```bash
git clone https://github.com/epfl-dlab/nelight.git
cd nelight
git lfs install && git lfs pull
```

---

## 1. Rebuild the paper tables (default)

Uses the saved score files from the original experiments. No GPU.

```bash
uv sync
bash scripts/reproduce_all.sh
```

Output: `artifacts/all_paper_tables.json`.

| Table | Contents |
|---|---|
| 1 | Easy / hard / overall mention counts |
| 2 | Main accuracy (P@1) |
| 3 | AIDA accuracy by entity type |
| 4–11 | Appendix (errors, ablations, context size, tie-breakers, runtimes, MRR) |

To also recompute heuristics from the shipped caches and check them against
Table 2: `bash scripts/audit_reproducibility.sh` (slower; needs GPU optional
only for the embedding step).

---

## 2. Re-run heuristics yourself

```bash
uv sync --extra from-scratch
uv run --extra from-scratch python -c \
  "import nltk; [nltk.download(x, quiet=True) for x in
   ('punkt','punkt_tab','wordnet','omw-1.4','stopwords')]"

# Popularity + text overlap (IScore, NIScore, EEIScore, UIScore)
uv run --extra from-scratch python scripts/run_heuristics.py --dataset both

# Also embedding methods (CSE, NCSE, CSSVE, UCSE)
uv run --extra from-scratch python scripts/run_heuristics.py \
  --dataset both --with-embeddings
```

Reads `caches/{quotebank,aida}/`. On Quotebank, ties are broken with a
popularity score, then by Wikidata id. On AIDA, the highest score wins as-is.

**AIDA note:** rebuilt embedding caches match CSE closely; CSSVE can differ by
about one percentage point from the paper. Table scripts still use the original
AIDA embedding scores in `score_cache/` for exact Table 2 numbers.

To rebuild the BART embedding files:

```bash
uv run --extra from-scratch python cache_building/build_text_embeddings.py entity \
  --entity-kb caches/aida/entity_kb.pkl --out caches/aida/entity_embeddings.pkl
# likewise: document / mention modes — see cache_building/README.md
```

---

## 3. Re-run mGENRE on a GPU

Saved scores (no GPU) are enough for the tables:

```bash
uv run python scripts/convert_mgenre_raw.py
```

Live GPU run (validated to match the paper’s P@1: Quotebank **0.963**, AIDA **0.682**):

```bash
bash scripts/setup_mgenre.sh    # once: code + model files + Python 3.10 env
source scripts/mgenre_env.sh
python scripts/run_mgenre.py --dataset quotebank --context 128 --device cuda:0
python scripts/run_mgenre.py --dataset aida --context 256 --device cuda:0
```

Writes `artifacts/from_scratch/.../mGENRE_live_t*.pkl` (does not overwrite the
saved paper scores). Needs a GPU and tens of GB of RAM. Roughly 20 minutes for
Quotebank and 45–60 minutes for AIDA after a few minutes of model load.

---

## 4. Eigenthemes

Table 2 Eigen numbers are already in
`artifacts/from_scratch/{quotebank,aida}/ranked_scores.pkl`.

Re-running Eigenthemes needs the original research tree (DeepWalk embeddings and
candidate lists). Those files are **not** redistributed here. If you have them:

```bash
# place at workspace/eigenthemes/
uv run python scripts/run_eigenthemes.py --dataset both --variant both
```

---

## PageRank

The shipped entity caches already include Wikipedia PageRank (**PRWP**) and
Wikidata PageRank (**PRWD**). Only needed if you rebuild a cache from a dump.

- **PRWP:** public 2021 ranks from
  [danker](https://danker.s3.amazonaws.com/index.html), e.g.
  `wget https://danker.s3.amazonaws.com/2021-11-15.allwiki.links.rank.bz2`
- **PRWD:** compute with [danker](https://github.com/athalhammer/danker)

Format: TSV `qid_number<TAB>score` (no leading `Q`). Pass with
`--wp-ranks` / `--wd-ranks` to `cache_building/build_entity_kb.py`.

---

## Mistakes in the paper PDF

These are print errors, not method bugs. Scripts target the “Reproduced” values.

| Issue | Printed | Reproduced |
|---|---|---|
| Quotebank NIScore overall (Table 2) | 0.851 | **0.898** |
| AIDA NIScore overall (Table 2) | 0.562 | **0.589** |
| AIDA Eigen easy (Table 2) | 0.859 | **0.858** |
| Table 11 AIDA MRR rows (several methods) | scrambled | same scores as Table 2 |
| Table 8 / 11 AIDA mGENRE MRR | 0.720 / 0.730 | **0.743 / 0.736** |
| Table 9 UIScore + PRWP | 0.942 | **0.943** |

Table 4 category labels are manual (we check the count of 14 errors). Table 10
runtimes are hardware-specific.

---

## Layout

```
data/            evaluation sets
score_cache/     saved method scores from the paper runs
caches/          entity DBs + embeddings (Git LFS)
scores/          popularity baselines
results/         IScore ablation numbers (Table 6)
scripts/         reproduction and runners
runlib/          scoring code
cache_building/  rebuild caches from Wikidata
artifacts/       script outputs
paper/           PDF + parsed tables
```

Local only (not in git): `.venv/`, `.venv-mgenre/`, `models/mgenre/`,
`third_party/` (created by setup scripts).
