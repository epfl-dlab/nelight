# Reproducing *Strong Heuristics for Named Entity Linking*

[Čuljak et al., NAACL SRW 2022](https://aclanthology.org/2022.naacl-srw.30/)

## Two ways to reproduce

**1. Evaluate saved scores (default).**  
When the paper was written, each method wrote out a score file per mention
(pickles under `score_cache/`). The table scripts just load those files and
compute P@1 / MRR. That is enough to rebuild Tables 1–11 and match the paper
(except the print mistakes listed at the end). No GPU.

**2. Re-run a method yourself.**  
You can also recompute scores with the code in this repo (heuristics, BART
embeddings, mGENRE, Eigenthemes). That is slower and may need a GPU. For
mGENRE especially, a new run on a modern stack will not produce *identical*
floating-point beams to 2022, but the chosen entity is almost always the same.
Use the saved score files if you care about matching the published digits.

---

## Setup

```bash
git clone git@github.com:epfl-dlab/nelight.git
cd nelight
git lfs install
git lfs pull
```

### What to install

| What you want to do | Install | Also need |
|---|---|---|
| Rebuild all paper tables from saved scores | `pip install -r requirements-repro.txt` | files already in the repo (`score_cache/`, `scores/`, `data/`, `caches/` via LFS) |
| Re-run heuristics / embedding scorers | `pip install -r requirements-from-scratch.txt` and download NLTK data (see §3) | `caches/` |
| Rebuild BART embedding caches | same as heuristics | a GPU helps |
| Re-run mGENRE on GPU | see §5 (fairseq + GENRE + downloads) | lots of RAM for the title map |
| Re-run Eigenthemes | see §6 | `workspace/eigenthemes/` from the original research tree |

Use Python 3.9+ for the table scripts and heuristics. mGENRE’s fairseq fork is
happiest on 3.8–3.10.

---

## Rebuild Tables 1–11 from saved scores

```bash
pip install -r requirements-repro.txt
bash scripts/reproduce_all.sh
```

Output: `artifacts/all_paper_tables.json`.

| Table | What it is |
|---|---|
| 1 | Easy / hard / overall mention counts |
| 2 | Main P@1 results |
| 3 | AIDA P@1 by entity type (PER / ORG / LOC / MISC) |
| 4 | UIScore errors on Quotebank (we check the count; the category labels in the paper are manual) |
| 5 | How ambiguous Quotebank gold mentions are |
| 6 | IScore ablation (which Wikidata fields / normalization) |
| 7 | Narrow vs full-document context for CSE and IScore |
| 8 | mGENRE at different context widths |
| 9 | Which popularity signal to use as a tie-breaker |
| 10 | Runtime on the paper’s hardware (we only report those numbers) |
| 11 | MRR for the same methods as Table 2 |

---

## Re-run heuristics (popularity, IScore, NIScore, EEIScore, UIScore)

```bash
pip install -r requirements-from-scratch.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('stopwords')"

PYTHONPATH=runlib python3 scripts/run_heuristics.py --dataset both
python3 scripts/reproduce_paper_from_scratch.py
```

This reads `caches/{quotebank,aida}/entity_kb.pkl` and `unambiguous_mentions.pkl`.
On Quotebank, ties are broken with a popularity score then LQID (see §7).
On AIDA there is no popularity tie-break—just `argmax`.
Scores land in `artifacts/from_scratch/{quotebank,aida}/`.

---

## Re-run embedding scorers (CSE, NCSE, CSSVE, UCSE)

```bash
PYTHONPATH=runlib python3 scripts/run_heuristics.py --dataset both --with-embeddings
```

Needs the BART caches in `caches/*/` (`entity_embeddings.pkl`,
`document_embeddings.pkl`, `mention_embeddings.pkl`).

For AIDA, the Table 2 comparison script still loads the original CSE/CSSVE/UCSE
score files from `score_cache/raw/AIDA/`. Recomputing from our rebuilt AIDA
embeddings matches CSE and UCSE; CSSVE can be off by about one point. Quotebank
embedding scores match the paper from the shipped caches.

### Optional: rebuild the BART caches

```bash
python cache_building/build_text_embeddings.py document \
  --data data/AIDA/data.json --out caches/aida/document_embeddings.pkl --device cuda:0
python cache_building/build_text_embeddings.py mention \
  --data data/AIDA/data.json --out caches/aida/mention_embeddings.pkl --device cuda:0
python cache_building/build_text_embeddings.py entity \
  --entity-kb caches/aida/entity_kb.pkl --out caches/aida/entity_embeddings.pkl --device cuda:0
```

Uses `facebook/bart-base`. We store a mean-pooled vector per text
(`[n, 1, H]`), which is what the scorers use anyway. To rebuild entity KBs from
a Wikidata dump, see `cache_building/README.md`.

---

## mGENRE

### Using the saved mGENRE scores (matches the paper)

The original runs already saved candidate scores. Copy them into
`artifacts/from_scratch/` (and optionally rebuild Quotebank scores from the
raw beam file to check the conversion):

```bash
pip install -r requirements-repro.txt
python3 scripts/convert_mgenre_raw.py
```

Best context widths in the paper: Quotebank **128** tokens (P@1 0.963),
AIDA **256** (P@1 0.682).  
Files: `score_cache/raw/genre_context_scores_qb.pkl`,
`genre_context_scores_aida.pkl`, and optionally `genre_context_scores_all.pkl`
(raw beams).

### Running mGENRE again on a GPU

This loads the public mGENRE checkpoint and scores every mention again. You need
a GPU, a fairseq build with constrained decoding, and several large downloads.

How the paper scored mentions:

1. Wrap the mention in `[START]` / `[END]`, keep at most `t` mBART tokens on each side.
2. Beam search with beam size 10. Do **not** turn on `marginalize`.
3. Map each hypothesis `title >> lang` to a Wikidata id by taking the max QID
   (`max(ids, key=lambda y: int(y[1:]))`).
4. Score = `exp(log-likelihood)`; candidates the model never proposes get 0.
5. On Quotebank, sum scores across mention offsets the same way as notebook
   `aa.ipynb` (the `cache.add` quirk in cell 10).

Install:

```bash
git clone --branch fixing_prefix_allowed_tokens_fn https://github.com/nicola-decao/fairseq
cd fairseq && pip install --editable ./ && cd ..
git clone https://github.com/facebookresearch/GENRE.git
# make the `genre` package importable, e.g. pip install -e GENRE
pip install -r requirements-mgenre.txt
```

```bash
export FAIRSEQ_ROOT=/path/to/fairseq
source scripts/mgenre_env.sh   # puts fairseq on PYTHONPATH
```

Download into `models/mgenre/`:

```bash
mkdir -p models/mgenre && cd models/mgenre
wget https://dl.fbaipublicfiles.com/GENRE/fairseq_multilingual_entity_disambiguation.tar.gz
tar -xzf fairseq_multilingual_entity_disambiguation.tar.gz
wget https://dl.fbaipublicfiles.com/GENRE/titles_lang_all105_marisa_trie_with_redirect.pkl
wget https://dl.fbaipublicfiles.com/GENRE/lang_title2wikidataID-normalized_with_redirect.pkl
cd ../..
```

The title→QID map alone needs on the order of tens of GB of RAM.

```bash
python scripts/run_mgenre.py --dataset quotebank --context 128 --device cuda:0
python scripts/run_mgenre.py --dataset aida --context 256 --device cuda:0
python3 scripts/eval_from_scratch.py
```

Expect predictions to agree with the saved scores on nearly all mentions, but not
byte-for-byte log-likelihoods. For the numbers printed in the paper, use the
saved score files above.

---

## Eigenthemes

Put the original Eigenthemes tree at `workspace/eigenthemes/` (DeepWalk
embeddings plus the candidate JSON lists). Then:

```bash
python3 scripts/run_eigenthemes.py --dataset both --variant both --reuse-raw
```

- **Eigen** — weighted eigen (`weigen`), `meanCenter=False`, 20 candidates, 10 components.  
  Quotebank uses NS weights and NS→LQID tie-break; AIDA uses degree weights and plain `argmax`.
- **Eigen (IScore)** — same, but on the historical `*_iscore_*_test_complete.json`
  lists, with NS filled in where DeepWalk has no vector.

---

## Wikipedia and Wikidata PageRank

The paper’s popularity baselines **PRWP** and **PRWD** are PageRank scores
stored on each entity in `entity_kb.pkl` as `pagerank` and `pagerank_wd`.
The shipped caches already include them. To rebuild a KB from scratch you need
the rank files below, then pass them into `cache_building/build_entity_kb.py`.

### PRWP — Wikipedia PageRank (field `pagerank`)

We used Andreas Thalhammer’s publicly released Wikipedia PageRank scores from
**2021** ([danker releases](https://danker.s3.amazonaws.com/index.html),
CC BY-SA 3.0). These are PageRank over the Wikipedia link graph, keyed by
Wikidata id.

A dump aligned with the paper’s Wikidata snapshot:

```bash
wget https://danker.s3.amazonaws.com/2021-11-15.allwiki.links.rank.bz2
bunzip2 2021-11-15.allwiki.links.rank.bz2
```

Newer dated files on the same page work the same way if you do not need a 2021
match.

### PRWD — Wikidata PageRank (field `pagerank_wd`)

We computed Wikidata PageRank ourselves with
[danker](https://github.com/athalhammer/danker) (same tooling Thalhammer uses
for the public Wikipedia ranks). Follow the danker README to build a link file
from a Wikidata / Wikipedia dump and run PageRank; write a TSV of scores.

### File format and attaching ranks

Both rank files are TSV lines `qid_number<TAB>score` (no leading `Q`),
optionally with a header line. Example: `42	0.00031`.

```bash
python cache_building/build_entity_kb.py \
  --dump /path/to/wikidata_subgraph_or_dump.json.gz \
  --labels-dir /path/to/entity_metadata \
  --qids /path/to/candidate_qids.pkl \
  --wp-ranks /path/to/2021-11-15.allwiki.links.rank \
  --wd-ranks /path/to/wikidata.ranks \
  --out caches/quotebank/entity_kb.pkl
```

`--wp-ranks` fills `pagerank` (PRWP); `--wd-ranks` fills `pagerank_wd` (PRWD).
See `cache_building/README.md` for the full KB pipeline.

---

## Scoring rules (short)

**Quotebank.** After the method score, break ties with a popularity feature, then
LQID. IScore uses NP→LQID; EEIScore and UIScore use PRWP→LQID; most others use
NS→LQID.  
UIScore = IScore + NIScore + EEIScore with weights (1, 1, 1).  
UCSE (paper 0.882) = transformed CSE / NCSE / CSSVE with weights (0.45, 0.9, 0.2)
— CSE gets ½(x+1), NCSE and CSSVE get a Laplacian `(x+1)/sum(x+1)`.

**AIDA.** No popularity tie-break.  
UIScore weights (0.9, 0, 1).  
UCSE = ½(NCSE+1) + Laplacian(CSSVE) with weights (0, 1, 1).

---

## Mistakes in the paper PDF

These are typos or scrambled cells, not bugs in the methods. Our scripts aim for
the “Reproduced” column.

| Issue | Printed | Reproduced | Notes |
|---|---|---|---|
| Quotebank NIScore overall (Table 2) | 0.851 | **0.898** | Impossible given easy/hard 0.966 / 0.571 on a 203+42 split |
| AIDA NIScore overall (Table 2) | 0.562 | **0.589** | Easy/hard match the paper; overall looks copied from EEIScore |
| AIDA Eigen easy (Table 2) | 0.859 | **0.858** | Overall 0.617 matches |
| Table 11 AIDA MRR for CSE / EEIScore / CSSVE / UCSE / NIScore | see PDF | from the same score files as Table 2 | Printed MRR rows look swapped |
| Table 8 / 11 AIDA mGENRE MRR | 0.720 / 0.730 | **0.743 / 0.736** | P@1 matches; printed MRR is too low |
| Table 9 UIScore + PRWP | 0.942 | **0.943** | Same run as Table 2 UIScore |

Table 4’s error *categories* are hand labels (we only check that there are 14
UIScore mistakes). Table 5’s counts are taken from the paper and checked for
internal consistency. Table 10 is whatever machine they timed on. Table 3
compares P@1 only, not the bootstrap intervals.

---

## Repo layout

```
data/                  evaluation JSON (Quotebank, AIDA, entity types)
scores/                popularity scores + IScore ablation
score_cache/raw/       saved method scores from the original runs
caches/                entity KB, unambiguous mentions, BART embeddings (Git LFS)
scripts/               table builders and re-runners
runlib/                scorers used by the heuristics
cache_building/        rebuild caches from a Wikidata dump
models/mgenre/         mGENRE weights (you download these; not in git)
workspace/eigenthemes/ DeepWalk + Eigen inputs (optional; not in git)
paper/                 PDFs + parsed table JSON
artifacts/             script output
```
