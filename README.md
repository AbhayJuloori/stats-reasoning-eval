# Statistics reasoning eval set for LLMs

An evaluation set for how well language models handle applied statistics: running the right
test on real data, and spotting methodology mistakes in someone else's analysis. Built on two
public sources, the [QRData](https://github.com/xxxiaol/QRData) benchmark sheets and the
[Hillstrom MineThatData](https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html)
email experiment (64,000 customers, randomized into three arms).

**Results page with a browsable explorer of all 60 flawed answers:** https://abhayjuloori.github.io/stats-reasoning-eval/

Everything runs on free models: four open models served locally with Ollama, plus one free
OpenRouter model as a second judge.

## What is in it

| Part | Size | Where |
|---|---|---|
| Data science prompts with reference solutions in pandas / statsmodels | 150 (99 on Hillstrom, 51 on QRData sheets) | `prompts/prompts.jsonl`, built by `evalset/build_prompts.py` |
| QRData benchmark items (stratified sample, official grading rule) | 100 of 411 | `evalset/run_solve.py --set qrdata` |
| Worked answers with one planted methodology flaw | 60 (12 each: peeking, target leakage, Simpson's paradox, wrong test, uncorrected multiple comparisons) | `prompts/worked_answers.jsonl`, built by `evalset/build_flaws.py` |
| Corrected twin of every flawed answer | 60 (15 used as clean controls in the review set) | same file |
| Grading rubric | 0 / 1 / 2 per review | `rubric/RUBRIC.md` |
| Feedback per failure type, with corrected reasoning and code | 5 | `feedback/` |

Every reference answer and every number inside a worked answer comes from executing the code
shown, so the flawed answers read like an analyst's honest write up of a broken method.

## How it works

1. **Solving.** The model gets a data preview and a question, writes a Python script, and the
   script runs in a subprocess. If it crashes, the traceback goes back once for a fix. The last
   printed line is graded against the reference with a per-item tolerance (`evalset/grading.py`).
2. **Reviewing.** The model reviews a worked answer and returns `VERDICT: SOUND|FLAWED`,
   `ISSUE`, `FIX`. The prompt never says a flaw was planted, and clean controls look the same.
3. **Judging.** An LLM judge (phi4:14b) scores each review on the rubric. Two versions:
   v1 hands the judge the whole rubric; v2 parses the verdict in code and asks the judge one
   yes/no question (did the review name the planted mechanism?).
4. **Judge checks.** Judge scores are compared with hand grades using Cohen's kappa
   (`evalset/hand_grade.py`), and the judge is shown each flawed / clean pair in both orders to
   measure position bias.

## Results

Full tables: [`results/tables.md`](results/tables.md). All numbers below come from
`python -m evalset.analyze` over the run files in `results/`.

### Catching planted flaws (judge v2, 60 flawed answers + 15 clean controls per model)

| model | names the planted flaw | flags it as flawed | flags clean controls as flawed |
|---|---|---|---|
| gemma3:12b | 60% | 98% | 100% |
| qwen3:14b | 57% | 100% | 100% |
| qwen3:8b | 45% | 100% | 100% |
| llama3.1:8b | 8% | 50% | 20% |

- **Flag rate on its own is meaningless here.** Three of the four models called every clean
  control flawed too, so a review set without controls would have scored them near perfect.
  The useful number is whether the review names the actual mechanism.
- **Multiple comparisons was nearly invisible:** 1 catch in 48 reviews. Models invented design
  problems ("the analysis ignores the third arm") instead of counting the tests.
- **Simpson's paradox was caught most** (31 of 48), mainly in the pooled-table framings. The
  regression framings did worse; on the Whickham logistic regression all four models attacked
  valid grouped binomial syntax and missed age as the confounder.

### Judge quality

| | v1 (full rubric) | v2 (verdict in code + yes/no) |
|---|---|---|
| judge scores that contradict the review's own verdict | 26% | 0% (by construction) |
| Cohen's kappa vs hand grades | see `results/tables.md` | see `results/tables.md` |

### Position bias (flawed vs corrected twin, both orders)

| judge | accuracy | correct when clean shown first | correct when clean shown second |
|---|---|---|---|
| phi4:14b (local) | 81% | 63% | 98% |
| nemotron-3-super-120b (OpenRouter free, 20 pairs) | 100% | 100% | 100% |

The smaller judge leans toward whichever answer it reads second; swapping the order exposes it.

### Solving (model writes a script, the script runs, the printed answer is graded)

| model | 150 written prompts | QRData sample (99 items) | QRData statistical | QRData causal |
|---|---|---|---|---|
| qwen3:8b | 65% | 53% | 60% | 48% |
| qwen3:14b | 62% | 54% | 66% | 47% |
| gemma3:12b | 60% | 47% | 57% | 42% |
| llama3.1:8b | 55% | 31% | 37% | 28% |

- The QRData paper reports 37% for its best open model and 58% for GPT-4 on all 411 items.
  The qwen3 and gemma3 models here are about a year newer than the paper's; their 95% intervals
  on 99 items are wide (qwen3:14b: 44% to 63%). Reference answers never appear in the prompt.
- Causal questions are the weak spot, as in the paper. Most are two-way multiple choice, and
  llama3.1:8b scored 23 of 60 multiple choice items against 22.8 expected from guessing.
- Of 236 misses on the written prompts, 114 were scripts that still failed after one repair turn
  (13% to 26% of prompts per model). The top causes were invented APIs (importing
  `proportions_ztest` or `ztest` from `scipy.stats`, 29 times) and passing pandas boolean dummy
  columns straight into statsmodels (21). The other misses printed a wrong number.
- Power analysis was missed by every model on every prompt (0 of 28 for sample size, 1 of 20
  for minimum detectable effect). In 19 of the 28 sample size scripts the model passed the raw
  difference in rates (e.g. 0.003) as the effect size, though `NormalIndPower` expects Cohen's h
  and the prompt asked for it; some scripts did this right under a comment defining Cohen's h.
  7 more crashed.

## Running it

```bash
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
./scripts/get_data.sh                      # QRData (CC BY-NC 4.0) and Hillstrom, not committed
.venv/bin/python -m evalset.build_prompts  # runs every reference solution
.venv/bin/python -m evalset.build_flaws    # runs every flawed and corrected analysis
ollama pull qwen3:8b && ollama pull llama3.1:8b && ollama pull gemma3:12b && ollama pull qwen3:14b && ollama pull phi4:14b
./scripts/run_mac.sh                       # reviews, judge, position bias
.venv/bin/python -m evalset.run_solve --set prompts --models ollama/qwen3:8b
.venv/bin/python -m evalset.hand_grade     # hand grades for the kappa check
.venv/bin/python -m evalset.analyze
.venv/bin/python -m evalset.export_site    # refreshes docs/data.json for the results page
.venv/bin/python -m pytest -q tests
```

The solve sets were run on a Kaggle 2x T4 notebook (`scripts/kaggle/`), one Ollama server per
GPU. On a 16 GB laptop keep one model in memory at a time; `evalset/llm.py` unloads the others
before each call.

## Limits

- 12 flawed answers per type and four small open models: per-type rates move a lot with one
  or two reviews (Wilson intervals are in `results/summary.json`).
- One judge model grades every review; its errors are measured against hand grades, not removed.
- Five Simpson's paradox cases use published tables (kidney stones, Berkeley admissions,
  Florida death penalty, Jeter vs Justice, Whickham) that models have likely seen in training.

## Data and licenses

QRData is CC BY-NC 4.0 and is downloaded by `scripts/get_data.sh`, not redistributed here.
The Hillstrom data is published by Kevin Hillstrom for public analysis.
