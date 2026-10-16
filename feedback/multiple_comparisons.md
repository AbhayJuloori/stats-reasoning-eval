# Uncorrected multiple comparisons

**What was planted.** Twelve subgroup analyses on the Hillstrom data (history segment, zip x channel, channel x newbie, recency month, and other splits) that ran 6 to 12 tests and reported every cell with p < 0.05 as a finding. In each case a Holm correction removes some or all of the hits.

**How the models failed.**
- This was the hardest type for every model. The words "multiple", "Bonferroni", "Holm" or "false discovery" almost never appeared.
- Instead the models invented design problems that are not there: "the analysis ignores the third arm and it may confound the results", "the email was randomized so you should not compare within segments". Randomization makes within-segment comparisons valid; the issue is how many of them were run.
- llama3.1:8b called all twelve sound.

**Corrected reasoning.** With k independent tests at alpha = 0.05, the chance of at least one false positive is 1 - 0.95^k (about 37% for 9 tests). Report how many tests were run, correct for them (Holm controls the family-wise error rate and is never worse than Bonferroni; Benjamini-Hochberg controls the false discovery rate), or pre-register the subgroups and test an interaction term instead of slicing.

**Corrected code** (multiple_comparisons_02):

```python
from statsmodels.stats.multitest import multipletests
reject, p_holm, _, _ = multipletests([p for _, p in results], alpha=0.05, method='holm')
# 7 history segments tested: 5 raw hits, 1 survives Holm (adjusted p = 0.045)
```

**Feedback to give a model.** "The comparisons within segments are fine because assignment was randomized. The problem is that seven tests were run and only the winners reported; after a Holm correction one of the five hits survives."
