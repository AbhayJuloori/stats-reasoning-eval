# Wrong test choice

**What was planted.** Twelve analyses where the test does not fit the data: a z-test on 4 vs 0 conversions (needs an exact test), Pearson correlation on label-encoded nominal categories (three cases), a one-sample t-test against the control mean as if it were a known constant, chi-square on row percentages instead of counts, paired tests on independent arms matched by row order (three cases), independent tests on paired data (two cases), and a two-proportion z-test on the same voters measured twice (needs McNemar).

**How the models failed.** (catches out of 4 models per case; see results/tables.md)
- Paired vs independent was caught most (3 of 4 on four of the five cases), and the label-encoded correlations 2 to 3 of 4.
- The tiny-count z-test was caught by nobody (0 of 4): reviews said the first 100 rows were a biased sample, or that the conclusion compared the wrong arms, but never that 4 vs 0 events breaks the normal approximation.
- The one-sample test was caught once: most reviews said "a t-test assumes continuous, normal data and conversion is binary" (fine at this sample size) instead of "the control mean is an estimate, not a constant".
- Chi-square on percentages was caught by 2 of 4; the others argued about whether p = 0.17 was "close to significant" without noticing the table held percentages.
- llama3.1:8b called 5 of 12 sound.

**Corrected reasoning.** Match the test to the design: independent groups get two-sample tests; the same units measured twice get paired tests (paired t, Wilcoxon signed-rank, McNemar); tiny counts get exact tests; nominal predictors get group comparisons (chi-square, ANOVA), never a correlation on arbitrary codes; contingency tests take counts.

**Corrected code** (wrong_test_05):

```python
t = d.loc[d['segment'] == 'Womens E-Mail', 'conversion']
c = d.loc[d['segment'] == 'No E-Mail', 'conversion']
stats.ttest_ind(t, c, equal_var=False).pvalue   # 0.13, not the 0.046 from the one-sample test
```

**Feedback to give a model.** "The binary outcome is not the problem with this many customers. The one-sample test treats the control rate as known and ignores its sampling error, which halves the standard error you should be using."
