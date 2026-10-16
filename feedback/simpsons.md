# Simpson's paradox and confounding

**What was planted.** Twelve pooled comparisons that reverse within strata of a confounder: two real reversals found in QRData sheets (medication vs recovery confounded by severity; treatment vs hospital stay confounded by hospital) and five published tables (kidney stones by stone size, Berkeley admissions by department, Florida death penalty by victim race, Jeter vs Justice by season, Whickham smoking by age), each as a pooled table and as an unadjusted regression.

**How the models failed.** (catches out of 4 models per case; see results/tables.md)
- The best caught type overall (31 of 48 reviews). The pooled-table framings were caught most: the medication and hospital sheets and the Berkeley table 4 of 4, kidney stones, Florida and batting 3 of 4.
- The regression framings were caught less (2 of 4 for the medication and Berkeley regressions, 0 of 4 for the Whickham logistic regression). On Whickham all four reviews went after the grouped data instead: the formula `'dead + alive ~ smoker'`, calling it an invalid outcome or saying logistic regression cannot use aggregated counts. That is statsmodels' standard grouped binomial syntax, so the review spent its one issue on a non-problem and missed age.
- Reviews of the binary-outcome version complained that dichotomizing recovery time loses information, which is true but not why the conclusion is wrong.
- On the clean twins, models sometimes argued against a correct stratified analysis, e.g. that comparing within severity bands "doesn't eliminate the effect of severity". That is backwards: stratifying is exactly what removes it.

**Corrected reasoning.** Before reading a pooled difference, check how the groups differ on things that also drive the outcome. If treatment assignment depends on severity and severity drives recovery, compare within severity (stratify, Mantel-Haenszel) or adjust for it in a regression. Report the adjusted effect and say which way the pooled number was biased.

**Corrected code** (simpsons_02):

```python
m = smf.ols('recovery ~ medication + severity + age + sex', data=df).fit()
# medication coefficient: -16.90 days (helps). Without severity it was +6.52 (looks harmful).
```

**Feedback to give a model.** "Name the confounder. Severity drives both who gets the drug and how long recovery takes; within severity bands the medicated patients recover faster."
