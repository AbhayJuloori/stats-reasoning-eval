# Target leakage

**What was planted.** Twelve models whose test score is inflated by information that will not exist at prediction time: post-outcome columns (visit, spend, conversion, paid_late_fees, balance, agreement), pricing fields derived from the label (interest_rate from grade), a target encoding fitted on all rows before the split, a school average computed with the test students, and a random split on a state-by-year panel used for forecasting.

**How the models failed.** (catches out of 4 models per case; see results/tables.md)
- Catches were spread fairly evenly across the Hillstrom, school and panel cases (1 to 3 of 4 each); the random split on a time panel was caught most (3 of 4).
- The loan cases were the hardest (0 to 1 of 4). On the delinquency model nobody named paid_late_fees or balance as repayment results recorded after the loan was issued. Reviews fell back on generic complaints: "class imbalance", "a single train/test split is not enough validation", "filling missing values with 0 may bias the model".
- Some reviews blamed the randomized design instead of the feature: "the model ignores that the email was a randomized experiment", when the problem was `visit` sitting in the feature list.
- Right diagnosis, wrong fix: for the target encoding case one model said to drop the feature entirely, when the fix is to fit the encoding inside the training fold.

**Corrected reasoning.** Ask one question per feature: is it known at the moment the prediction is made? Anything measured after the treatment or in the outcome window is out. Any statistic of the target (encodings, group means) must be computed from training rows only. For forecasting, split by time.

**Corrected code** (leakage_05, target encoding learned on training rows only):

```python
tr, te = train_test_split(df, test_size=0.3, random_state=0)
rates = tr.groupby('key')['conversion'].mean()      # fit on training rows only
prior = tr['conversion'].mean()
tr = tr.assign(key_rate=tr['key'].map(rates))
te = te.assign(key_rate=te['key'].map(rates).fillna(prior))
# test AUC drops from 0.896 (leaky) to 0.497: the key carried no real signal
```

**Feedback to give a model.** "Name the leaking feature and why it is unavailable at prediction time. 'Overfitting' is not the issue here: the AUC is high on held out data because the feature encodes the label."
