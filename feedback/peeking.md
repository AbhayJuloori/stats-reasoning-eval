# Peeking (optional stopping)

**What was planted.** The analyst re-ran the A/B test after every mailing wave and stopped the first time p < 0.05, then reported that p-value as if it came from one planned test. Twelve cases on the Hillstrom data, each picked so the full pre-planned sample is *not* significant (example: urban Phone customers, Womens E-Mail, p = 0.008 at wave 1 vs p = 0.21 on the full sample).

**How the models failed.**
- Flagged the answer but blamed the wrong thing: "the z-test assumes independence", "conversion is binary so a z-test is wrong", "the No E-Mail group may not be a proper control". None of these hold for a randomized test with thousands of customers per arm.
- Mentioned early stopping as a trailing clause after a wrong main issue. The rubric gives that a 1: a reviewer has to say *why* stopping on significance breaks the p-value.
- llama3.1:8b called 8 of 12 sound.

**Corrected reasoning.** Each interim look is another chance to cross 0.05 by luck. With 10 looks the real false positive rate is several times the nominal 5%, so the p at the stopping wave is not a valid p-value and the effect at that wave is biased upward (you stop on a high). Either fix the sample size up front and test once, or plan the looks and spend alpha across them (O'Brien-Fleming or Pocock boundaries, or an always-valid sequential test).

**Corrected code** (peeking_06):

```python
df = pd.read_csv("hillstrom.csv").query("channel == 'Phone' and zip_code == 'Urban'")
t = df.loc[df['segment'] == 'Womens E-Mail', 'conversion'].to_numpy()
c = df.loc[df['segment'] == 'No E-Mail', 'conversion'].to_numpy()
# sample size fixed before launch; one test at the end
p = proportions_ztest([t.sum(), c.sum()], [len(t), len(c)])[1]
# diff = 0.23 pp, p = 0.21: no evidence of a lift (the peeked answer claimed 2.33 pp, p = 0.008)
```

**Feedback to give a model.** "Your issue is not the main problem. The test was checked after every wave and stopped at the first p < 0.05; that inflates the false positive rate, so p = 0.008 is not valid. On the full planned sample p = 0.21."
