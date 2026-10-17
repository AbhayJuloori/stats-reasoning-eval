# Review grading rubric

Each model reviews one worked answer and replies with `VERDICT`, `ISSUE` and `FIX`.
A grader (hand or LLM judge) sees the worked answer's answer key and the model's review,
then gives one score.

## Worked answers with a planted flaw

| Score | Label | Rule |
|---|---|---|
| 2 | caught | Verdict is FLAWED **and** the ISSUE names the planted problem or describes the same mechanism in other words. Extra concerns are fine as long as the planted one is stated clearly. |
| 1 | partial | Verdict is FLAWED but the ISSUE misses the planted problem: a different concern, a generic caution ("check assumptions", "sample may be small", "correlation is not causation" with no confounder named), or the right area with the wrong mechanism. |
| 0 | missed | Verdict is SOUND, or the reply is unusable. |

What counts as naming the mechanism, per flaw type:

- **peeking**: repeated interim tests with stopping at the first p < 0.05 (optional stopping) inflates false positives. "Sample is small" alone is a 1.
- **leakage**: a feature (or encoding / split) carries information about the target that will not exist at prediction time. Must point at the leaking feature or step. "Overfitting" alone is a 1.
- **simpsons**: a confounder (severity, department, stone size, age, victim race, at-bat mix, hospital) reverses or erases the pooled comparison; needs stratifying or adjusting. Must name the confounder or the stratification problem.
- **wrong_test**: the test does not match the data (independent vs paired, tiny counts, nominal codes, percent instead of counts, one-sample against an estimated benchmark). Must name why the test is wrong. A different wrong-test complaint that does not apply is a 1.
- **multiple_comparisons**: many subgroup tests with no multiplicity correction. Must mention the number of tests / correction (Bonferroni, Holm, FDR) or the forking paths.

## Clean controls (no planted flaw)

| Score | Label | Rule |
|---|---|---|
| 2 | correct | Verdict is SOUND. |
| 1 | minor | Verdict is FLAWED but the issue is a minor caveat that does not overturn the conclusion. |
| 0 | false alarm | Verdict is FLAWED and the issue claims a serious error that is not there. |

## Metrics

- **Catch rate** = share of flawed answers scored 2.
- **Flag rate** = share of flawed answers with verdict FLAWED (score 1 or 2).
- **False alarm rate** = share of clean controls with verdict FLAWED.
- Grader agreement is reported as Cohen's kappa next to raw percent agreement; raw agreement alone looks high when most items fall in one class.

## Hand-reviewed grades

80 reviews, sampled with a fixed seed and stratified by flaw type and verdict
(`evalset/hand_grade.py`). Each review got a draft score with a one line reason from a second
grader that never saw the judge's scores (`results/draft_grades.jsonl`). The draft was then
confirmed or changed by hand, row by row (`results/hand_grades.jsonl`, field `changed`).
Judge agreement is reported against these grades with Cohen's kappa.
