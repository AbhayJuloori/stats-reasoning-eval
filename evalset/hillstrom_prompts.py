"""Prompts on the Hillstrom MineThatData email experiment, each with a reference solution.

64,000 customers were randomly split into three arms: Mens E-Mail, Womens E-Mail and
No E-Mail (control). Outcomes over the next two weeks: visit, conversion, spend.

Each template turns a (segment, outcome, subgroup) choice into a question plus a short
pandas / statsmodels solution. The reference answer is whatever that code prints.
"""
import itertools
import random

DESCRIPTION = (
    "hillstrom.csv is the MineThatData email experiment: 64,000 customers who purchased in the last "
    "12 months were randomly assigned to one of three arms in column `segment`: 'Mens E-Mail', "
    "'Womens E-Mail' or 'No E-Mail' (control). Pre-treatment columns: recency (months since last "
    "purchase), history_segment, history (dollars spent last year), mens and womens (1 if they bought "
    "that line last year), zip_code ('Urban', 'Surburban' [sic], 'Rural'), newbie (1 if new in the "
    "last 12 months), channel ('Phone', 'Web', 'Multichannel'). Outcomes in the two weeks after the "
    "email: visit (0/1), conversion (0/1), spend (dollars)."
)

SUBGROUPS = {
    "all": (None, "all customers"),
    "newbie": ("newbie == 1", "new customers (newbie == 1)"),
    "returning": ("newbie == 0", "returning customers (newbie == 0)"),
    "web": ("channel == 'Web'", "customers whose channel is 'Web'"),
    "phone": ("channel == 'Phone'", "customers whose channel is 'Phone'"),
    "multichannel": ("channel == 'Multichannel'", "customers whose channel is 'Multichannel'"),
    "urban": ("zip_code == 'Urban'", "urban customers (zip_code == 'Urban')"),
    "suburban": ("zip_code == 'Surburban'", "suburban customers (zip_code == 'Surburban')"),
    "rural": ("zip_code == 'Rural'", "rural customers (zip_code == 'Rural')"),
    "mens_buyers": ("mens == 1", "customers who bought men's merchandise last year (mens == 1)"),
    "womens_buyers": ("womens == 1", "customers who bought women's merchandise last year (womens == 1)"),
}
SEGMENTS = {"mens": "Mens E-Mail", "womens": "Womens E-Mail"}
RATE_OUTCOMES = ["visit", "conversion"]


def _load(sub):
    query, _ = SUBGROUPS[sub]
    code = "import pandas as pd\nimport numpy as np\ndf = pd.read_csv('hillstrom.csv')\n"
    if query:
        code += f'df = df.query("{query}")\n'
    return code


def _two_arms(seg):
    return (f"df = df[df['segment'].isin(['{SEGMENTS[seg]}', 'No E-Mail'])].copy()\n"
            f"df['treat'] = (df['segment'] == '{SEGMENTS[seg]}').astype(int)\n")


COVARIATE_TERMS = {"recency": "recency", "history": "history", "mens": "mens", "womens": "womens",
                   "newbie": "newbie", "zip_code": "C(zip_code)", "channel": "C(channel)"}


def _covariates(sub):
    """Covariates for a regression, minus the one the subgroup holds constant."""
    query = SUBGROUPS[sub][0] or ""
    terms = [t for name, t in COVARIATE_TERMS.items() if not query.startswith(name + " ")]
    names = [name for name in COVARIATE_TERMS if not query.startswith(name + " ")]
    return " + ".join(terms), ", ".join(names)


# --- templates -------------------------------------------------------------------------
# Each returns (question, code, tolerance, skill). Questions say exactly what to compute
# and how to round, so a correct method lands inside the tolerance.

def t_lift_pp(seg, outcome, sub):
    q = (f"Among {SUBGROUPS[sub][1]}, what is the difference in {outcome} rate between the "
         f"{SEGMENTS[seg]} arm and the No E-Mail arm, in percentage points (treatment minus control)? "
         f"Round to 2 decimals.")
    code = _load(sub) + (
        f"rates = df.groupby('segment')['{outcome}'].mean()\n"
        f"print(round(100 * (rates['{SEGMENTS[seg]}'] - rates['No E-Mail']), 2))\n")
    return q, code, {"kind": "rel", "rel": 0.03, "decimals": 2}, "difference in proportions"


def t_ztest_p(seg, outcome, sub):
    q = (f"Among {SUBGROUPS[sub][1]}, run a two-sided two-proportion z-test (pooled variance, no "
         f"continuity correction) comparing the {outcome} rate of the {SEGMENTS[seg]} arm with the "
         f"No E-Mail arm. Report the p-value rounded to 4 decimals.")
    code = _load(sub) + (
        "from statsmodels.stats.proportion import proportions_ztest\n"
        f"g = df[df['segment'].isin(['{SEGMENTS[seg]}', 'No E-Mail'])].groupby('segment')['{outcome}']\n"
        "stat, p = proportions_ztest(g.sum().values, g.count().values)\n"
        "print(round(p, 4))\n")
    return q, code, {"kind": "pvalue"}, "two-proportion z-test"


def t_ci_low(seg, outcome, sub):
    q = (f"Among {SUBGROUPS[sub][1]}, compute a 95% Wald confidence interval (unpooled standard error) "
         f"for the difference in {outcome} rate, {SEGMENTS[seg]} minus No E-Mail. Report the lower "
         f"bound in percentage points, rounded to 2 decimals.")
    code = _load(sub) + (
        "from scipy import stats\n"
        f"t = df.loc[df['segment'] == '{SEGMENTS[seg]}', '{outcome}']\n"
        f"c = df.loc[df['segment'] == 'No E-Mail', '{outcome}']\n"
        "p1, p0 = t.mean(), c.mean()\n"
        "se = np.sqrt(p1 * (1 - p1) / len(t) + p0 * (1 - p0) / len(c))\n"
        "print(round(100 * (p1 - p0 - stats.norm.ppf(0.975) * se), 2))\n")
    return q, code, {"kind": "rel", "rel": 0.03, "decimals": 2}, "confidence interval"


def t_welch_t(seg, outcome, sub):
    q = (f"Among {SUBGROUPS[sub][1]}, run Welch's two-sample t-test (unequal variances) on spend, "
         f"{SEGMENTS[seg]} versus No E-Mail, using every customer in the two arms (zeros included). "
         f"Report the t statistic rounded to 3 decimals.")
    code = _load(sub) + (
        "from scipy import stats\n"
        f"t = df.loc[df['segment'] == '{SEGMENTS[seg]}', 'spend']\n"
        "c = df.loc[df['segment'] == 'No E-Mail', 'spend']\n"
        "print(round(stats.ttest_ind(t, c, equal_var=False).statistic, 3))\n")
    return q, code, {"kind": "rel", "rel": 0.03, "decimals": 3}, "Welch t-test"


def t_rel_lift(seg, outcome, sub):
    q = (f"Among {SUBGROUPS[sub][1]}, what is the relative lift in {outcome} rate of the "
         f"{SEGMENTS[seg]} arm over the No E-Mail arm, as a percent of the control rate? "
         f"Round to 1 decimal.")
    code = _load(sub) + (
        f"rates = df.groupby('segment')['{outcome}'].mean()\n"
        f"print(round(100 * (rates['{SEGMENTS[seg]}'] / rates['No E-Mail'] - 1), 1))\n")
    return q, code, {"kind": "rel", "rel": 0.03, "decimals": 1}, "relative lift"


def t_logit_or(seg, outcome, sub):
    terms, names = _covariates(sub)
    q = (f"Among {SUBGROUPS[sub][1]} in the {SEGMENTS[seg]} and No E-Mail arms only, fit a logistic "
         f"regression of {outcome} on a treatment indicator (1 = {SEGMENTS[seg]}) plus {names} "
         f"(zip_code and channel as categorical). Report the odds ratio for treatment, rounded to 3 decimals.")
    code = _load(sub) + _two_arms(seg) + (
        "import statsmodels.formula.api as smf\n"
        f"m = smf.logit('{outcome} ~ treat + {terms}', data=df).fit(disp=0)\n"
        "print(round(np.exp(m.params['treat']), 3))\n")
    return q, code, {"kind": "rel", "rel": 0.03, "decimals": 3}, "logistic regression"


def t_ols_spend(seg, outcome, sub):
    terms, names = _covariates(sub)
    q = (f"Among {SUBGROUPS[sub][1]} in the {SEGMENTS[seg]} and No E-Mail arms only, regress spend "
         f"on a treatment indicator (1 = {SEGMENTS[seg]}) plus {names} (zip_code and channel as "
         f"categorical) with OLS. Report the treatment coefficient in dollars, rounded to 4 decimals.")
    code = _load(sub) + _two_arms(seg) + (
        "import statsmodels.formula.api as smf\n"
        f"m = smf.ols('spend ~ treat + {terms}', data=df).fit()\n"
        "print(round(m.params['treat'], 4))\n")
    return q, code, {"kind": "rel", "rel": 0.03, "decimals": 4}, "regression adjustment"


def t_chi2(seg, outcome, sub):
    q = (f"Among {SUBGROUPS[sub][1]}, run a chi-square test of independence between segment (all "
         f"three arms) and {outcome}. Report the chi-square statistic rounded to 2 decimals.")
    code = _load(sub) + (
        "from scipy.stats import chi2_contingency\n"
        f"table = pd.crosstab(df['segment'], df['{outcome}'])\n"
        "print(round(chi2_contingency(table)[0], 2))\n")
    return q, code, {"kind": "rel", "rel": 0.03, "decimals": 2}, "chi-square test"


def t_srm(seg, outcome, sub):
    q = (f"Sample ratio check: among {SUBGROUPS[sub][1]}, test whether customers are split equally "
         f"across the three arms using a chi-square goodness of fit test against equal expected "
         f"counts. Report the p-value rounded to 4 decimals.")
    code = _load(sub) + (
        "from scipy.stats import chisquare\n"
        "print(round(chisquare(df['segment'].value_counts().values).pvalue, 4))\n")
    return q, code, {"kind": "pvalue"}, "sample ratio mismatch"


def t_smd(seg, outcome, sub):
    q = (f"Balance check: among {SUBGROUPS[sub][1]}, compute the standardized mean difference of "
         f"history between the {SEGMENTS[seg]} arm and the No E-Mail arm, defined as (mean_t - mean_c) "
         f"/ sqrt((var_t + var_c) / 2) with sample variances. Round to 4 decimals.")
    code = _load(sub) + (
        f"t = df.loc[df['segment'] == '{SEGMENTS[seg]}', 'history']\n"
        "c = df.loc[df['segment'] == 'No E-Mail', 'history']\n"
        "print(round((t.mean() - c.mean()) / np.sqrt((t.var() + c.var()) / 2), 4))\n")
    return q, code, {"kind": "rel", "rel": 0.05, "decimals": 4}, "covariate balance"


def t_power_n(seg, outcome, sub, lift_pp=1.0):
    q = (f"You are planning a new test for {SUBGROUPS[sub][1]}. Use the No E-Mail arm's {outcome} rate "
         f"in this data as the baseline. How many customers per arm are needed to detect an absolute "
         f"increase of {lift_pp} percentage points with a two-sided test at alpha = 0.05 and 80% power? "
         f"Use Cohen's h as the effect size with statsmodels NormalIndPower and equal arms. Round up to "
         f"a whole number.")
    code = _load(sub) + (
        "import math\n"
        "from statsmodels.stats.power import NormalIndPower\n"
        "from statsmodels.stats.proportion import proportion_effectsize\n"
        f"p0 = df.loc[df['segment'] == 'No E-Mail', '{outcome}'].mean()\n"
        f"h = proportion_effectsize(p0 + {lift_pp / 100}, p0)\n"
        "n = NormalIndPower().solve_power(effect_size=h, alpha=0.05, power=0.8, ratio=1.0)\n"
        "print(math.ceil(n))\n")
    return q, code, {"kind": "rel", "rel": 0.02, "decimals": 0}, "A/B test sample size"


def t_mde(seg, outcome, sub):
    q = (f"Given the actual arm sizes of {SEGMENTS[seg]} and No E-Mail among {SUBGROUPS[sub][1]}, what "
         f"is the minimum detectable absolute lift in {outcome} rate at alpha = 0.05 (two-sided) and 80% "
         f"power? Use the control {outcome} rate as baseline, Cohen's h with statsmodels NormalIndPower "
         f"(nobs1 = control size, ratio = treatment size / control size), and convert h back to a rate. "
         f"Report in percentage points rounded to 3 decimals.")
    code = _load(sub) + (
        "from statsmodels.stats.power import NormalIndPower\n"
        "c = df.loc[df['segment'] == 'No E-Mail', '" + outcome + "']\n"
        f"n_t = (df['segment'] == '{SEGMENTS[seg]}').sum()\n"
        "h = NormalIndPower().solve_power(effect_size=None, nobs1=len(c), alpha=0.05, power=0.8, ratio=n_t / len(c))\n"
        "p0 = c.mean()\n"
        "p1 = np.sin(np.arcsin(np.sqrt(p0)) + h / 2) ** 2\n"
        "print(round(100 * (p1 - p0), 3))\n")
    return q, code, {"kind": "rel", "rel": 0.03, "decimals": 3}, "minimum detectable effect"


def t_holm(seg, outcome, sub):
    q = (f"Split customers into the 9 cells of zip_code x channel. In each cell, run a two-sided "
         f"two-proportion z-test (pooled, no continuity correction) of {outcome} rate, {SEGMENTS[seg]} "
         f"vs No E-Mail. After a Holm correction across the 9 tests at alpha = 0.05, in how many cells "
         f"is the difference significant? Answer with a whole number.")
    code = _load("all") + (
        "from statsmodels.stats.proportion import proportions_ztest\n"
        "from statsmodels.stats.multitest import multipletests\n"
        "pvals = []\n"
        "for _, cell in df.groupby(['zip_code', 'channel']):\n"
        f"    g = cell[cell['segment'].isin(['{SEGMENTS[seg]}', 'No E-Mail'])].groupby('segment')['{outcome}']\n"
        "    pvals.append(proportions_ztest(g.sum().values, g.count().values)[1])\n"
        "print(int(multipletests(pvals, alpha=0.05, method='holm')[0].sum()))\n")
    return q, code, {"kind": "exact"}, "multiple comparisons"


def t_interaction(seg, outcome, sub):
    q = (f"Using the {SEGMENTS[seg]} and No E-Mail arms only, fit a logistic regression of {outcome} on "
         f"treat (1 = {SEGMENTS[seg]}), newbie and their interaction treat:newbie (no other terms). "
         f"Report the interaction coefficient on the log-odds scale, rounded to 4 decimals.")
    code = _load("all") + _two_arms(seg) + (
        "import statsmodels.formula.api as smf\n"
        f"m = smf.logit('{outcome} ~ treat * newbie', data=df).fit(disp=0)\n"
        "print(round(m.params['treat:newbie'], 4))\n")
    return q, code, {"kind": "rel", "rel": 0.03, "decimals": 4}, "heterogeneous effects"


def t_cuped(seg, outcome, sub):
    q = (f"Among {SUBGROUPS[sub][1]} in the {SEGMENTS[seg]} and No E-Mail arms, apply CUPED to {outcome} "
         f"using history as the pre-period covariate: theta = cov({outcome}, history) / var(history) on "
         f"the pooled two arms, adjusted = {outcome} - theta * (history - mean(history)). By what percent "
         f"does the variance of the adjusted metric fall relative to the raw metric (pooled)? Round to "
         f"3 decimals.")
    code = _load(sub) + _two_arms(seg) + (
        f"y, x = df['{outcome}'], df['history']\n"
        "theta = np.cov(y, x)[0, 1] / x.var()\n"
        "adj = y - theta * (x - x.mean())\n"
        "print(round(100 * (1 - adj.var() / y.var()), 3))\n")
    return q, code, {"kind": "rel", "rel": 0.05, "decimals": 3}, "variance reduction (CUPED)"


def t_spend_diff(seg, outcome, sub):
    q = (f"Among {SUBGROUPS[sub][1]}, what is the incremental spend per customer (mean spend of "
         f"{SEGMENTS[seg]} minus mean spend of No E-Mail), counting non-buyers as zero? Report in "
         f"dollars rounded to 4 decimals.")
    code = _load(sub) + (
        "m = df.groupby('segment')['spend'].mean()\n"
        f"print(round(m['{SEGMENTS[seg]}'] - m['No E-Mail'], 4))\n")
    return q, code, {"kind": "rel", "rel": 0.03, "decimals": 4}, "revenue per customer"


def t_head_to_head(seg, outcome, sub):
    q = (f"Among {SUBGROUPS[sub][1]}, compare the two email versions directly: run a two-sided "
         f"two-proportion z-test (pooled, no continuity correction) of {outcome} rate, Mens E-Mail vs "
         f"Womens E-Mail. Report the p-value rounded to 4 decimals.")
    code = _load(sub) + (
        "from statsmodels.stats.proportion import proportions_ztest\n"
        f"g = df[df['segment'] != 'No E-Mail'].groupby('segment')['{outcome}']\n"
        "print(round(proportions_ztest(g.sum().values, g.count().values)[1], 4))\n")
    return q, code, {"kind": "pvalue"}, "two-proportion z-test"


# (template, how many prompts, which outcomes it accepts, whether the subgroup varies)
PLAN = [
    (t_lift_pp, 11, RATE_OUTCOMES, True),
    (t_ztest_p, 10, RATE_OUTCOMES, True),
    (t_ci_low, 8, RATE_OUTCOMES, True),
    (t_welch_t, 7, ["spend"], True),
    (t_rel_lift, 6, RATE_OUTCOMES, True),
    (t_logit_or, 5, RATE_OUTCOMES, True),
    (t_ols_spend, 5, ["spend"], True),
    (t_chi2, 5, RATE_OUTCOMES, True),
    (t_srm, 4, ["none"], True),
    (t_smd, 5, ["history"], True),
    (t_power_n, 7, RATE_OUTCOMES, True),
    (t_mde, 5, RATE_OUTCOMES, True),
    (t_holm, 4, RATE_OUTCOMES, False),
    (t_interaction, 4, RATE_OUTCOMES, False),
    (t_cuped, 4, ["visit"], True),
    (t_spend_diff, 5, ["spend"], True),
    (t_head_to_head, 4, RATE_OUTCOMES, True),
]


def build(seed=7):
    rng = random.Random(seed)
    items = []
    for template, count, outcomes, vary_sub in PLAN:
        subs = list(SUBGROUPS) if vary_sub else ["all"]
        combos = list(itertools.product(SEGMENTS, outcomes, subs))
        rng.shuffle(combos)
        seen_questions = set()
        for seg, outcome, sub in combos:
            if len([i for i in items if i["template"] == template.__name__]) == count:
                break
            kwargs = {}
            if template is t_power_n:
                kwargs["lift_pp"] = rng.choice([0.5, 1.0, 2.0]) if outcome == "visit" else rng.choice([0.2, 0.3, 0.5])
            q, code, tol, skill = template(seg, outcome, sub, **kwargs)
            if q in seen_questions:  # some templates ignore seg or sub, which makes repeats
                continue
            seen_questions.add(q)
            n = len([i for i in items if i["template"] == template.__name__]) + 1
            items.append({
                "id": f"hs_{template.__name__[2:]}_{n:02d}",
                "source": "hillstrom",
                "files": ["hillstrom.csv"],
                "description": DESCRIPTION,
                "question": q,
                "reference_code": code,
                "tolerance": tol,
                "skill": skill,
                "template": template.__name__,
            })
    return items
