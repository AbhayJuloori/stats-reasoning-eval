"""Worked answers with one planted methodology flaw, each paired with a corrected twin.

Every case is real code on real data. The builder runs both versions, captures their
output, and fills the conclusion from the printed numbers, so the flawed answer reads
like an analyst's honest write up of a broken method.

A case is a dict:
    flaw         peeking | leakage | simpsons | wrong_test | multiple_comparisons
    task         the question the analyst was asked
    data         one line on the data
    flawed_code / flawed_conclusion
    clean_code  / clean_conclusion
    key          what a reviewer should catch, used by the rubric and the judge
Conclusions are format strings over the `name: value` lines the code prints.
"""
import itertools

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportions_ztest

from evalset.sandbox import SHEETS

HILL = ("hillstrom.csv: randomized email test, 64,000 customers; segment is 'Mens E-Mail', 'Womens E-Mail' "
        "or 'No E-Mail'; outcomes visit, conversion (0/1) and spend in the two weeks after the email; "
        "pre-treatment columns recency, history, history_segment, mens, womens, zip_code, newbie, channel.")

HEAD = "import pandas as pd\nimport numpy as np\n"


# ---------------------------------------------------------------------------------------
# Peeking: interim looks with a stop-at-first-significance rule
# ---------------------------------------------------------------------------------------

PEEK_CELLS = [
    ("channel == 'Phone'", "Phone customers"),
    ("mens == 1", "customers who bought men's merchandise last year"),
    ("channel == 'Web' and zip_code == 'Urban'", "urban Web customers"),
    ("history_segment == '3) $200 - $350'", "customers in the $200 to $350 history segment"),
    ("history_segment == '6) $750 - $1,000'", "customers in the $750 to $1,000 history segment"),
    ("channel == 'Phone' and zip_code == 'Urban'", "urban Phone customers"),
    ("zip_code == 'Rural'", "rural customers"),
    ("newbie == 1 and channel == 'Web'", "new Web customers"),
    ("history_segment == '1) $0 - $100'", "customers in the $0 to $100 history segment"),
    ("channel == 'Multichannel'", "Multichannel customers"),
    ("zip_code == 'Surburban' and channel == 'Web'", "suburban Web customers"),
    ("newbie == 0 and channel == 'Phone'", "returning Phone customers"),
    ("history_segment == '4) $350 - $500'", "customers in the $350 to $500 history segment"),
    ("womens == 1 and newbie == 1", "new customers who bought women's merchandise"),
    ("zip_code == 'Urban'", "urban customers"),
    ("newbie == 1 and channel == 'Phone'", "new Phone customers"),
    ("history_segment == '5) $500 - $750'", "customers in the $500 to $750 history segment"),
    ("history_segment == '2) $100 - $200'", "customers in the $100 to $200 history segment"),
    ("recency <= 3", "customers who bought in the last 3 months"),
    ("recency >= 10", "customers whose last purchase was 10 or more months ago"),
    ("zip_code == 'Rural' and newbie == 1", "new rural customers"),
    ("channel == 'Multichannel' and newbie == 0", "returning Multichannel customers"),
    ("history_segment == '7) $1,000 +'", "customers who spent over $1,000 last year"),
    ("mens == 1 and womens == 1", "customers who bought both men's and women's merchandise"),
]


def _test(metric, a, b):
    if metric == "spend":
        return stats.ttest_ind(a, b, equal_var=False).pvalue
    if a.sum() + b.sum() == 0:
        return 1.0
    return proportions_ztest([a.sum(), b.sum()], [len(a), len(b)])[1]


def _find_peek(df, seg, metric, query):
    """Return (seed, wave) where sequential looks cross p < 0.05 although the full test does not."""
    d = df.query(query)
    t_all = d.loc[d["segment"] == seg, metric]
    c_all = d.loc[d["segment"] == "No E-Mail", metric]
    if _test(metric, t_all.to_numpy(), c_all.to_numpy()) < 0.05:
        return None
    for seed in [None] + list(range(1, 40)):
        dd = d if seed is None else d.sample(frac=1, random_state=seed)
        t = dd.loc[dd["segment"] == seg, metric].to_numpy()
        c = dd.loc[dd["segment"] == "No E-Mail", metric].to_numpy()
        n = min(len(t), len(c))
        wave = max(100, (n // 10) // 100 * 100)
        for m in range(wave, n + 1, wave):
            if _test(metric, t[:m], c[:m]) < 0.05:
                # the analysis stops at the first significant look; keep it only if the email looks better
                if t[:m].mean() > c[:m].mean():
                    return seed, wave
                break
    return None


def peeking_cases(df, want=12):
    cases = []
    combos = itertools.product(PEEK_CELLS, ["conversion", "spend"], ["Womens E-Mail", "Mens E-Mail"])
    used_cells = set()
    for (query, label), metric, seg in combos:
        if len(cases) == want:
            break
        if query in used_cells:
            continue
        found = _find_peek(df, seg, metric, query)
        if not found:
            continue
        seed, wave = found
        used_cells.add(query)
        order = "" if seed is None else f"df = df.sample(frac=1, random_state={seed})  # order in which customers were mailed\n"
        if metric == "spend":
            imp = "from scipy import stats\n"
            test_line = "p = stats.ttest_ind(t[:n], c[:n], equal_var=False).pvalue"
            full_line = "p = stats.ttest_ind(t, c, equal_var=False).pvalue"
            effect = "diff = t[:n].mean() - c[:n].mean()"
            effect_full = "diff = t.mean() - c.mean()"
            unit, test_name = "dollars per customer", "Welch t-test"
        else:
            imp = "from statsmodels.stats.proportion import proportions_ztest\n"
            test_line = "p = proportions_ztest([t[:n].sum(), c[:n].sum()], [n, n])[1]"
            full_line = "p = proportions_ztest([t.sum(), c.sum()], [len(t), len(c)])[1]"
            effect = "diff = 100 * (t[:n].mean() - c[:n].mean())"
            effect_full = "diff = 100 * (t.mean() - c.mean())"
            unit, test_name = "percentage points", "two-proportion z-test"
        load = (HEAD + imp + f'df = pd.read_csv("hillstrom.csv").query("{query}")\n' + order +
                f"t = df.loc[df['segment'] == '{seg}', '{metric}'].to_numpy()\n"
                f"c = df.loc[df['segment'] == 'No E-Mail', '{metric}'].to_numpy()\n")
        flawed = load + (
            f"\n# mailed in waves of {wave} per arm; checked the {test_name} after each wave\n"
            f"for wave, n in enumerate(range({wave}, min(len(t), len(c)) + 1, {wave}), start=1):\n"
            f"    {test_line}\n"
            f"    {effect}\n"
            f"    print(f'wave {{wave}}: n per arm = {{n}}, diff = {{diff:.3f}}, p = {{p:.4f}}')\n"
            f"    if p < 0.05:\n"
            f"        print(f'stopped_wave: {{wave}}')\n"
            f"        print(f'diff: {{diff:.3f}}')\n"
            f"        print(f'p: {{p:.4f}}')\n"
            f"        break\n")
        clean = load + (
            f"\n# sample size fixed before launch; one {test_name} on the full sample at the end\n"
            f"{full_line}\n"
            f"{effect_full}\n"
            f"print(f'n_treat: {{len(t)}}')\n"
            f"print(f'n_control: {{len(c)}}')\n"
            f"print(f'diff: {{diff:.3f}}')\n"
            f"print(f'p: {{p:.4f}}')\n")
        metric_txt = "conversion rate" if metric == "conversion" else "spend per customer"
        cases.append(dict(
            flaw="peeking",
            task=f"Did the {seg} raise {metric_txt} among {label}?",
            data=HILL,
            flawed_code=flawed,
            flawed_conclusion=(f"The {seg} raised {metric_txt} among {label} by {{diff}} {unit} "
                               f"(p = {{p}}). The result was significant at wave {{stopped_wave}}, so we ended "
                               f"the test early and recommend sending the {seg} to all {label}."),
            clean_code=clean,
            clean_conclusion=(f"With the full pre-planned sample the difference is {{diff}} {unit} (p = {{p}}). "
                              f"There is no significant evidence that the {seg} changes {metric_txt} among "
                              f"{label}."),
            key=(f"Optional stopping / peeking: the {test_name} was re-run after every wave and the test stopped "
                 f"the first time p < 0.05. Repeated looks inflate the false positive rate far above 5%, so the "
                 f"reported p-value is not valid. Fix: fix the sample size in advance and test once, or use a "
                 f"sequential design with alpha spending (e.g. O'Brien-Fleming). On the full sample the effect "
                 f"is not significant."),
        ))
    return cases


# ---------------------------------------------------------------------------------------
# Multiple comparisons: many subgroup tests, significant ones reported without correction
# ---------------------------------------------------------------------------------------

PARTITIONS = [
    (["history_segment"], "history segment"),
    (["zip_code", "channel"], "zip code x channel cell"),
    (["channel", "newbie"], "channel x newbie cell"),
    (["recency"], "recency month"),
    (["zip_code", "newbie", "mens"], "zip code x newbie x mens cell"),
    (["history_segment", "newbie"], "history segment x newbie cell"),
    (["channel", "mens", "womens"], "channel x purchase history cell"),
]


def _cell_pvals(df, seg, metric, cols):
    rows = []
    for key, cell in df.groupby(cols):
        t = cell.loc[cell["segment"] == seg, metric].to_numpy()
        c = cell.loc[cell["segment"] == "No E-Mail", metric].to_numpy()
        if len(t) < 50 or len(c) < 50:
            continue
        rows.append((key, _test(metric, t, c)))
    return rows


def multiple_comparison_cases(df, want=12):
    cases = []
    for (cols, label), metric, seg in itertools.product(PARTITIONS, ["conversion", "spend"],
                                                        ["Mens E-Mail", "Womens E-Mail"]):
        if len(cases) == want:
            break
        rows = _cell_pvals(df, seg, metric, cols)
        p = np.array([r[1] for r in rows])
        raw = (p < 0.05).sum()
        holm = multipletests(p, method="holm")[0].sum()
        if raw == 0 or holm == raw or raw == len(p) or len(p) < 5:
            continue
        test_name = "Welch t-test" if metric == "spend" else "two-proportion z-test"
        if metric == "spend":
            imp = "from scipy import stats\n"
            test = "stats.ttest_ind(t, c, equal_var=False).pvalue"
        else:
            imp = "from statsmodels.stats.proportion import proportions_ztest\n"
            test = "proportions_ztest([t.sum(), c.sum()], [len(t), len(c)])[1]"
        loop = (HEAD + imp + "df = pd.read_csv('hillstrom.csv')\n\nresults = []\n"
                f"for cell, g in df.groupby({cols!r}):\n"
                f"    t = g.loc[g['segment'] == '{seg}', '{metric}'].to_numpy()\n"
                f"    c = g.loc[g['segment'] == 'No E-Mail', '{metric}'].to_numpy()\n"
                "    if len(t) < 50 or len(c) < 50:\n"
                "        continue\n"
                f"    results.append((cell, {test}))\n")
        flawed = loop + (
            "\nprint(f'cells_tested: {len(results)}')\n"
            "winners = [(cell, round(p, 4)) for cell, p in results if p < 0.05]\n"
            "print(f'significant_cells: {len(winners)}')\n"
            "for cell, p in winners:\n"
            "    print(f'  {cell}: p = {p}')\n")
        clean = loop + (
            "\nfrom statsmodels.stats.multitest import multipletests\n"
            "reject, p_holm, _, _ = multipletests([p for _, p in results], alpha=0.05, method='holm')\n"
            "print(f'cells_tested: {len(results)}')\n"
            "print(f'significant_raw: {sum(p < 0.05 for _, p in results)}')\n"
            "print(f'significant_holm: {int(reject.sum())}')\n"
            "print(f'smallest_holm_p: {p_holm.min():.4f}')\n")
        metric_txt = "conversion" if metric == "conversion" else "spend"
        cases.append(dict(
            flaw="multiple_comparisons",
            task=f"In which customer groups does the {seg} change {metric_txt}? Split by {label}.",
            data=HILL,
            flawed_code=flawed,
            flawed_conclusion=(f"We ran a {test_name} in each of the {{cells_tested}} cells. {{significant_cells}} "
                               f"cell(s) show a significant effect of the {seg} on {metric_txt} at p < 0.05 "
                               f"(listed above). We recommend targeting the {seg} at those cells."),
            clean_code=clean,
            clean_conclusion=(f"Across {{cells_tested}} cell-level tests, {{significant_raw}} had raw p < 0.05, but "
                              f"after a Holm correction only {{significant_holm}} remain significant (smallest adjusted "
                              f"p = {{smallest_holm_p}}). Only the cells that survive the correction should be reported "
                              f"as effects; the other raw hits are consistent with chance."),
            key=(f"Uncorrected multiple comparisons: {len(p)} subgroup tests were run and only the ones with p < 0.05 "
                 f"were reported. With that many tests a few false positives are expected by chance. Fix: correct "
                 f"for multiplicity (Holm or Bonferroni, or control FDR) or pre-register the subgroups / test an "
                 f"interaction; after Holm correction {holm} of {raw} raw hits survive."),
        ))
    return cases


# ---------------------------------------------------------------------------------------
# Target leakage: features that encode the outcome, or evaluation that sees the future
# ---------------------------------------------------------------------------------------

def leakage_cases():
    base_feats = "['recency', 'history', 'mens', 'womens', 'newbie']"
    skl = ("from sklearn.linear_model import LogisticRegression\n"
           "from sklearn.model_selection import train_test_split\n"
           "from sklearn.metrics import roc_auc_score\n")

    def hill_clf(target, extra, extra_desc, task):
        code = lambda feats: (HEAD + skl +
            "df = pd.read_csv('hillstrom.csv')\n"
            "X = pd.get_dummies(df[" + feats + " + ['zip_code', 'channel', 'segment']], drop_first=True).astype(float)\n"
            f"y = df['{target}']\n"
            "X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)\n"
            "model = LogisticRegression(max_iter=2000).fit(X_tr, y_tr)\n"
            "print(f\"test_auc: {roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]):.3f}\")\n")
        leaky = base_feats[:-1] + f", '{extra}']"
        return dict(
            flaw="leakage", task=task, data=HILL,
            flawed_code=code(leaky),
            flawed_conclusion=(f"A logistic regression on customer attributes, email arm and {extra_desc} predicts "
                               f"{target} with test AUC {{test_auc}}. The model is ready to score customers before "
                               f"the next campaign."),
            clean_code=code(base_feats),
            clean_conclusion=(f"Using only information known before the email (customer attributes and arm), the "
                              f"model reaches test AUC {{test_auc}}. That is the honest estimate for pre-campaign scoring."),
            key=(f"Target leakage: `{extra}` is measured after the email, in the same window as {target}, and is "
                 f"almost a direct function of it, so it will not be available when scoring customers before a "
                 f"campaign. The AUC is inflated. Fix: drop post-treatment / post-outcome features and use only "
                 f"what is known at prediction time."),
        )

    cases = [
        hill_clf("conversion", "visit", "whether the customer visited the site",
                 "Build a model that predicts which customers will convert so marketing can pick who to email next time."),
        hill_clf("visit", "conversion", "the conversion flag",
                 "Build a model that predicts which customers will visit the site after an email."),
    ]

    # spend leaks conversion exactly (spend > 0 iff conversion)
    code_spend = lambda feats: (HEAD + skl +
        "df = pd.read_csv('hillstrom.csv')\n"
        f"X = pd.get_dummies(df[{feats}], drop_first=True).astype(float)\n"
        "y = df['conversion']\n"
        "X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)\n"
        "model = LogisticRegression(max_iter=5000).fit(X_tr, y_tr)\n"
        "print(f\"test_auc: {roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]):.3f}\")\n")
    cases.append(dict(
        flaw="leakage", task="Predict which emailed customers convert, to size next month's campaign.", data=HILL,
        flawed_code=code_spend("['recency', 'history', 'newbie', 'channel', 'spend']"),
        flawed_conclusion="The conversion model reaches test AUC {test_auc} on held out customers, so it can rank customers almost perfectly.",
        clean_code=code_spend("['recency', 'history', 'newbie', 'channel']"),
        clean_conclusion="With pre-campaign features only, the conversion model reaches test AUC {test_auc}.",
        key="Target leakage: `spend` is positive exactly when the customer converted, so it encodes the label. AUC near 1 is an artifact. Fix: remove spend (and any outcome-window variable) from the features.",
    ))

    # conversion leaks into a spend regression
    code_reg = lambda feats: (HEAD +
        "from sklearn.linear_model import LinearRegression\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.metrics import r2_score\n"
        "df = pd.read_csv('hillstrom.csv')\n"
        f"X = pd.get_dummies(df[{feats}], drop_first=True).astype(float)\n"
        "X_tr, X_te, y_tr, y_te = train_test_split(X, df['spend'], test_size=0.3, random_state=0)\n"
        "model = LinearRegression().fit(X_tr, y_tr)\n"
        "print(f\"test_r2: {r2_score(y_te, model.predict(X_te)):.3f}\")\n")
    cases.append(dict(
        flaw="leakage", task="Forecast spend per customer for the next email campaign.", data=HILL,
        flawed_code=code_reg("['recency', 'history', 'newbie', 'channel', 'segment', 'conversion', 'visit']"),
        flawed_conclusion="The linear model explains R^2 = {test_r2} of spend on the test set, a strong forecast for campaign revenue.",
        clean_code=code_reg("['recency', 'history', 'newbie', 'channel', 'segment']"),
        clean_conclusion="With features known before the campaign the model explains R^2 = {test_r2} of spend; spend is close to unpredictable from pre-campaign data.",
        key="Target leakage: `conversion` and `visit` are outcomes from the same two-week window as spend, and spend is zero whenever conversion is zero. They cannot be known when forecasting. Fix: forecast from pre-campaign features only.",
    ))

    # target encoding of a high-cardinality key fitted on all rows before the split
    for target in ["conversion", "visit"]:
        enc_full = (HEAD + skl +
            "df = pd.read_csv('hillstrom.csv')\n"
            "df['key'] = df['zip_code'] + '_' + df['channel'] + '_' + df['history'].round(0).astype(str)\n"
            f"df['key_rate'] = df.groupby('key')['{target}'].transform('mean')  # target encoding\n"
            "X = pd.get_dummies(df[['recency', 'history', 'newbie', 'segment', 'key_rate']], drop_first=True).astype(float)\n"
            f"X_tr, X_te, y_tr, y_te = train_test_split(X, df['{target}'], test_size=0.3, random_state=0)\n"
            "model = LogisticRegression(max_iter=2000).fit(X_tr, y_tr)\n"
            "print(f\"test_auc: {roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]):.3f}\")\n"
            "print(f\"keys: {df['key'].nunique()}\")\n")
        enc_fold = (HEAD + skl +
            "df = pd.read_csv('hillstrom.csv')\n"
            "df['key'] = df['zip_code'] + '_' + df['channel'] + '_' + df['history'].round(0).astype(str)\n"
            "tr, te = train_test_split(df, test_size=0.3, random_state=0)\n"
            f"rates = tr.groupby('key')['{target}'].mean()  # encoding learned on the training rows only\n"
            f"prior = tr['{target}'].mean()\n"
            "tr = tr.assign(key_rate=tr['key'].map(rates))\n"
            "te = te.assign(key_rate=te['key'].map(rates).fillna(prior))\n"
            "cols = ['recency', 'history', 'newbie', 'segment', 'key_rate']\n"
            "X_tr = pd.get_dummies(tr[cols], drop_first=True).astype(float)\n"
            "X_te = pd.get_dummies(te[cols], drop_first=True).astype(float)\n"
            f"model = LogisticRegression(max_iter=2000).fit(X_tr, tr['{target}'])\n"
            f"print(f\"test_auc: {{roc_auc_score(te['{target}'], model.predict_proba(X_te)[:, 1]):.3f}}\")\n"
            "print(f\"keys: {df['key'].nunique()}\")\n")
        cases.append(dict(
            flaw="leakage", task=f"Build a {target} model that uses a location x channel x spend-level key as a feature.", data=HILL,
            flawed_code=enc_full,
            flawed_conclusion=f"Target encoding the {{keys}}-level key lifts test AUC for {target} to {{test_auc}}; the key is the strongest feature.",
            clean_code=enc_fold,
            clean_conclusion=f"With the target encoding learned on training rows only, test AUC for {target} is {{test_auc}}.",
            key=(f"Target leakage through target encoding: the per-key {target} rate was computed on all rows before the "
                 f"train/test split, and most keys hold one or a few customers, so each test row's own label is baked "
                 f"into its feature. Fix: fit the encoding inside the training fold (out-of-fold encoding with "
                 f"smoothing) and apply it to the test rows."),
        ))

    # loans: interest rate is set from the grade
    code_grade = lambda feats: (HEAD + skl +
        "df = pd.read_csv('loans_full_schema.csv')\n"
        "df['risky'] = df['grade'].isin(['D', 'E', 'F', 'G']).astype(int)\n"
        f"X = df[{feats}].fillna(0)\n"
        "X_tr, X_te, y_tr, y_te = train_test_split(X, df['risky'], test_size=0.3, random_state=0, stratify=df['risky'])\n"
        "model = LogisticRegression(max_iter=5000).fit(X_tr, y_tr)\n"
        "print(f\"test_auc: {roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]):.3f}\")\n")
    loans = ("loans_full_schema.csv: 10,000 Lending Club loans with applicant credit history, loan terms "
             "(loan_amount, term, interest_rate, installment, grade, sub_grade) and repayment status.")
    app_feats = "['annual_income', 'debt_to_income', 'inquiries_last_12m', 'total_credit_utilized', 'delinq_2y', 'accounts_opened_24m']"
    cases.append(dict(
        flaw="leakage", task="Build a model that flags risky applicants (grade D to G) at application time.", data=loans,
        flawed_code=code_grade(app_feats[:-1] + ", 'interest_rate', 'installment', 'loan_amount']"),
        flawed_conclusion="The application screen flags risky applicants with test AUC {test_auc}, so manual underwriting review can be dropped.",
        clean_code=code_grade(app_feats[:-1] + ", 'loan_amount']"),
        clean_conclusion="Using applicant information available before pricing, the screen reaches test AUC {test_auc}.",
        key="Target leakage: interest_rate (and the installment derived from it) is set from the assigned grade, so it is a consequence of the label and not known before grading. Fix: drop pricing fields and use only applicant data available at application.",
    ))

    code_late = lambda feats: (HEAD + skl +
        "df = pd.read_csv('loans_full_schema.csv')\n"
        "df['trouble'] = df['loan_status'].isin(['Late (31-120 days)', 'Late (16-30 days)', 'In Grace Period', 'Charged Off']).astype(int)\n"
        f"X = df[{feats}].fillna(0)\n"
        "X_tr, X_te, y_tr, y_te = train_test_split(X, df['trouble'], test_size=0.3, random_state=0, stratify=df['trouble'])\n"
        "model = LogisticRegression(max_iter=5000, class_weight='balanced').fit(X_tr, y_tr)\n"
        "print(f\"test_auc: {roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]):.3f}\")\n")
    cases.append(dict(
        flaw="leakage", task="Predict which new loans will fall behind on payments.", data=loans,
        flawed_code=code_late("['annual_income', 'debt_to_income', 'interest_rate', 'loan_amount', 'paid_late_fees', 'paid_total', 'balance']"),
        flawed_conclusion="The delinquency model reaches test AUC {test_auc}; we can use it to decline applicants likely to fall behind.",
        clean_code=code_late("['annual_income', 'debt_to_income', 'interest_rate', 'loan_amount', 'inquiries_last_12m', 'delinq_2y']"),
        clean_conclusion="With information known when the loan is issued, the delinquency model reaches test AUC {test_auc}.",
        key="Target leakage: paid_late_fees, paid_total and balance are repayment results recorded after issue; late fees exist only because the loan went late. They are not available for new applicants. Fix: use only origination-time features.",
    ))

    code_paid = lambda feats: (HEAD + skl +
        "df = pd.read_csv('loans_full_schema.csv')\n"
        "df['paid_off'] = (df['loan_status'] == 'Fully Paid').astype(int)\n"
        f"X = df[{feats}].fillna(0)\n"
        "X_tr, X_te, y_tr, y_te = train_test_split(X, df['paid_off'], test_size=0.3, random_state=0, stratify=df['paid_off'])\n"
        "model = LogisticRegression(max_iter=5000, class_weight='balanced').fit(X_tr, y_tr)\n"
        "print(f\"test_auc: {roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]):.3f}\")\n")
    cases.append(dict(
        flaw="leakage", task="Predict which loans will be paid off early, for prepayment planning.", data=loans,
        flawed_code=code_paid("['annual_income', 'loan_amount', 'term', 'interest_rate', 'balance', 'paid_principal']"),
        flawed_conclusion="The prepayment model reaches test AUC {test_auc}, accurate enough for the funding plan.",
        clean_code=code_paid("['annual_income', 'loan_amount', 'term', 'interest_rate', 'debt_to_income']"),
        clean_conclusion="With origination-time features the prepayment model reaches test AUC {test_auc}.",
        key="Target leakage: balance and paid_principal are repayment outcomes; a zero balance is what 'Fully Paid' means. Fix: drop post-origination fields.",
    ))

    # temporal leakage: random split on a state-year panel
    smoke = ("smoking2.csv: cigarette sales per capita (cigsale) by US state and year, 1970 to 2000, with "
             "retail price (retprice), log income, beer consumption and youth share.")
    smoke_head = (HEAD + "from sklearn.ensemble import RandomForestRegressor\nfrom sklearn.metrics import r2_score\n"
                  "df = pd.read_csv('smoking2.csv')\n"
                  "X = pd.get_dummies(df[['state', 'year', 'retprice']].astype({'state': str}), drop_first=True).astype(float)\n"
                  "y = df['cigsale']\n")
    cases.append(dict(
        flaw="leakage", task="Build a model to forecast next year's cigarette sales by state.", data=smoke,
        flawed_code=smoke_head + (
            "from sklearn.model_selection import train_test_split\n"
            "X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)\n"
            "model = RandomForestRegressor(n_estimators=300, random_state=0).fit(X_tr, y_tr)\n"
            "print(f\"test_r2: {r2_score(y_te, model.predict(X_te)):.3f}\")\n"),
        flawed_conclusion="On a held out 20% of state-years the forest reaches R^2 = {test_r2}, so it is reliable for next year's forecast.",
        clean_code=smoke_head + (
            "train, test = df['year'] <= 1994, df['year'] > 1994  # forecast forward in time\n"
            "model = RandomForestRegressor(n_estimators=300, random_state=0).fit(X[train], y[train])\n"
            "print(f\"test_r2: {r2_score(y[test], model.predict(X[test])):.3f}\")\n"),
        clean_conclusion="Trained on 1970 to 1994 and tested on 1995 to 2000, the forest reaches R^2 = {test_r2}; that is the honest forecasting accuracy.",
        key="Temporal leakage: a random split puts later years of the same state in training, so the model interpolates between neighbouring years instead of forecasting. Fix: split by time (train on earlier years, test on later ones) or use rolling-origin evaluation.",
    ))

    # group-level target leakage: school mean outcome computed with test students
    mind = ("learning_mindset.csv: 10,391 students in 76 schools; achievement_score (standardized), intervention, "
            "student characteristics and school-level covariates (schoolid).")
    cases.append(dict(
        flaw="leakage", task="Predict student achievement_score for students at the same schools next year.", data=mind,
        flawed_code=HEAD + (
            "from sklearn.linear_model import LinearRegression\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.metrics import r2_score\n"
            "df = pd.read_csv('learning_mindset.csv')\n"
            "df['school_avg_score'] = df.groupby('schoolid')['achievement_score'].transform('mean')\n"
            "cols = ['intervention', 'success_expect', 'frst_in_family', 'school_mindset', 'school_poverty', 'school_avg_score']\n"
            "X_tr, X_te, y_tr, y_te = train_test_split(df[cols], df['achievement_score'], test_size=0.3, random_state=0)\n"
            "model = LinearRegression().fit(X_tr, y_tr)\n"
            "print(f\"test_r2: {r2_score(y_te, model.predict(X_te)):.3f}\")\n"),
        flawed_conclusion="With the school's average score as a feature the model reaches test R^2 = {test_r2}; school context is the strongest predictor of achievement.",
        clean_code=HEAD + (
            "from sklearn.linear_model import LinearRegression\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.metrics import r2_score\n"
            "df = pd.read_csv('learning_mindset.csv')\n"
            "tr, te = train_test_split(df, test_size=0.3, random_state=0)\n"
            "avg = tr.groupby('schoolid')['achievement_score'].mean()  # school average from training students only\n"
            "tr = tr.assign(school_avg_score=tr['schoolid'].map(avg))\n"
            "te = te.assign(school_avg_score=te['schoolid'].map(avg))\n"
            "cols = ['intervention', 'success_expect', 'frst_in_family', 'school_mindset', 'school_poverty', 'school_avg_score']\n"
            "model = LinearRegression().fit(tr[cols], tr['achievement_score'])\n"
            "print(f\"test_r2: {r2_score(te['achievement_score'], model.predict(te[cols])):.3f}\")\n"),
        clean_conclusion="With the school average computed from training students only, test R^2 is {test_r2}.",
        key="Target leakage: school_avg_score is the mean of the target computed over all students, including the test students, so each test row's label feeds its own feature. Fix: compute the school average from training rows only (or out-of-fold) before the split.",
    ))

    # collections: agreement is a post-email outcome of the same process
    coll = ("collections_email.csv: 5,000 debtors; email (randomized), opened, agreement (signed a payment plan "
            "after the email), credit_limit and risk_score (pre-email), payments (amount paid).")
    code_coll = lambda feats: (HEAD +
        "from sklearn.linear_model import LinearRegression\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.metrics import r2_score\n"
        "df = pd.read_csv('collections_email.csv')\n"
        f"X = df[{feats}].fillna(0)\n"
        "X_tr, X_te, y_tr, y_te = train_test_split(X, df['payments'], test_size=0.3, random_state=0)\n"
        "model = LinearRegression().fit(X_tr, y_tr)\n"
        "print(f\"test_r2: {r2_score(y_te, model.predict(X_te)):.3f}\")\n")
    cases.append(dict(
        flaw="leakage", task="Predict how much each debtor will pay, to prioritize collections calls before sending emails.", data=coll,
        flawed_code=code_coll("['credit_limit', 'risk_score', 'email', 'opened', 'agreement']"),
        flawed_conclusion="The payment model reaches test R^2 = {test_r2}; we will use it to rank debtors for calls.",
        clean_code=code_coll("['credit_limit', 'risk_score']"),
        clean_conclusion="Using only pre-email fields, the payment model reaches test R^2 = {test_r2}.",
        key="Target leakage: opened and agreement are recorded after the email, in the same window as payments, and are unknown when ranking debtors beforehand. Fix: use only pre-email features.",
    ))
    return cases


# ---------------------------------------------------------------------------------------
# Simpson's paradox: a pooled comparison that reverses within strata of a confounder
# ---------------------------------------------------------------------------------------

KIDNEY = ("Kidney stone treatments (Charig et al., BMJ 1986): success counts for open surgery (A) and "
          "percutaneous nephrolithotomy (B), split by stone size. Patients were not randomized.")
BERKELEY = ("UC Berkeley graduate admissions, fall 1973, six largest departments (Bickel, Hammel and O'Connell, "
            "Science 1975): applicants admitted and rejected by gender and department.")
FLORIDA = ("Florida murder cases 1976 to 1987 (Radelet and Pierce 1991, as tabulated in Agresti, Categorical "
           "Data Analysis): death penalty verdicts by defendant race and victim race.")
BATTING = "Hits and at-bats for Derek Jeter and David Justice in the 1995 and 1996 MLB seasons."
WHICKHAM = ("Whickham study (Appleton, French and Vanderpump 1996): women surveyed in 1972 to 1974 by smoking "
            "status and age group, and whether they were alive 20 years later.")

KIDNEY_DF = ("rows = [('A', 'small', 81, 87), ('A', 'large', 192, 263),\n"
             "        ('B', 'small', 234, 270), ('B', 'large', 55, 80)]\n"
             "df = pd.DataFrame(rows, columns=['treatment', 'stone', 'success', 'n'])\n")
BERKELEY_DF = ("rows = [('A', 'M', 512, 313), ('A', 'F', 89, 19), ('B', 'M', 353, 207), ('B', 'F', 17, 8),\n"
               "        ('C', 'M', 120, 205), ('C', 'F', 202, 391), ('D', 'M', 138, 279), ('D', 'F', 131, 244),\n"
               "        ('E', 'M', 53, 138), ('E', 'F', 94, 299), ('F', 'M', 22, 351), ('F', 'F', 24, 317)]\n"
               "df = pd.DataFrame(rows, columns=['dept', 'gender', 'admitted', 'rejected'])\n")
FLORIDA_DF = ("rows = [('white', 'white', 53, 414), ('white', 'black', 0, 16),\n"
              "        ('black', 'white', 11, 37), ('black', 'black', 4, 139)]\n"
              "df = pd.DataFrame(rows, columns=['defendant', 'victim', 'death', 'no_death'])\n")
BATTING_DF = ("rows = [('Jeter', 1995, 12, 48), ('Jeter', 1996, 183, 582),\n"
              "        ('Justice', 1995, 104, 411), ('Justice', 1996, 45, 140)]\n"
              "df = pd.DataFrame(rows, columns=['player', 'year', 'hits', 'at_bats'])\n")
WHICKHAM_DF = ("rows = [('18-24', 'yes', 2, 53), ('18-24', 'no', 1, 61), ('25-34', 'yes', 3, 121), ('25-34', 'no', 5, 152),\n"
               "        ('35-44', 'yes', 14, 95), ('35-44', 'no', 7, 114), ('45-54', 'yes', 27, 103), ('45-54', 'no', 12, 66),\n"
               "        ('55-64', 'yes', 51, 64), ('55-64', 'no', 40, 81), ('65-74', 'yes', 29, 7), ('65-74', 'no', 101, 28),\n"
               "        ('75+', 'yes', 13, 0), ('75+', 'no', 64, 0)]\n"
               "df = pd.DataFrame(rows, columns=['age', 'smoker', 'dead', 'alive'])\n")

MED = ("medicine_impact_recovery.csv: 20,000 patients (observational); sex, age, severity (0 to 1), "
       "medication (0/1), recovery (days to recover; lower is better).")
HOSP = ("hospital_treatment.csv: 80 patients in two hospitals (observational); treatment (0/1), "
        "severity, days in hospital (lower is better).")


def simpsons_cases():
    S = "simpsons"
    return [
        dict(flaw=S, task="Does the medication speed up recovery?", data=MED,
             flawed_code=HEAD + "df = pd.read_csv('medicine_impact_recovery.csv')\nm = df.groupby('medication')['recovery'].mean()\nprint(f\"recovery_med: {m[1]:.2f}\")\nprint(f\"recovery_no_med: {m[0]:.2f}\")\nprint(f\"diff: {m[1] - m[0]:.2f}\")\n",
             flawed_conclusion="Patients on the medication took {diff} more days to recover on average ({recovery_med} vs {recovery_no_med}). The medication appears to slow recovery and should not be prescribed.",
             clean_code=HEAD + "df = pd.read_csv('medicine_impact_recovery.csv')\ndf['severity_band'] = pd.qcut(df['severity'], 4)\nby = df.groupby(['severity_band', 'medication'], observed=True)['recovery'].mean().unstack()\nby['diff'] = by[1] - by[0]\nprint(by.round(2))\nprint(f\"bands_where_med_faster: {(by['diff'] < 0).sum()} of {by['diff'].notna().sum()}\")\nprint(f\"share_of_severe_on_med: {df.loc[df['severity'] > df['severity'].median(), 'medication'].mean():.2f}\")\n",
             clean_conclusion="Within every severity band the medicated patients recover faster ({bands_where_med_faster} bands). The pooled gap is reversed because sicker patients were far more likely to get the medication ({share_of_severe_on_med} of above-median severity). Adjusting for severity, the medication helps.",
             key="Simpson's paradox / confounding by severity: sicker patients were much more likely to receive the medication and also recover more slowly. The pooled difference reverses within severity strata. Fix: stratify or adjust for severity (regression, matching, weighting) before drawing a causal conclusion."),
        dict(flaw=S, task="Estimate the effect of the medication on recovery time with a regression.", data=MED,
             flawed_code=HEAD + "import statsmodels.formula.api as smf\ndf = pd.read_csv('medicine_impact_recovery.csv')\nm = smf.ols('recovery ~ medication + age + sex', data=df).fit()\nprint(f\"coef: {m.params['medication']:.2f}\")\nprint(f\"p: {m.pvalues['medication']:.4f}\")\n",
             flawed_conclusion="Controlling for age and sex, the medication adds {coef} days to recovery (p = {p}). The effect is large and significant, so the medication is harmful.",
             clean_code=HEAD + "import statsmodels.formula.api as smf\ndf = pd.read_csv('medicine_impact_recovery.csv')\nm = smf.ols('recovery ~ medication + severity + age + sex', data=df).fit()\nprint(f\"coef: {m.params['medication']:.2f}\")\nprint(f\"p: {m.pvalues['medication']:.4f}\")\n",
             clean_conclusion="Adding severity, the medication coefficient is {coef} days (p = {p}): it shortens recovery. Severity drives both who gets the drug and how long recovery takes.",
             key="Simpson's paradox / omitted confounder: severity drives both treatment and recovery, and leaving it out flips the sign of the medication effect. Fix: include severity (the confounder) or stratify on it; age and sex alone do not remove the confounding."),
        dict(flaw=S, task="Is the medication associated with fast recovery (under 20 days)?", data=MED,
             flawed_code=HEAD + "from statsmodels.stats.proportion import proportions_ztest\ndf = pd.read_csv('medicine_impact_recovery.csv')\ndf['fast'] = (df['recovery'] < 20).astype(int)\ng = df.groupby('medication')['fast']\nprint(f\"fast_med: {g.mean()[1]:.3f}\")\nprint(f\"fast_no_med: {g.mean()[0]:.3f}\")\nprint(f\"p: {proportions_ztest(g.sum().values, g.count().values)[1]:.4f}\")\n",
             flawed_conclusion="Only {fast_med} of medicated patients recovered within 20 days versus {fast_no_med} without it (p = {p}). The medication lowers the chance of a fast recovery.",
             clean_code=HEAD + "from statsmodels.stats.contingency_tables import StratifiedTable\ndf = pd.read_csv('medicine_impact_recovery.csv')\ndf['fast'] = (df['recovery'] < 20).astype(int)\ndf['severity_band'] = pd.qcut(df['severity'], 4)\ntables = []\nfor _, g in df.groupby('severity_band', observed=True):\n    t = pd.crosstab(g['medication'], g['fast']).reindex(index=[1, 0], columns=[1, 0], fill_value=0).values\n    if (t.sum(axis=1) > 0).all():\n        tables.append(t)\nst = StratifiedTable(tables)\nprint(f\"mh_odds_ratio: {st.oddsratio_pooled:.2f}\")\nprint(f\"p: {st.test_null_odds().pvalue:.4f}\")\n",
             clean_conclusion="Stratifying by severity, the Mantel-Haenszel odds ratio of fast recovery for medicated patients is {mh_odds_ratio} (p = {p}): within comparable patients the medication helps.",
             key="Simpson's paradox: the crude comparison mixes severity levels; medicated patients are mostly severe cases. Within severity strata the association reverses. Fix: stratify (Mantel-Haenszel) or adjust for severity."),
        dict(flaw=S, task="Does the new treatment shorten hospital stays?", data=HOSP,
             flawed_code=HEAD + "df = pd.read_csv('hospital_treatment.csv')\nm = df.groupby('treatment')['days'].mean()\nprint(f\"days_treated: {m[1]:.1f}\")\nprint(f\"days_untreated: {m[0]:.1f}\")\nprint(f\"diff: {m[1] - m[0]:.1f}\")\n",
             flawed_conclusion="Treated patients stayed {diff} days longer ({days_treated} vs {days_untreated}). The treatment lengthens hospital stays.",
             clean_code=HEAD + "df = pd.read_csv('hospital_treatment.csv')\nprint(df.groupby(['hospital', 'treatment'])['days'].agg(['mean', 'size']).round(1))\nwithin = df.groupby(['hospital', 'treatment'])['days'].mean().unstack()\nprint(f\"diff_hospital_0: {within.loc[0, 1] - within.loc[0, 0]:.1f}\")\nprint(f\"diff_hospital_1: {within.loc[1, 1] - within.loc[1, 0]:.1f}\")\n",
             clean_conclusion="Within each hospital the treated patients stay shorter ({diff_hospital_0} and {diff_hospital_1} days). Hospital 1 treats almost everyone and has much longer stays, which produces the misleading pooled gap. Treated groups are small, so the estimate is noisy.",
             key="Simpson's paradox: almost all treated patients are in hospital 1, which has longer stays (sicker patients). Pooling across hospitals reverses the within-hospital effect. Fix: compare within hospital / adjust for hospital and severity."),
        dict(flaw=S, task="Estimate the treatment effect on days in hospital with a regression.", data=HOSP,
             flawed_code=HEAD + "import statsmodels.formula.api as smf\ndf = pd.read_csv('hospital_treatment.csv')\nm = smf.ols('days ~ treatment', data=df).fit()\nprint(f\"coef: {m.params['treatment']:.2f}\")\nprint(f\"p: {m.pvalues['treatment']:.4f}\")\n",
             flawed_conclusion="The regression shows treatment adds {coef} days in hospital (p = {p}), a clear harmful effect.",
             clean_code=HEAD + "import statsmodels.formula.api as smf\ndf = pd.read_csv('hospital_treatment.csv')\nm = smf.ols('days ~ treatment + severity + C(hospital)', data=df).fit()\nprint(f\"coef: {m.params['treatment']:.2f}\")\nprint(f\"p: {m.pvalues['treatment']:.4f}\")\n",
             clean_conclusion="Adjusting for severity and hospital, treatment changes stay length by {coef} days (p = {p}); the sign flips once the confounders are in.",
             key="Simpson's paradox / omitted confounders: treatment assignment depends on hospital and severity, which also drive length of stay. The unadjusted coefficient has the wrong sign. Fix: adjust for hospital and severity."),
        dict(flaw=S, task="Which kidney stone treatment has the higher success rate?", data=KIDNEY,
             flawed_code=HEAD + KIDNEY_DF + "tot = df.groupby('treatment')[['success', 'n']].sum()\nrate = tot['success'] / tot['n']\nprint(f\"rate_A: {rate['A']:.3f}\")\nprint(f\"rate_B: {rate['B']:.3f}\")\n",
             flawed_conclusion="Treatment B succeeds more often ({rate_B} vs {rate_A}), so B should be the default treatment.",
             clean_code=HEAD + KIDNEY_DF + "df['rate'] = df['success'] / df['n']\nwide = df.pivot(index='stone', columns='treatment', values='rate')\nprint(wide.round(3))\nprint(f\"small_A_minus_B: {wide.loc['small', 'A'] - wide.loc['small', 'B']:.3f}\")\nprint(f\"large_A_minus_B: {wide.loc['large', 'A'] - wide.loc['large', 'B']:.3f}\")\n",
             clean_conclusion="A has the higher success rate for both small ({small_A_minus_B}) and large stones ({large_A_minus_B}). B only looks better pooled because it was used mostly on easier small stones.",
             key="Simpson's paradox: stone size confounds the comparison. A was given mostly to large (harder) stones. Within each size A beats B, so the pooled ranking reverses. Fix: compare within stone size or adjust for it."),
        dict(flaw=S, task="Is there evidence of gender bias against women in graduate admissions?", data=BERKELEY,
             flawed_code=HEAD + "from scipy.stats import chi2_contingency\n" + BERKELEY_DF + "tot = df.groupby('gender')[['admitted', 'rejected']].sum()\nrate = tot['admitted'] / tot.sum(axis=1)\nprint(f\"admit_men: {rate['M']:.3f}\")\nprint(f\"admit_women: {rate['F']:.3f}\")\nprint(f\"p: {chi2_contingency(tot.values)[1]:.2e}\")\n",
             flawed_conclusion="Men were admitted at {admit_men} versus {admit_women} for women (chi-square p = {p}). The university admits women at a significantly lower rate, which points to bias against women.",
             clean_code=HEAD + "from statsmodels.stats.contingency_tables import StratifiedTable\n" + BERKELEY_DF + "tables = [g[['admitted', 'rejected']].loc[g['gender'].map({'F': 0, 'M': 1}).sort_values().index].values for _, g in df.groupby('dept')]\nst = StratifiedTable(tables)\nprint(f\"mh_odds_ratio_women_vs_men: {st.oddsratio_pooled:.2f}\")\nprint(f\"p: {st.test_null_odds().pvalue:.4f}\")\ndf['rate'] = df['admitted'] / (df['admitted'] + df['rejected'])\nwide = df.pivot(index='dept', columns='gender', values='rate')\nprint(f\"depts_women_higher: {(wide['F'] > wide['M']).sum()} of 6\")\n",
             clean_conclusion="Within departments women are admitted at similar or higher rates ({depts_women_higher} departments favour women; Mantel-Haenszel odds ratio women vs men {mh_odds_ratio_women_vs_men}, p = {p}). Women applied more to departments with low admission rates; the pooled gap does not show bias in admission decisions.",
             key="Simpson's paradox: department confounds the pooled comparison; women applied mostly to departments that reject most applicants. Within departments there is no disadvantage for women. Fix: stratify by department (Mantel-Haenszel) or model admission ~ gender + department."),
        dict(flaw=S, task="Model the probability of admission by gender.", data=BERKELEY,
             flawed_code=HEAD + "import statsmodels.api as sm\nimport statsmodels.formula.api as smf\n" + BERKELEY_DF + "m = smf.glm('admitted + rejected ~ gender', data=df, family=sm.families.Binomial()).fit()\nprint(f\"odds_ratio_men: {np.exp(m.params['gender[T.M]']):.2f}\")\nprint(f\"p: {m.pvalues['gender[T.M]']:.2e}\")\n",
             flawed_conclusion="Men have {odds_ratio_men} times the odds of admission (p = {p}). Gender has a strong, significant effect on admission.",
             clean_code=HEAD + "import statsmodels.api as sm\nimport statsmodels.formula.api as smf\n" + BERKELEY_DF + "m = smf.glm('admitted + rejected ~ gender + C(dept)', data=df, family=sm.families.Binomial()).fit()\nprint(f\"odds_ratio_men: {np.exp(m.params['gender[T.M]']):.2f}\")\nprint(f\"p: {m.pvalues['gender[T.M]']:.4f}\")\n",
             clean_conclusion="With department in the model, men's odds ratio is {odds_ratio_men} (p = {p}): no evidence that men are favoured once department is accounted for.",
             key="Simpson's paradox / omitted confounder: department drives both who applies where and admission rates. The gender-only model attributes the department effect to gender. Fix: include department."),
        dict(flaw=S, task="Are black defendants more likely than white defendants to receive the death penalty?", data=FLORIDA,
             flawed_code=HEAD + FLORIDA_DF + "tot = df.groupby('defendant')[['death', 'no_death']].sum()\nrate = tot['death'] / tot.sum(axis=1)\nprint(f\"death_rate_white_def: {rate['white']:.3f}\")\nprint(f\"death_rate_black_def: {rate['black']:.3f}\")\n",
             flawed_conclusion="White defendants received the death penalty more often ({death_rate_white_def}) than black defendants ({death_rate_black_def}), so there is no evidence of a racial disadvantage for black defendants.",
             clean_code=HEAD + FLORIDA_DF + "df['rate'] = df['death'] / (df['death'] + df['no_death'])\nwide = df.pivot(index='victim', columns='defendant', values='rate')\nprint(wide.round(3))\nprint(f\"white_victim_black_minus_white_def: {wide.loc['white', 'black'] - wide.loc['white', 'white']:.3f}\")\nprint(f\"black_victim_black_minus_white_def: {wide.loc['black', 'black'] - wide.loc['black', 'white']:.3f}\")\n",
             clean_conclusion="For each victim race, black defendants receive the death penalty more often (gaps {white_victim_black_minus_white_def} and {black_victim_black_minus_white_def}). The pooled comparison reverses because killings of white victims draw more death sentences and most white defendants had white victims.",
             key="Simpson's paradox: victim race confounds the comparison (white-victim cases get more death sentences and are mostly white-defendant cases). Within victim race, black defendants have the higher rate. Fix: stratify by victim race."),
        dict(flaw=S, task="Who was the better hitter over 1995 to 1996, Derek Jeter or David Justice?", data=BATTING,
             flawed_code=HEAD + BATTING_DF + "df['avg'] = df['hits'] / df['at_bats']\nwide = df.pivot(index='year', columns='player', values='avg')\nprint(wide.round(3))\nprint(f\"seasons_justice_higher: {(wide['Justice'] > wide['Jeter']).sum()} of 2\")\n",
             flawed_conclusion="Justice had the higher batting average in both seasons ({seasons_justice_higher}), so Justice was the better hitter over 1995 to 1996 combined.",
             clean_code=HEAD + BATTING_DF + "tot = df.groupby('player')[['hits', 'at_bats']].sum()\navg = tot['hits'] / tot['at_bats']\nprint(f\"avg_jeter: {avg['Jeter']:.3f}\")\nprint(f\"avg_justice: {avg['Justice']:.3f}\")\n",
             clean_conclusion="Over the two seasons combined Jeter hit {avg_jeter} and Justice {avg_justice}. Justice led each season, but Jeter had most of his at-bats in his strong 1996 season, so the combined average favours Jeter.",
             key="Simpson's paradox: per-season averages and the combined average disagree because the at-bat weights differ sharply across seasons. A claim about the combined period has to use the combined hits / at-bats. Fix: weight by at-bats (compute the pooled average) for the two-season question."),
        dict(flaw=S, task="Did smokers in the Whickham study have higher 20 year mortality than non-smokers?", data=WHICKHAM,
             flawed_code=HEAD + WHICKHAM_DF + "tot = df.groupby('smoker')[['dead', 'alive']].sum()\nrate = tot['dead'] / tot.sum(axis=1)\nprint(f\"mortality_smokers: {rate['yes']:.3f}\")\nprint(f\"mortality_nonsmokers: {rate['no']:.3f}\")\n",
             flawed_conclusion="Smokers had lower 20 year mortality ({mortality_smokers}) than non-smokers ({mortality_nonsmokers}). In this cohort smoking was not associated with higher mortality.",
             clean_code=HEAD + "from statsmodels.stats.contingency_tables import StratifiedTable\n" + WHICKHAM_DF + "tables = [g.set_index('smoker').loc[['yes', 'no'], ['dead', 'alive']].values + 0.5 for _, g in df.groupby('age')]\nst = StratifiedTable(tables)\nprint(f\"mh_odds_ratio: {st.oddsratio_pooled:.2f}\")\nprint(f\"share_smokers_over_65: {df[(df['smoker'] == 'yes') & df['age'].isin(['65-74', '75+'])][['dead', 'alive']].values.sum() / df[df['smoker'] == 'yes'][['dead', 'alive']].values.sum():.3f}\")\n",
             clean_conclusion="Within age groups smokers have higher mortality (Mantel-Haenszel odds ratio {mh_odds_ratio}). Smokers were younger (only {share_smokers_over_65} were 65 or older), which makes their crude death rate look lower.",
             key="Simpson's paradox: age confounds the comparison; older women were less likely to smoke and much more likely to die within 20 years. Age-stratified rates show higher mortality for smokers. Fix: stratify or adjust for age."),
        dict(flaw=S, task="Does smoking predict 20 year mortality? Fit a logistic regression.", data=WHICKHAM,
             flawed_code=HEAD + "import statsmodels.api as sm\nimport statsmodels.formula.api as smf\n" + WHICKHAM_DF + "m = smf.glm('dead + alive ~ smoker', data=df, family=sm.families.Binomial()).fit()\nprint(f\"odds_ratio_smoker: {np.exp(m.params['smoker[T.yes]']):.2f}\")\nprint(f\"p: {m.pvalues['smoker[T.yes]']:.4f}\")\n",
             flawed_conclusion="Smokers have {odds_ratio_smoker} times the odds of death (p = {p}), so smoking appears protective in this cohort.",
             clean_code=HEAD + "import statsmodels.api as sm\nimport statsmodels.formula.api as smf\n" + WHICKHAM_DF + "d = df[df['alive'] > 0]  # the 75+ group has no survivors and is dropped to keep the fit finite\nm = smf.glm('dead + alive ~ smoker + C(age)', data=d, family=sm.families.Binomial()).fit()\nprint(f\"odds_ratio_smoker: {np.exp(m.params['smoker[T.yes]']):.2f}\")\nprint(f\"p: {m.pvalues['smoker[T.yes]']:.4f}\")\n",
             clean_conclusion="With age in the model the smoker odds ratio is {odds_ratio_smoker} (p = {p}): smoking is associated with higher mortality once age is held fixed.",
             key="Simpson's paradox / omitted confounder: age drives both smoking status and death. The smoker-only model has the wrong sign. Fix: adjust for age."),
    ]


# ---------------------------------------------------------------------------------------
# Wrong test choice
# ---------------------------------------------------------------------------------------

MINWAGE = ("minwage.csv: 358 fast food restaurants surveyed before and after New Jersey raised its minimum "
           "wage; the same store is measured twice (fullBefore/fullAfter, partBefore/partAfter); location "
           "is PA or a NJ region.")
WAGE = ("wage.csv: 935 working men; includes meduc and feduc, the years of schooling of each man's mother "
        "and father.")
SOCIAL = ("social.csv: Michigan voters in a get-out-the-vote experiment; primary2004 and primary2006 record "
          "whether the same person voted in each primary; messages is the mailer arm.")
ONLINE = ("online_classroom.csv: 323 students randomized to online (format_ol = 1), blended "
          "(format_blended = 1) or face to face; falsexam is the final exam score. Different students in each format.")
DIABETES = ("diabetes2.csv: 699 children with type 2 diabetes randomized to lifestyle, met or rosi; outcome "
            "is success or failure.")


def wrong_test_cases():
    W = "wrong_test"
    hill_load = HEAD + "from scipy import stats\ndf = pd.read_csv('hillstrom.csv')\n"
    return [
        dict(flaw=W, task="Pilot readout: did the Womens E-Mail raise conversion among Multichannel customers? Use the first 100 customers per arm.", data=HILL,
             flawed_code=HEAD + "from statsmodels.stats.proportion import proportions_ztest\ndf = pd.read_csv('hillstrom.csv').query(\"channel == 'Multichannel'\")\nt = df.loc[df['segment'] == 'Womens E-Mail', 'conversion'].to_numpy()[:100]\nc = df.loc[df['segment'] == 'No E-Mail', 'conversion'].to_numpy()[:100]\nprint(f\"conversions: {t.sum()} vs {c.sum()}\")\nprint(f\"p: {proportions_ztest([t.sum(), c.sum()], [100, 100])[1]:.4f}\")\n",
             flawed_conclusion="The pilot shows {conversions} conversions (email vs control) and the two-proportion z-test gives p = {p}, so the Womens E-Mail significantly lifts Multichannel conversion.",
             clean_code=HEAD + "from scipy.stats import fisher_exact\ndf = pd.read_csv('hillstrom.csv').query(\"channel == 'Multichannel'\")\nt = df.loc[df['segment'] == 'Womens E-Mail', 'conversion'].to_numpy()[:100]\nc = df.loc[df['segment'] == 'No E-Mail', 'conversion'].to_numpy()[:100]\nprint(f\"conversions: {t.sum()} vs {c.sum()}\")\nprint(f\"p: {fisher_exact([[t.sum(), 100 - t.sum()], [c.sum(), 100 - c.sum()]])[1]:.4f}\")\n",
             clean_conclusion="With {conversions} conversions the counts are far too small for the normal approximation; Fisher's exact test gives p = {p}. The pilot does not show a significant lift.",
             key="Wrong test: a two-proportion z-test relies on a normal approximation that fails with 4 vs 0 events (expected counts far below 5). Fix: use Fisher's exact test (or an exact / permutation test); it is not significant."),
        dict(flaw=W, task="Is spend per customer related to the purchase channel?", data=HILL,
             flawed_code=hill_load + "codes = df['channel'].astype('category').cat.codes  # Multichannel=0, Phone=1, Web=2\nr, p = stats.pearsonr(codes, df['spend'])\nprint(f\"r: {r:.4f}\")\nprint(f\"p: {p:.4f}\")\n",
             flawed_conclusion="The correlation between channel and spend is r = {r} (p = {p}). Channel is unrelated to spend, so channel can be dropped from the spend analysis.",
             clean_code=hill_load + "groups = [g['spend'] for _, g in df.groupby('channel')]\nprint(df.groupby('channel')['spend'].mean().round(3))\nprint(f\"anova_p: {stats.f_oneway(*groups).pvalue:.4f}\")\n",
             clean_conclusion="Comparing mean spend across the three channels (one-way ANOVA, p = {anova_p}), spend does differ by channel; Multichannel customers spend the most.",
             key="Wrong test: channel is nominal, so coding it 0/1/2 and computing a Pearson correlation tests an arbitrary linear trend in the codes, not whether spend differs by channel. Fix: compare groups (one-way ANOVA / Welch ANOVA or Kruskal-Wallis); it is significant."),
        dict(flaw=W, task="Does conversion rate differ by zip code type?", data=HILL,
             flawed_code=hill_load + "codes = df['zip_code'].astype('category').cat.codes  # Rural=0, Surburban=1, Urban=2\nr, p = stats.pearsonr(codes, df['conversion'])\nprint(f\"r: {r:.4f}\")\nprint(f\"p: {p:.4f}\")\n",
             flawed_conclusion="Correlation between zip code type and conversion is r = {r} (p = {p}): conversion does not depend on location.",
             clean_code=hill_load + "from scipy.stats import chi2_contingency\ntable = pd.crosstab(df['zip_code'], df['conversion'])\nprint(df.groupby('zip_code')['conversion'].mean().round(4))\nprint(f\"chi2_p: {chi2_contingency(table)[1]:.4f}\")\n",
             clean_conclusion="A chi-square test of zip code type by conversion gives p = {chi2_p}: conversion does differ by location (rural customers convert most).",
             key="Wrong test: zip_code is nominal; a Pearson correlation on arbitrary category codes misses non-monotone differences. Fix: chi-square test of independence for a categorical predictor and a binary outcome; it is significant."),
        dict(flaw=W, task="Does conversion rate differ by purchase channel?", data=HILL,
             flawed_code=hill_load + "codes = df['channel'].astype('category').cat.codes  # Multichannel=0, Phone=1, Web=2\nr, p = stats.pearsonr(codes, df['conversion'])\nprint(f\"r: {r:.4f}\")\nprint(f\"p: {p:.4f}\")\n",
             flawed_conclusion="Channel and conversion are uncorrelated (r = {r}, p = {p}), so conversion is the same across channels.",
             clean_code=hill_load + "from scipy.stats import chi2_contingency\nprint(df.groupby('channel')['conversion'].mean().round(4))\nprint(f\"chi2_p: {chi2_contingency(pd.crosstab(df['channel'], df['conversion']))[1]:.5f}\")\n",
             clean_conclusion="A chi-square test gives p = {chi2_p}: conversion differs by channel (Multichannel highest, Phone lowest).",
             key="Wrong test: Pearson correlation on label-encoded nominal categories; the ordering Multichannel/Phone/Web is arbitrary and the true pattern is not linear in the codes. Fix: chi-square test of independence; it is highly significant."),
        dict(flaw=W, task="Did the Womens E-Mail lift conversion among Phone customers?", data=HILL,
             flawed_code=hill_load + "d = df.query(\"channel == 'Phone'\")\nt = d.loc[d['segment'] == 'Womens E-Mail', 'conversion']\ncontrol_rate = d.loc[d['segment'] == 'No E-Mail', 'conversion'].mean()\nprint(f\"treat_rate: {t.mean():.4f}\")\nprint(f\"control_rate: {control_rate:.4f}\")\nprint(f\"p: {stats.ttest_1samp(t, control_rate).pvalue:.4f}\")\n",
             flawed_conclusion="The emailed group converted at {treat_rate} against the control benchmark of {control_rate}; a one-sample t-test against the benchmark gives p = {p}, a significant lift.",
             clean_code=hill_load + "d = df.query(\"channel == 'Phone'\")\nt = d.loc[d['segment'] == 'Womens E-Mail', 'conversion']\nc = d.loc[d['segment'] == 'No E-Mail', 'conversion']\nprint(f\"treat_rate: {t.mean():.4f}\")\nprint(f\"control_rate: {c.mean():.4f}\")\nprint(f\"p: {stats.ttest_ind(t, c, equal_var=False).pvalue:.4f}\")\n",
             clean_conclusion="Comparing the two arms with a two-sample test (Welch, p = {p}), the lift from {control_rate} to {treat_rate} is not significant.",
             key="Wrong test: the control rate is itself an estimate from a sample of similar size, but the one-sample t-test treats it as a known constant and ignores its sampling error, understating the standard error. Fix: two-sample test (two-proportion z-test or Welch t-test); not significant."),
        dict(flaw=W, task="Do the three diabetes treatments differ in success rate?", data=DIABETES,
             flawed_code=HEAD + "from scipy.stats import chi2_contingency\ndf = pd.read_csv('diabetes2.csv')\ncounts = pd.crosstab(df['treatment'], df['outcome'])\npct = counts.div(counts.sum(axis=1), axis=0) * 100  # success / failure percent per arm\nprint(pct.round(1))\nprint(f\"low: {pct['success'].min():.1f}\")\nprint(f\"high: {pct['success'].max():.1f}\")\nprint(f\"p: {chi2_contingency(pct)[1]:.4f}\")\n",
             flawed_conclusion="Success rates range from {low} to {high} percent across arms, but the chi-square test gives p = {p}, so the treatments do not differ.",
             clean_code=HEAD + "from scipy.stats import chi2_contingency\ndf = pd.read_csv('diabetes2.csv')\ncounts = pd.crosstab(df['treatment'], df['outcome'])\nprint(counts)\nprint(f\"p: {chi2_contingency(counts)[1]:.4f}\")\n",
             clean_conclusion="On the actual counts the chi-square test gives p = {p}: success rates differ across treatments (rosi highest).",
             key="Wrong test input: the chi-square test was run on row percentages instead of counts. The statistic depends on sample size, so rescaling each arm to 100 throws away most of the information and changes the p-value. Fix: run the test on the raw counts; it is significant."),
        dict(flaw=W, task="Did the Mens E-Mail raise spend among rural customers?", data=HILL,
             flawed_code=hill_load + "d = df.query(\"zip_code == 'Rural'\")\nt = d.loc[d['segment'] == 'Mens E-Mail', 'spend'].to_numpy()\nc = d.loc[d['segment'] == 'No E-Mail', 'spend'].to_numpy()\nn = min(len(t), len(c))\nprint(f\"n_pairs: {n}\")\nprint(f\"p: {stats.ttest_rel(t[:n], c[:n]).pvalue:.4f}\")\n",
             flawed_conclusion="A paired t-test over {n_pairs} customer pairs gives p = {p}: the Mens E-Mail significantly increased rural spend.",
             clean_code=hill_load + "d = df.query(\"zip_code == 'Rural'\")\nt = d.loc[d['segment'] == 'Mens E-Mail', 'spend']\nc = d.loc[d['segment'] == 'No E-Mail', 'spend']\nprint(f\"n_treat: {len(t)}\")\nprint(f\"n_control: {len(c)}\")\nprint(f\"p: {stats.ttest_ind(t, c, equal_var=False).pvalue:.4f}\")\n",
             clean_conclusion="Using all {n_treat} and {n_control} customers in an independent-samples Welch t-test, p = {p}.",
             key="Wrong test: the two arms are different customers, so there are no pairs; a paired t-test on rows matched by file position is meaningless and also drops the extra customers in the larger arm. Fix: independent two-sample (Welch) t-test on all customers."),
        dict(flaw=W, task="Do online students score differently on the final exam than face to face students?", data=ONLINE,
             flawed_code=HEAD + "from scipy import stats\ndf = pd.read_csv('online_classroom.csv')\nonline = df.loc[df['format_ol'] == 1, 'falsexam'].to_numpy()\nf2f = df.loc[(df['format_ol'] == 0) & (df['format_blended'] == 0), 'falsexam'].to_numpy()\nn = min(len(online), len(f2f))\nres = stats.ttest_rel(online[:n], f2f[:n])\nprint(f\"n_pairs: {n}\")\nprint(f\"mean_diff: {online[:n].mean() - f2f[:n].mean():.2f}\")\nprint(f\"p: {res.pvalue:.4f}\")\n",
             flawed_conclusion="A paired t-test on {n_pairs} student pairs shows online students score {mean_diff} points lower (p = {p}).",
             clean_code=HEAD + "from scipy import stats\ndf = pd.read_csv('online_classroom.csv')\nonline = df.loc[df['format_ol'] == 1, 'falsexam']\nf2f = df.loc[(df['format_ol'] == 0) & (df['format_blended'] == 0), 'falsexam']\nprint(f\"mean_diff: {online.mean() - f2f.mean():.2f}\")\nprint(f\"p: {stats.ttest_ind(online, f2f, equal_var=False).pvalue:.4f}\")\n",
             clean_conclusion="With an independent-samples Welch t-test on all students, online students score {mean_diff} points lower (p = {p}).",
             key="Wrong test: online and face to face students are different people; pairing them by row order is arbitrary and discards students from the larger group. Fix: independent two-sample (Welch) t-test."),
        dict(flaw=W, task="Did full time employment at New Jersey restaurants change after the minimum wage increase?", data=MINWAGE,
             flawed_code=HEAD + "from scipy import stats\ndf = pd.read_csv('minwage.csv')\nnj = df[df['location'] != 'PA'].dropna(subset=['fullBefore', 'fullAfter'])\nprint(f\"mean_change: {(nj['fullAfter'] - nj['fullBefore']).mean():.2f}\")\nprint(f\"p: {stats.ttest_ind(nj['fullBefore'], nj['fullAfter']).pvalue:.4f}\")\n",
             flawed_conclusion="Full time employment changed by {mean_change} workers per store; a two-sample t-test of before vs after gives p = {p}, so there is no change.",
             clean_code=HEAD + "from scipy import stats\ndf = pd.read_csv('minwage.csv')\nnj = df[df['location'] != 'PA'].dropna(subset=['fullBefore', 'fullAfter'])\nprint(f\"mean_change: {(nj['fullAfter'] - nj['fullBefore']).mean():.2f}\")\nprint(f\"p: {stats.ttest_rel(nj['fullBefore'], nj['fullAfter']).pvalue:.4f}\")\n",
             clean_conclusion="Using the paired t-test on each store's before and after counts, the mean change is {mean_change} (p = {p}). Even done right this is only a before-after change; the causal question needs the PA comparison (difference in differences).",
             key="Wrong test: before and after are the same stores, so the samples are paired; an independent two-sample t-test ignores the within-store correlation. Fix: paired t-test on store-level changes (and, for a causal claim, difference in differences against PA)."),
        dict(flaw=W, task="Do the men's mothers and fathers differ in years of schooling?", data=WAGE,
             flawed_code=HEAD + "from scipy import stats\ndf = pd.read_csv('wage.csv').dropna(subset=['meduc', 'feduc'])\nprint(f\"mean_diff: {(df['meduc'] - df['feduc']).mean():.3f}\")\nres = stats.ttest_ind(df['meduc'], df['feduc'])\nprint(f\"t: {res.statistic:.2f}\")\nprint(f\"p: {res.pvalue:.4f}\")\n",
             flawed_conclusion="Mothers have {mean_diff} more years of schooling on average (two-sample t = {t}, p = {p}).",
             clean_code=HEAD + "from scipy import stats\ndf = pd.read_csv('wage.csv').dropna(subset=['meduc', 'feduc'])\nprint(f\"mean_diff: {(df['meduc'] - df['feduc']).mean():.3f}\")\nres = stats.ttest_rel(df['meduc'], df['feduc'])\nprint(f\"t: {res.statistic:.2f}\")\nprint(f\"p: {res.pvalue:.2e}\")\n",
             clean_conclusion="With a paired t-test on each man's parents (t = {t}, p = {p}), mothers average {mean_diff} more years; pairing removes the shared family-level variation.",
             key="Wrong test: mother's and father's schooling come from the same family, so the two columns are paired and correlated; an independent-samples t-test uses the wrong standard error. Fix: paired t-test (or Wilcoxon signed-rank) on the within-family differences."),
        dict(flaw=W, task="Among voters who got the Neighbors mailer, did turnout change between the 2004 and 2006 primaries?", data=SOCIAL,
             flawed_code=HEAD + "from statsmodels.stats.proportion import proportions_ztest\ndf = pd.read_csv('social.csv').query(\"messages == 'Neighbors'\")\nyes = [df['primary2004'].sum(), df['primary2006'].sum()]\nprint(f\"turnout_2004: {df['primary2004'].mean():.3f}\")\nprint(f\"turnout_2006: {df['primary2006'].mean():.3f}\")\nprint(f\"z: {proportions_ztest(yes, [len(df), len(df)])[0]:.2f}\")\n",
             flawed_conclusion="Turnout fell from {turnout_2004} to {turnout_2006}; a two-proportion z-test gives z = {z}, a significant drop.",
             clean_code=HEAD + "from statsmodels.stats.contingency_tables import mcnemar\ndf = pd.read_csv('social.csv').query(\"messages == 'Neighbors'\")\ntable = pd.crosstab(df['primary2004'], df['primary2006'])\nres = mcnemar(table.values, exact=False)\nprint(f\"turnout_2004: {df['primary2004'].mean():.3f}\")\nprint(f\"turnout_2006: {df['primary2006'].mean():.3f}\")\nprint(f\"chi2: {res.statistic:.1f}\")\n",
             clean_conclusion="Each voter appears in both years, so McNemar's test on the discordant pairs is the right test (chi-square = {chi2}). Turnout changed from {turnout_2004} to {turnout_2006}.",
             key="Wrong test: the 2004 and 2006 turnout indicators are for the same people (paired binary data); a two-proportion z-test assumes independent samples. Fix: McNemar's test on the discordant pairs."),
        dict(flaw=W, task="Do Web customers who bought after the Mens E-Mail spend more than those who bought after the Womens E-Mail?", data=HILL,
             flawed_code=hill_load + "d = df.query(\"channel == 'Web' and spend > 0\")\nm = d.loc[d['segment'] == 'Mens E-Mail', 'spend'].to_numpy()\nw = d.loc[d['segment'] == 'Womens E-Mail', 'spend'].to_numpy()\nn = min(len(m), len(w))\nprint(f\"n_pairs: {n}\")\nprint(f\"p: {stats.wilcoxon(m[:n], w[:n]).pvalue:.4f}\")\n",
             flawed_conclusion="A Wilcoxon signed-rank test over {n_pairs} buyer pairs gives p = {p}, so there is no significant difference in order size between the two emails.",
             clean_code=hill_load + "d = df.query(\"channel == 'Web' and spend > 0\")\nm = d.loc[d['segment'] == 'Mens E-Mail', 'spend']\nw = d.loc[d['segment'] == 'Womens E-Mail', 'spend']\nprint(f\"n_mens: {len(m)}\")\nprint(f\"n_womens: {len(w)}\")\nprint(f\"p: {stats.mannwhitneyu(m, w, alternative='two-sided').pvalue:.4f}\")\n",
             clean_conclusion="With a Mann-Whitney U test on all {n_mens} and {n_womens} buyers, p = {p}: no evidence that order size differs. Spend among buyers is also conditioned on converting, so it is not a randomized comparison.",
             key="Wrong test: the Wilcoxon signed-rank test is for paired observations; Mens and Womens buyers are different customers, paired arbitrarily by row order. Fix: Mann-Whitney U (or Welch t-test / bootstrap) on independent groups; also note that conditioning on buyers breaks randomization."),
    ]
