"""Hand written prompts on QRData data sheets (questions are new, not QRData's own).

Each entry: files, a one line data description, the question, the reference solution and
the answer tolerance. The reference answer is whatever the code prints.
"""

R2 = {"kind": "rel", "rel": 0.03, "decimals": 2}
R3 = {"kind": "rel", "rel": 0.03, "decimals": 3}
R4 = {"kind": "rel", "rel": 0.03, "decimals": 4}
P = {"kind": "pvalue"}
EXACT = {"kind": "exact"}

HEAD = "import pandas as pd\nimport numpy as np\n"

SHEETS = [
    # --- resume.csv: callback field experiment (Bertrand and Mullainathan) -------------
    dict(files=["resume.csv"], skill="difference in proportions",
         description="resume.csv: fictitious resumes sent to job ads; firstname, sex, race (black/white, signalled by the name), call (1 if the employer called back).",
         question="What is the callback rate for white-sounding names minus the rate for black-sounding names, in percentage points? Round to 2 decimals.",
         code=HEAD + "df = pd.read_csv('resume.csv')\nr = df.groupby('race')['call'].mean()\nprint(round(100 * (r['white'] - r['black']), 2))\n", tol=R2),
    dict(files=["resume.csv"], skill="chi-square test",
         description="resume.csv: fictitious resumes sent to job ads; firstname, sex, race (black/white, signalled by the name), call (1 if the employer called back).",
         question="Test whether callback is independent of race with a chi-square test on the 2x2 table, without Yates continuity correction. Report the p-value rounded to 4 decimals.",
         code=HEAD + "from scipy.stats import chi2_contingency\ndf = pd.read_csv('resume.csv')\nprint(round(chi2_contingency(pd.crosstab(df['race'], df['call']), correction=False)[1], 4))\n", tol=P),
    dict(files=["resume.csv"], skill="logistic regression",
         description="resume.csv: fictitious resumes sent to job ads; firstname, sex, race (black/white, signalled by the name), call (1 if the employer called back).",
         question="Fit a logistic regression of call on race and sex (both categorical, white and female as reference levels). Report the odds ratio for black versus white, rounded to 3 decimals.",
         code=HEAD + "import statsmodels.formula.api as smf\ndf = pd.read_csv('resume.csv')\nm = smf.logit(\"call ~ C(race, Treatment('white')) + C(sex, Treatment('female'))\", data=df).fit(disp=0)\nprint(round(np.exp(m.params[\"C(race, Treatment('white'))[T.black]\"]), 3))\n", tol=R3),
    dict(files=["resume.csv"], skill="stratified analysis",
         description="resume.csv: fictitious resumes sent to job ads; firstname, sex, race (black/white, signalled by the name), call (1 if the employer called back).",
         question="Compute the Mantel-Haenszel pooled odds ratio of callback for white versus black names, stratified by sex. Round to 3 decimals.",
         code=HEAD + "from statsmodels.stats.contingency_tables import StratifiedTable\ndf = pd.read_csv('resume.csv')\ntables = []\nfor _, g in df.groupby('sex'):\n    t = pd.crosstab(g['race'], g['call']).loc[['white', 'black'], [1, 0]].values\n    tables.append(t)\nprint(round(StratifiedTable(tables).oddsratio_pooled, 3))\n", tol=R3),

    # --- learning_mindset.csv: National Study of Learning Mindsets (simulated) --------
    dict(files=["learning_mindset.csv"], skill="difference in means",
         description="learning_mindset.csv: students offered a growth mindset intervention (intervention = 1) or not; achievement_score is standardized; other columns are student and school characteristics.",
         question="What is the naive difference in mean achievement_score between students with intervention = 1 and intervention = 0? Round to 4 decimals.",
         code=HEAD + "df = pd.read_csv('learning_mindset.csv')\nm = df.groupby('intervention')['achievement_score'].mean()\nprint(round(m[1] - m[0], 4))\n", tol=R4),
    dict(files=["learning_mindset.csv"], skill="regression adjustment",
         description="learning_mindset.csv: students offered a growth mindset intervention (intervention = 1) or not; achievement_score is standardized; other columns are student and school characteristics.",
         question="Regress achievement_score on intervention, success_expect, frst_in_family, gender (categorical), ethnicity (categorical), school_urbanicity (categorical), school_mindset, school_achievement, school_ethnic_minority, school_poverty and school_size with OLS. Report the intervention coefficient rounded to 4 decimals.",
         code=HEAD + "import statsmodels.formula.api as smf\ndf = pd.read_csv('learning_mindset.csv')\nf = 'achievement_score ~ intervention + success_expect + frst_in_family + C(gender) + C(ethnicity) + C(school_urbanicity) + school_mindset + school_achievement + school_ethnic_minority + school_poverty + school_size'\nprint(round(smf.ols(f, data=df).fit().params['intervention'], 4))\n", tol=R4),
    dict(files=["learning_mindset.csv"], skill="clustered standard errors",
         description="learning_mindset.csv: students offered a growth mindset intervention (intervention = 1) or not; achievement_score is standardized; other columns are student and school characteristics.",
         question="Regress achievement_score on intervention alone with OLS and compute standard errors clustered by schoolid. Report the clustered standard error of the intervention coefficient, rounded to 4 decimals.",
         code=HEAD + "import statsmodels.formula.api as smf\ndf = pd.read_csv('learning_mindset.csv')\nm = smf.ols('achievement_score ~ intervention', data=df).fit(cov_type='cluster', cov_kwds={'groups': df['schoolid']})\nprint(round(m.bse['intervention'], 4))\n", tol=R4),
    dict(files=["learning_mindset.csv"], skill="covariate balance",
         description="learning_mindset.csv: students offered a growth mindset intervention (intervention = 1) or not; achievement_score is standardized; other columns are student and school characteristics.",
         question="Is the intervention balanced on success_expect? Run Welch's t-test of success_expect between intervention groups and report the p-value rounded to 4 decimals.",
         code=HEAD + "from scipy import stats\ndf = pd.read_csv('learning_mindset.csv')\na = df.loc[df['intervention'] == 1, 'success_expect']\nb = df.loc[df['intervention'] == 0, 'success_expect']\nprint(round(stats.ttest_ind(a, b, equal_var=False).pvalue, 4))\n", tol=P),

    # --- online_classroom.csv ---------------------------------------------------------
    dict(files=["online_classroom.csv"], skill="regression with dummies",
         description="online_classroom.csv: students randomized to online (format_ol = 1), blended (format_blended = 1) or face to face (both 0); falsexam is the final exam score; gender and race dummies.",
         question="Regress falsexam on format_ol, format_blended, gender, asian, black, hawaiian, hispanic and unknown with OLS (white is the omitted race). Report the format_ol coefficient rounded to 3 decimals.",
         code=HEAD + "import statsmodels.formula.api as smf\ndf = pd.read_csv('online_classroom.csv')\nm = smf.ols('falsexam ~ format_ol + format_blended + gender + asian + black + hawaiian + hispanic + unknown', data=df).fit()\nprint(round(m.params['format_ol'], 3))\n", tol=R3),
    dict(files=["online_classroom.csv"], skill="Welch t-test",
         description="online_classroom.csv: students randomized to online (format_ol = 1), blended (format_blended = 1) or face to face (both 0); falsexam is the final exam score; gender and race dummies.",
         question="Compare falsexam for online students with face to face students (format_ol = 0 and format_blended = 0) using Welch's t-test. Report the two-sided p-value rounded to 4 decimals.",
         code=HEAD + "from scipy import stats\ndf = pd.read_csv('online_classroom.csv')\non = df.loc[df['format_ol'] == 1, 'falsexam']\nf2f = df.loc[(df['format_ol'] == 0) & (df['format_blended'] == 0), 'falsexam']\nprint(round(stats.ttest_ind(on, f2f, equal_var=False).pvalue, 4))\n", tol=P),
    dict(files=["online_classroom.csv"], skill="one-way ANOVA",
         description="online_classroom.csv: students randomized to online (format_ol = 1), blended (format_blended = 1) or face to face (both 0); falsexam is the final exam score; gender and race dummies.",
         question="Run a one-way ANOVA of falsexam across the three class formats. Report the F statistic rounded to 3 decimals.",
         code=HEAD + "from scipy import stats\ndf = pd.read_csv('online_classroom.csv')\nfmt = np.where(df['format_ol'] == 1, 'online', np.where(df['format_blended'] == 1, 'blended', 'f2f'))\nprint(round(stats.f_oneway(*[df.loc[fmt == k, 'falsexam'] for k in ['online', 'blended', 'f2f']]).statistic, 3))\n", tol=R3),

    # --- billboard_impact.csv: difference in differences ------------------------------
    dict(files=["billboard_impact.csv"], skill="difference in differences",
         description="billboard_impact.csv: bank deposits for customers in Porto Alegre (poa = 1, where billboards went up) and Florianopolis (poa = 0), in May (jul = 0) and July (jul = 1).",
         question="Compute the difference in differences estimate of the billboard effect on mean deposits: (POA July - POA May) - (Florianopolis July - Florianopolis May). Round to 2 decimals.",
         code=HEAD + "df = pd.read_csv('billboard_impact.csv')\nm = df.groupby(['poa', 'jul'])['deposits'].mean()\nprint(round((m[1, 1] - m[1, 0]) - (m[0, 1] - m[0, 0]), 2))\n", tol=R2),
    dict(files=["billboard_impact.csv"], skill="difference in differences",
         description="billboard_impact.csv: bank deposits for customers in Porto Alegre (poa = 1, where billboards went up) and Florianopolis (poa = 0), in May (jul = 0) and July (jul = 1).",
         question="Fit OLS deposits ~ poa * jul with HC1 robust standard errors. Report the p-value of the poa:jul interaction rounded to 4 decimals.",
         code=HEAD + "import statsmodels.formula.api as smf\ndf = pd.read_csv('billboard_impact.csv')\nprint(round(smf.ols('deposits ~ poa * jul', data=df).fit(cov_type='HC1').pvalues['poa:jul'], 4))\n", tol=P),

    # --- app_engagement_push.csv: noncompliance ---------------------------------------
    dict(files=["app_engagement_push.csv"], skill="intent to treat",
         description="app_engagement_push.csv: users randomly assigned a push notification (push_assigned); push_delivered says whether it actually arrived; in_app_purchase is spend in the app.",
         question="Estimate the intent to treat effect: mean in_app_purchase for push_assigned = 1 minus push_assigned = 0. Round to 3 decimals.",
         code=HEAD + "df = pd.read_csv('app_engagement_push.csv')\nm = df.groupby('push_assigned')['in_app_purchase'].mean()\nprint(round(m[1] - m[0], 3))\n", tol=R3),
    dict(files=["app_engagement_push.csv"], skill="instrumental variables",
         description="app_engagement_push.csv: users randomly assigned a push notification (push_assigned); push_delivered says whether it actually arrived; in_app_purchase is spend in the app.",
         question="Using push_assigned as an instrument for push_delivered, compute the Wald (IV) estimate of the effect of a delivered push on in_app_purchase. Round to 3 decimals.",
         code=HEAD + "df = pd.read_csv('app_engagement_push.csv')\ny = df.groupby('push_assigned')['in_app_purchase'].mean()\nd = df.groupby('push_assigned')['push_delivered'].mean()\nprint(round((y[1] - y[0]) / (d[1] - d[0]), 3))\n", tol=R3),
    dict(files=["app_engagement_push.csv"], skill="compliance rate",
         description="app_engagement_push.csv: users randomly assigned a push notification (push_assigned); push_delivered says whether it actually arrived; in_app_purchase is spend in the app.",
         question="What share of users assigned a push actually had it delivered? Round to 4 decimals.",
         code=HEAD + "df = pd.read_csv('app_engagement_push.csv')\nprint(round(df.loc[df['push_assigned'] == 1, 'push_delivered'].mean(), 4))\n", tol=R4),

    # --- collections_email.csv --------------------------------------------------------
    dict(files=["collections_email.csv"], skill="difference in means",
         description="collections_email.csv: debtors randomly sent a collections email (email = 1) or not; payments is the amount paid; credit_limit and risk_score are pre-treatment; opened and agreement are post-email.",
         question="What is the difference in mean payments between email = 1 and email = 0? Round to 3 decimals.",
         code=HEAD + "df = pd.read_csv('collections_email.csv')\nm = df.groupby('email')['payments'].mean()\nprint(round(m[1] - m[0], 3))\n", tol=R3),
    dict(files=["collections_email.csv"], skill="regression adjustment",
         description="collections_email.csv: debtors randomly sent a collections email (email = 1) or not; payments is the amount paid; credit_limit and risk_score are pre-treatment; opened and agreement are post-email.",
         question="Regress payments on email, credit_limit and risk_score with OLS (pre-treatment covariates only). Report the email coefficient rounded to 3 decimals.",
         code=HEAD + "import statsmodels.formula.api as smf\ndf = pd.read_csv('collections_email.csv')\nprint(round(smf.ols('payments ~ email + credit_limit + risk_score', data=df).fit().params['email'], 3))\n", tol=R3),
    dict(files=["collections_email.csv"], skill="regression adjustment",
         description="collections_email.csv: debtors randomly sent a collections email (email = 1) or not; payments is the amount paid; credit_limit and risk_score are pre-treatment; opened and agreement are post-email.",
         question="By what percent does the standard error of the email coefficient shrink when credit_limit and risk_score are added to an OLS of payments on email? Use classical OLS standard errors. Round to 2 decimals.",
         code=HEAD + "import statsmodels.formula.api as smf\ndf = pd.read_csv('collections_email.csv')\nse0 = smf.ols('payments ~ email', data=df).fit().bse['email']\nse1 = smf.ols('payments ~ email + credit_limit + risk_score', data=df).fit().bse['email']\nprint(round(100 * (1 - se1 / se0), 2))\n", tol=R2),

    # --- invest_email_rnd_test.csv ----------------------------------------------------
    dict(files=["invest_email_rnd_test.csv"], skill="chi-square test",
         description="invest_email_rnd_test.csv: customers randomly sent combinations of three emails (em1, em2, em3 are 0/1); converted is 0/1; age, income, insurance, invested are pre-treatment.",
         question="Compare conversion between customers who received em1 and those who did not, using a chi-square test of independence without continuity correction. Report the chi-square statistic rounded to 2 decimals.",
         code=HEAD + "from scipy.stats import chi2_contingency\ndf = pd.read_csv('invest_email_rnd_test.csv')\nprint(round(chi2_contingency(pd.crosstab(df['em1'], df['converted']), correction=False)[0], 2))\n", tol=R2),
    dict(files=["invest_email_rnd_test.csv"], skill="linear probability model",
         description="invest_email_rnd_test.csv: customers randomly sent combinations of three emails (em1, em2, em3 are 0/1); converted is 0/1; age, income, insurance, invested are pre-treatment.",
         question="Fit a linear probability model converted ~ em1 + em2 + em3 with OLS. Which email has the largest coefficient? Answer with em1, em2 or em3.",
         code=HEAD + "import statsmodels.formula.api as smf\ndf = pd.read_csv('invest_email_rnd_test.csv')\nprint(smf.ols('converted ~ em1 + em2 + em3', data=df).fit().params[['em1', 'em2', 'em3']].idxmax())\n", tol=EXACT),
    dict(files=["invest_email_rnd_test.csv"], skill="logistic regression",
         description="invest_email_rnd_test.csv: customers randomly sent combinations of three emails (em1, em2, em3 are 0/1); converted is 0/1; age, income, insurance, invested are pre-treatment.",
         question="Fit a logistic regression converted ~ em1 + em2 + em3 + age + income + insurance + invested. Report the odds ratio for em3 rounded to 3 decimals.",
         code=HEAD + "import statsmodels.formula.api as smf\ndf = pd.read_csv('invest_email_rnd_test.csv')\nm = smf.logit('converted ~ em1 + em2 + em3 + age + income + insurance + invested', data=df).fit(disp=0)\nprint(round(np.exp(m.params['em3']), 3))\n", tol=R3),

    # --- minwage.csv: Card and Krueger -------------------------------------------------
    dict(files=["minwage.csv"], skill="difference in differences",
         description="minwage.csv: fast food restaurants in New Jersey (location ends in NJ) and Pennsylvania (location PA) before and after NJ raised its minimum wage; fullBefore/fullAfter and partBefore/partAfter are full and part time employees.",
         question="Compute the difference in differences in mean full time employees: (NJ after - NJ before) - (PA after - PA before). Round to 3 decimals.",
         code=HEAD + "df = pd.read_csv('minwage.csv')\ndf['nj'] = df['location'] != 'PA'\nchg = (df['fullAfter'] - df['fullBefore']).groupby(df['nj']).mean()\nprint(round(chg[True] - chg[False], 3))\n", tol=R3),
    dict(files=["minwage.csv"], skill="proportion",
         description="minwage.csv: fast food restaurants in New Jersey (location ends in NJ) and Pennsylvania (location PA) before and after NJ raised its minimum wage; wageBefore/wageAfter are starting wages.",
         question="What proportion of New Jersey restaurants paid a starting wage below 5.05 before the increase? Round to 4 decimals.",
         code=HEAD + "df = pd.read_csv('minwage.csv')\nnj = df[df['location'] != 'PA']\nprint(round((nj['wageBefore'] < 5.05).mean(), 4))\n", tol=R4),

    # --- ak91.csv: Angrist and Krueger returns to schooling -----------------------------
    dict(files=["ak91.csv"], skill="OLS",
         description="ak91.csv: 1980 census men; log_wage, years_of_schooling, year_of_birth, quarter_of_birth (1 to 4), state_of_birth.",
         question="Regress log_wage on years_of_schooling with OLS (intercept, no other controls). Report the slope rounded to 4 decimals.",
         code=HEAD + "import statsmodels.formula.api as smf\ndf = pd.read_csv('ak91.csv')\nprint(round(smf.ols('log_wage ~ years_of_schooling', data=df).fit().params['years_of_schooling'], 4))\n", tol=R4),
    dict(files=["ak91.csv"], skill="instrumental variables",
         description="ak91.csv: 1980 census men; log_wage, years_of_schooling, year_of_birth, quarter_of_birth (1 to 4), state_of_birth.",
         question="Estimate the return to schooling by two stage least squares, instrumenting years_of_schooling with quarter of birth dummies (quarter 1 as the omitted level) and no other controls. Report the 2SLS coefficient rounded to 4 decimals.",
         code=HEAD + "import statsmodels.formula.api as smf\ndf = pd.read_csv('ak91.csv')\ndf['school_hat'] = smf.ols('years_of_schooling ~ C(quarter_of_birth)', data=df).fit().fittedvalues\nprint(round(smf.ols('log_wage ~ school_hat', data=df).fit().params['school_hat'], 4))\n", tol=R4),
    dict(files=["ak91.csv"], skill="weak instruments",
         description="ak91.csv: 1980 census men; log_wage, years_of_schooling, year_of_birth, quarter_of_birth (1 to 4), state_of_birth.",
         question="In the first stage regression of years_of_schooling on quarter of birth dummies (quarter 1 omitted), what is the F statistic for the joint significance of the dummies? Round to 2 decimals.",
         code=HEAD + "import statsmodels.formula.api as smf\ndf = pd.read_csv('ak91.csv')\nprint(round(smf.ols('years_of_schooling ~ C(quarter_of_birth)', data=df).fit().fvalue, 2))\n", tol=R2),

    # --- medicine_impact_recovery.csv: confounding by severity --------------------------
    dict(files=["medicine_impact_recovery.csv"], skill="difference in means",
         description="medicine_impact_recovery.csv: observational patients; sex, age, severity (0 to 1), medication (0/1), recovery (days to recover, lower is better).",
         question="What is the naive difference in mean recovery between patients with medication = 1 and medication = 0? Round to 3 decimals.",
         code=HEAD + "df = pd.read_csv('medicine_impact_recovery.csv')\nm = df.groupby('medication')['recovery'].mean()\nprint(round(m[1] - m[0], 3))\n", tol=R3),
    dict(files=["medicine_impact_recovery.csv"], skill="regression adjustment",
         description="medicine_impact_recovery.csv: observational patients; sex, age, severity (0 to 1), medication (0/1), recovery (days to recover, lower is better).",
         question="Regress recovery on medication, severity, age and sex with OLS. Report the medication coefficient rounded to 3 decimals.",
         code=HEAD + "import statsmodels.formula.api as smf\ndf = pd.read_csv('medicine_impact_recovery.csv')\nprint(round(smf.ols('recovery ~ medication + severity + age + sex', data=df).fit().params['medication'], 3))\n", tol=R3),
    dict(files=["medicine_impact_recovery.csv"], skill="propensity score weighting",
         description="medicine_impact_recovery.csv: observational patients; sex, age, severity (0 to 1), medication (0/1), recovery (days to recover, lower is better).",
         question="Fit a logistic propensity model medication ~ severity + age + sex, then estimate the average treatment effect of medication on recovery with inverse probability weighting (Horvitz-Thompson: mean(T*Y/e) - mean((1-T)*Y/(1-e))). Round to 2 decimals.",
         code=HEAD + "import statsmodels.formula.api as smf\ndf = pd.read_csv('medicine_impact_recovery.csv')\ne = smf.logit('medication ~ severity + age + sex', data=df).fit(disp=0).predict(df)\nt, y = df['medication'], df['recovery']\nprint(round(np.mean(t * y / e) - np.mean((1 - t) * y / (1 - e)), 2))\n", tol={"kind": "rel", "rel": 0.05, "decimals": 2}),

    # --- hospital_treatment.csv --------------------------------------------------------
    dict(files=["hospital_treatment.csv"], skill="difference in means",
         description="hospital_treatment.csv: patients in two hospitals; treatment (0/1), severity, days in hospital.",
         question="What is the naive difference in mean days between treated and untreated patients? Round to 2 decimals.",
         code=HEAD + "df = pd.read_csv('hospital_treatment.csv')\nm = df.groupby('treatment')['days'].mean()\nprint(round(m[1] - m[0], 2))\n", tol=R2),
    dict(files=["hospital_treatment.csv"], skill="regression adjustment",
         description="hospital_treatment.csv: patients in two hospitals; treatment (0/1), severity, days in hospital.",
         question="Regress days on treatment, severity and hospital (categorical) with OLS. Report the treatment coefficient rounded to 2 decimals.",
         code=HEAD + "import statsmodels.formula.api as smf\ndf = pd.read_csv('hospital_treatment.csv')\nprint(round(smf.ols('days ~ treatment + severity + C(hospital)', data=df).fit().params['treatment'], 2))\n", tol=R2),

    # --- wage.csv ----------------------------------------------------------------------
    dict(files=["wage.csv"], skill="log-linear regression",
         description="wage.csv: monthly wage of working men with hours, IQ, educ (years), exper, tenure, age, married, black, south, urban, sibs, brthord, meduc, feduc.",
         question="Regress log(wage) on educ, exper and tenure with OLS. Report the educ coefficient rounded to 4 decimals.",
         code=HEAD + "import statsmodels.formula.api as smf\ndf = pd.read_csv('wage.csv')\nprint(round(smf.ols('np.log(wage) ~ educ + exper + tenure', data=df).fit().params['educ'], 4))\n", tol=R4),
    dict(files=["wage.csv"], skill="omitted variable bias",
         description="wage.csv: monthly wage of working men with hours, IQ, educ (years), exper, tenure, age, married, black, south, urban, sibs, brthord, meduc, feduc.",
         question="How much does the educ coefficient in an OLS of log(wage) on educ, exper and tenure fall when IQ is added as a control? Report (coef without IQ - coef with IQ) rounded to 4 decimals.",
         code=HEAD + "import statsmodels.formula.api as smf\ndf = pd.read_csv('wage.csv')\nb0 = smf.ols('np.log(wage) ~ educ + exper + tenure', data=df).fit().params['educ']\nb1 = smf.ols('np.log(wage) ~ educ + exper + tenure + IQ', data=df).fit().params['educ']\nprint(round(b0 - b1, 4))\n", tol=R4),
    dict(files=["wage.csv"], skill="correlation",
         description="wage.csv: monthly wage of working men with hours, IQ, educ (years), exper, tenure, age, married, black, south, urban, sibs, brthord, meduc, feduc.",
         question="What is the Spearman rank correlation between IQ and educ? Round to 4 decimals.",
         code=HEAD + "from scipy import stats\ndf = pd.read_csv('wage.csv')\nprint(round(stats.spearmanr(df['IQ'], df['educ']).correlation, 4))\n", tol=R4),

    # --- social.csv: Gerber, Green and Larimer social pressure experiment ----------------
    dict(files=["social.csv"], skill="difference in proportions",
         description="social.csv: Michigan households randomized to get a get-out-the-vote mailer (messages: Control, Civic Duty, Hawthorne, Neighbors); primary2006 is 1 if the person voted; primary2004 is past turnout; hhsize.",
         question="What is the 2006 turnout rate of the Neighbors group minus the Control group, in percentage points? Round to 3 decimals.",
         code=HEAD + "df = pd.read_csv('social.csv')\nr = df.groupby('messages')['primary2006'].mean()\nprint(round(100 * (r['Neighbors'] - r['Control']), 3))\n", tol=R3),
    dict(files=["social.csv"], skill="regression adjustment",
         description="social.csv: Michigan households randomized to get a get-out-the-vote mailer (messages: Control, Civic Duty, Hawthorne, Neighbors); primary2006 is 1 if the person voted; primary2004 is past turnout; hhsize.",
         question="Fit OLS primary2006 ~ C(messages, Treatment('Control')) + primary2004 + hhsize. Report the coefficient for Hawthorne rounded to 4 decimals.",
         code=HEAD + "import statsmodels.formula.api as smf\ndf = pd.read_csv('social.csv')\nm = smf.ols(\"primary2006 ~ C(messages, Treatment('Control')) + primary2004 + hhsize\", data=df).fit()\nprint(round(m.params[\"C(messages, Treatment('Control'))[T.Hawthorne]\"], 4))\n", tol=R4),
    dict(files=["social.csv"], skill="regression t statistic",
         description="social.csv: Michigan individuals; households were randomized to get a get-out-the-vote mailer (messages: Control, Civic Duty, Hawthorne, Neighbors); primary2006 is 1 if the person voted; primary2004 is past turnout; hhsize.",
         question="Treat each individual as independent and fit OLS primary2006 ~ C(messages, Treatment('Control')). What is the t statistic of the Civic Duty coefficient with classical standard errors? Round to 2 decimals.",
         code=HEAD + "import statsmodels.formula.api as smf\ndf = pd.read_csv('social.csv')\nm = smf.ols(\"primary2006 ~ C(messages, Treatment('Control'))\", data=df).fit()\nprint(round(m.tvalues[\"C(messages, Treatment('Control'))[T.Civic Duty]\"], 2))\n", tol=R2),

    # --- small clinical tables -------------------------------------------------------------
    dict(files=["diabetes2.csv"], skill="chi-square test",
         description="diabetes2.csv: children with type 2 diabetes randomized to lifestyle, met (metformin) or rosi (rosiglitazone) treatment; outcome is success or failure.",
         question="Test whether outcome is independent of treatment (3x2 table) with a chi-square test. Report the p-value rounded to 4 decimals.",
         code=HEAD + "from scipy.stats import chi2_contingency\ndf = pd.read_csv('diabetes2.csv')\nprint(round(chi2_contingency(pd.crosstab(df['treatment'], df['outcome']))[1], 4))\n", tol=P),
    dict(files=["diabetes2.csv"], skill="multiple comparisons",
         description="diabetes2.csv: children with type 2 diabetes randomized to lifestyle, met (metformin) or rosi (rosiglitazone) treatment; outcome is success or failure.",
         question="Run the three pairwise two-proportion z-tests of success rate (pooled, no continuity correction) between treatments and apply a Bonferroni correction. Report the smallest adjusted p-value rounded to 4 decimals.",
         code=HEAD + "from itertools import combinations\nfrom statsmodels.stats.proportion import proportions_ztest\ndf = pd.read_csv('diabetes2.csv')\ns = df['outcome'].eq('success').groupby(df['treatment'])\np = [proportions_ztest([s.sum()[a], s.sum()[b]], [s.count()[a], s.count()[b]])[1] for a, b in combinations(s.sum().index, 2)]\nprint(round(min(1.0, 3 * min(p)), 4))\n", tol=P),
    dict(files=["stent365.csv"], skill="exact test",
         description="stent365.csv: patients at risk of stroke randomized to a stent (treatment) or medical management (control); outcome is stroke or no event after 365 days.",
         question="Run a two-sided Fisher exact test of stroke versus group. Report the p-value rounded to 4 decimals.",
         code=HEAD + "from scipy.stats import fisher_exact\ndf = pd.read_csv('stent365.csv')\nprint(round(fisher_exact(pd.crosstab(df['group'], df['outcome']))[1], 4))\n", tol=P),
    dict(files=["stent365.csv"], skill="relative risk",
         description="stent365.csv: patients at risk of stroke randomized to a stent (treatment) or medical management (control); outcome is stroke or no event after 365 days.",
         question="What is the relative risk of stroke, treatment divided by control? Round to 3 decimals.",
         code=HEAD + "df = pd.read_csv('stent365.csv')\nr = df['outcome'].eq('stroke').groupby(df['group']).mean()\nprint(round(r['treatment'] / r['control'], 3))\n", tol=R3),
    dict(files=["cpr.csv"], skill="exact test",
         description="cpr.csv: patients who got CPR, randomized to a blood thinner (treatment) or not (control); outcome is survived or died.",
         question="Run a two-sided Fisher exact test of survival versus group and report the p-value rounded to 4 decimals.",
         code=HEAD + "from scipy.stats import fisher_exact\ndf = pd.read_csv('cpr.csv')\nprint(round(fisher_exact(pd.crosstab(df['group'], df['outcome']))[1], 4))\n", tol=P),
    dict(files=["sinusitis.csv"], skill="two-proportion z-test",
         description="sinusitis.csv: sinusitis patients randomized to antibiotics (treatment) or placebo (control); self_reported_improvement is yes or no.",
         question="Run a two-sided two-proportion z-test (pooled, no continuity correction) of improvement rate, treatment vs control. Report the p-value rounded to 4 decimals.",
         code=HEAD + "from statsmodels.stats.proportion import proportions_ztest\ndf = pd.read_csv('sinusitis.csv')\ng = df['self_reported_improvement'].eq('yes').groupby(df['group'])\nprint(round(proportions_ztest(g.sum().values, g.count().values)[1], 4))\n", tol=P),
    dict(files=["sleep_deprivation.csv"], skill="chi-square test",
         description="sleep_deprivation.csv: survey of transportation workers and a control group; sleep is hours on work days (<6, 6-8, >8); profession.",
         question="Test whether sleep category is independent of profession with a chi-square test. Report the chi-square statistic rounded to 2 decimals.",
         code=HEAD + "from scipy.stats import chi2_contingency\ndf = pd.read_csv('sleep_deprivation.csv')\nprint(round(chi2_contingency(pd.crosstab(df['profession'], df['sleep']))[0], 2))\n", tol=R2),

    # --- gss2010.csv -----------------------------------------------------------------------
    dict(files=["gss2010.csv"], skill="one-way ANOVA",
         description="gss2010.csv: 2010 General Social Survey; hrsrelax (hours relaxing per day), mntlhlth, hrs1, degree, grass. Missing values are NA.",
         question="Drop missing hrsrelax and run a one-way ANOVA of hrsrelax across degree levels. Report the F statistic rounded to 3 decimals.",
         code=HEAD + "from scipy import stats\ndf = pd.read_csv('gss2010.csv').dropna(subset=['hrsrelax'])\nprint(round(stats.f_oneway(*[g['hrsrelax'] for _, g in df.groupby('degree')]).statistic, 3))\n", tol=R3),
    dict(files=["gss2010.csv"], skill="nonparametric test",
         description="gss2010.csv: 2010 General Social Survey; hrsrelax (hours relaxing per day), mntlhlth, hrs1, degree, grass. Missing values are NA.",
         question="Drop missing hrsrelax and compare BACHELOR with HIGH SCHOOL respondents using a two-sided Mann-Whitney U test. Report the p-value rounded to 4 decimals.",
         code=HEAD + "from scipy import stats\ndf = pd.read_csv('gss2010.csv').dropna(subset=['hrsrelax'])\na = df.loc[df['degree'] == 'BACHELOR', 'hrsrelax']\nb = df.loc[df['degree'] == 'HIGH SCHOOL', 'hrsrelax']\nprint(round(stats.mannwhitneyu(a, b, alternative='two-sided').pvalue, 4))\n", tol=P),

    # --- women.csv: Chattopadhyay and Duflo reserved council seats ----------------------
    dict(files=["women.csv"], skill="difference in means",
         description="women.csv: Indian village councils (GP); reserved = 1 if the leader seat was randomly reserved for a woman; female leader; irrigation and water are new public goods projects.",
         question="What is the difference in mean water projects between reserved and unreserved villages? Round to 3 decimals.",
         code=HEAD + "df = pd.read_csv('women.csv')\nm = df.groupby('reserved')['water'].mean()\nprint(round(m[1] - m[0], 3))\n", tol=R3),
    dict(files=["women.csv"], skill="Welch t-test",
         description="women.csv: Indian village councils (GP); reserved = 1 if the leader seat was randomly reserved for a woman; female leader; irrigation and water are new public goods projects.",
         question="Run Welch's t-test of irrigation between reserved and unreserved villages. Report the two-sided p-value rounded to 4 decimals.",
         code=HEAD + "from scipy import stats\ndf = pd.read_csv('women.csv')\nprint(round(stats.ttest_ind(df.loc[df['reserved'] == 1, 'irrigation'], df.loc[df['reserved'] == 0, 'irrigation'], equal_var=False).pvalue, 4))\n", tol=P),

    # --- drinking.csv: regression discontinuity at age 21 -------------------------------
    dict(files=["drinking.csv"], skill="regression discontinuity",
         description="drinking.csv: US death rates per 100,000 by age cell (agecell, in years, around 19 to 23); all is deaths from all causes; other columns split by cause.",
         question="Estimate the jump in the all-cause death rate at age 21 with a regression discontinuity: OLS of all on over21, (agecell - 21) and over21 * (agecell - 21), where over21 = agecell >= 21. Report the over21 coefficient rounded to 3 decimals.",
         code=HEAD + "import statsmodels.formula.api as smf\ndf = pd.read_csv('drinking.csv').dropna(subset=['all'])\ndf['over21'] = (df['agecell'] >= 21).astype(int)\ndf['run'] = df['agecell'] - 21\nprint(round(smf.ols('all ~ over21 * run', data=df).fit().params['over21'], 3))\n", tol=R3),

    # --- smoking2.csv: California Proposition 99 -----------------------------------------
    dict(files=["smoking2.csv"], skill="difference in differences",
         description="smoking2.csv: state by year cigarette sales per capita (cigsale), 1970 to 2000; california is True for California; after_treatment is True for years after Proposition 99 (1989 on).",
         question="Compute the difference in differences estimate of Proposition 99 on mean cigsale: (CA after - CA before) - (other states after - other states before). Round to 3 decimals.",
         code=HEAD + "df = pd.read_csv('smoking2.csv')\nm = df.groupby(['california', 'after_treatment'])['cigsale'].mean()\nprint(round((m[True, True] - m[True, False]) - (m[False, True] - m[False, False]), 3))\n", tol=R3),
]


def build():
    items = []
    counters = {}
    for s in SHEETS:
        stem = s["files"][0].replace(".csv", "")
        counters[stem] = counters.get(stem, 0) + 1
        items.append({
            "id": f"qs_{stem}_{counters[stem]}",
            "source": "qrdata_sheets",
            "files": s["files"],
            "description": s["description"],
            "question": s["question"],
            "reference_code": s["code"],
            "tolerance": s["tol"],
            "skill": s["skill"],
            "template": "hand_written",
        })
    return items
