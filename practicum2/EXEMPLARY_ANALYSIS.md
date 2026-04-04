# Practicum 2 — Exemplary Answer Analysis

This document breaks down *exactly* how every exemplary answer in the study practicum was constructed, what separated it from satisfactory/not-yet, and what you must replicate in the actual practicum to earn full manual-graded points.

---

## Meta-Principles That Apply to Every Question

Before diving question-by-question, here are the cross-cutting patterns shared by every exemplary response:

### 1. Code and Writeup Must Mirror Each Other Exactly
The grader reads your code *and* your writeup. If you say "I used OrdinalEncoder because rating has natural order" but your code uses OneHotEncoder, you are automatically Not Yet. Every sentence in the writeup must correspond to something real in the code.

### 2. Use Variables — Never Re-Type Strings
Exemplary code uses `col_encode` and `col_cat` variables everywhere downstream. Not-Yet code copy-pastes `'Rating'` and `'Source'` as raw strings throughout, making it clear the student doesn't understand what they built. Always reference the variable.

### 3. No For/While Loops Where Library Alternatives Exist
Bootstrap must use `np.random.choice(..., size=(10000, n))` + `np.average(..., axis=1)`. A for loop bootstrapping 10,000 iterations is an automatic ceiling at Not Yet.

### 4. Address the Data Structure Implications
Every question requires you to consciously choose `anime` vs `df_merged` and justify it. Not-Yet answers ignore that `df_merged` has duplicate rows per anime (one per genre). Exemplary answers explicitly reason about this.

### 5. Writeup Must Explain *Why*, Not Just *What*
Not Yet: "I used Logistic Regression."  
Satisfactory: "I used Logistic Regression because it works well for classification."  
Exemplary: "I used Logistic Regression over Naive Bayes (which assumes feature independence — unlikely here since Members and Watching are correlated) and over KNN (which doesn't scale as well to this dataset size)."

---

## Question 1: Data Merging (15 pts — 2 autograded, 13 manual)

### What the Exemplary Code Did

```python
df_merged = pd.merge(anime, genres, on='MAL_ID', how='left')
rows = df_merged.shape[0]        # 50,239
columns = df_merged.shape[1]     # 36
statistic = 3885                  # Number of Action genre animes
```

**Key code choices:**
- Explicit `on='MAL_ID'` and `how='left'` — no ambiguity about join key or join type
- `shape[0]` / `shape[1]` for `rows` / `columns` — clean, not `.count()` or `len()`
- `statistic` is an integer count with a clear label — not a float average that needs context

### What the Exemplary Writeup Said

#### "What does a row represent?"
The exemplary answer said:

> "Each row represents one anime paired with one of its genres. Since an anime can have multiple genres, a single anime (identified by MAL_ID) can appear on multiple rows — one per genre. If an anime has no genre in the genres dataset, it still appears in df_merged with NaN in the Genre column because we used a left join."

**Why this is exemplary:**
- States the unit of analysis precisely (anime + one genre, not just "an anime")
- Explains the many-to-one relationship (multiple rows per anime)
- Explicitly mentions the NaN case for anime with no genres — satisfactory answers often omit this
- Connects directly to the left join behavior

#### "What does your code do?"
The exemplary answer said:

> "I performed a left join merging `anime` (the left table) with `genres` on MAL_ID. This preserves every row from anime and attaches matching genre rows. If an anime has 3 genres, it gets 3 rows in df_merged. Unmatched genres are discarded. The result has 50,239 rows and 36 columns."

**Why this is exemplary:**
- Explicitly states which table is "left" and why that matters (preserves all anime)
- Explains the multiplication effect on rows (3 genres → 3 rows)
- Confirms the discard behavior for unmatched genres
- States actual numbers from the output

#### "Describe your descriptive statistic"
The exemplary answer said:

> "I computed that there are 3,885 rows where Genre == 'Action', meaning approximately 3,885 Action-classified anime entries in the merged dataset — roughly 22% of all genre-labeled rows. This gives a sense of how Action is one of the more common genres."

**Why this is exemplary:**
- Identifies *what the number measures* (not just prints it)
- Contextualizes it as a percentage of the total
- Draws a brief interpretive conclusion ("one of the more common genres")
- Satisfactory answers computed a valid statistic but didn't interpret it in context

### Why Satisfactory Was Not Exemplary
The satisfactory answer used `mode()[0]` on Genre and reported the mean Avg Score of 4.704 — but the explanation didn't cleanly align with the code. The statistic was a float (4.704) which required context (mean of what? filtered to what?) that wasn't provided inline.

### Why Not Yet Failed
Not Yet correctly merged but then stored a statistic whose explanation didn't match the code. The writeup described a "mean of means" conceptually but the code computed a simple mean — showing the student didn't understand what their code actually produced.

---

## Question 2: Bootstrapping & Confidence Intervals (25 pts — 4 autograded, 21 manual)

### What the Exemplary Code Did

```python
# Extract genre subsets
action = df_merged[df_merged['Genre'] == 'Action']
drama  = df_merged[df_merged['Genre'] == 'Drama']

# Bootstrap — vectorized, no loops
action_samples = np.random.choice(action['Watching'], 
                                   size=(10000, len(action)), replace=True)
action_means   = np.average(action_samples, axis=1)
action_ci_l    = np.percentile(action_means, 2.5)
action_ci_r    = np.percentile(action_means, 97.5)
# (mirror for drama)
interval1 = (action_ci_l, action_ci_r)   # lower at index 0 — critical
interval2 = (drama_ci_l, drama_ci_r)

# Set operations
genre1_set = set(df_merged[df_merged['Genre'] == 'Action']['MAL_ID'])
genre2_set = set(df_merged[df_merged['Genre'] == 'Drama']['MAL_ID'])
left  = len(genre1_set - genre2_set)          # 3,233 only in Action
both  = len(genre1_set & genre2_set)          # 652 in both
right = len(genre2_set - genre1_set)          # 1,965 only in Drama
```

**Key code choices:**
- `np.random.choice(..., size=(10000, n))` — 2D array, 10,000 rows each of size n. No loop.
- `np.average(..., axis=1)` — mean of each resample row in one call
- `np.percentile(means, 2.5)` and `np.percentile(means, 97.5)` — correct 95% CI endpoints
- Lower bound at index 0 in the tuple — exactly as required
- Set operations use `MAL_ID`, not row index, to correctly identify unique anime

### What the Exemplary Writeup Said

#### "Explain your code"
> "I isolated the 'Watching' column for Action (n=3,885) and Drama (n=2,617) animes from df_merged. For each genre, I drew 10,000 bootstrap resamples (with replacement) of the same size as the original sample using a 2D np.random.choice call, then computed the mean of each resample. The 95% CI is the interval between the 2.5th and 97.5th percentiles of those 10,000 means."

**Why this is exemplary:**
- Describes the bootstrap procedure step by step
- Mentions sample sizes explicitly
- States 10,000 resamples (not just "many resamples")
- Explains the percentile method for the CI

#### "Why those genres and that feature?"
> "I chose Action and Drama because they are two of the largest genres (≥ 20 animes each, with 3,885 and 2,617 respectively) and represent stylistically opposite content categories. I chose 'Watching' (current active viewers) as the feature because it's a direct measure of current audience engagement — a genuinely interesting comparison across genre."

**Why this is exemplary:**
- Explicitly confirms ≥ 20 anime requirement is met with actual counts
- Provides a substantive reason for the genre comparison (stylistically opposite)
- Explains *why the feature is interesting* (not just "it's a column")

#### "State your hypotheses"
> "H₀: The mean number of currently-watching viewers is the same for Action and Drama anime.  
> Hₐ: The mean number of currently-watching viewers differs between Action and Drama anime."

**Why this is exemplary:**
- Hypotheses are testable with CIs (they compare means)
- Uses precise language ("mean number of currently-watching viewers") — not vague ("the values are different")
- Directly maps to the feature chosen

#### "Do the intervals overlap? What does this mean?"
> "The CIs are [4,097, 5,701] for Action and [3,316, 5,128] for Drama. These intervals overlap substantially. Therefore, we cannot reject H₀ at the 95% confidence level — the data is consistent with no significant difference in mean active viewers between the two genres.  
>
> However, this conclusion must account for the overlap in anime: 652 MAL_IDs appear in **both** genres (the `both` variable). These animes contribute observations to **both** bootstrap samples, meaning the samples are not independent. This violates a strict assumption of the two-sample CI comparison and means we should interpret the overlap conclusion cautiously — the true sampling distributions are not fully independent."

**Why this is exemplary:**
- States whether intervals overlap with actual numbers
- Connects overlap status directly to the hypothesis decision
- **Crucially addresses the `left`/`both`/`right` values** — this is the single biggest differentiator from satisfactory
- Explains *why* the overlap in animes matters statistically (non-independent samples)
- Doesn't overstate ("cannot reject H₀" not "there is no difference")

### Why Satisfactory Was Not Exemplary
The satisfactory answer noted whether CIs overlapped and stated the correct H₀/Hₐ, but did not discuss `left`/`both`/`right` at all. It skipped the independence issue entirely.

### Why Not Yet Failed
Not Yet used a for loop to bootstrap and used `Avg Score` without acknowledging it's a mean-of-means (biased). The writeup didn't connect the overlap discussion to the hypothesis.

---

## Question 3: Linear Regression (20 pts — 3 autograded, 17 manual)

### What the Exemplary Code Did

```python
col_encode = "Rating"
all_cols4  = ["Popularity", "Members", "Watching", col_encode]

# Filter to valid + ≤10 categories
q3_df = anime[anime['Rating'] != "Unknown"][["Avg Score"] + all_cols4].copy()

# OrdinalEncoder with explicit ordering
categories = [['G - All Ages', 'PG - Children', 'PG-13 - Teens 13 or older',
               'R - 17+ (violence & profanity)', 'R+ - Mild Nudity', 'Rx - Hentai']]
encoder = OrdinalEncoder(categories=categories)
q3_df["Rating_encoded"] = encoder.fit_transform(q3_df[['Rating']])

# Model 1: all features including encoded Rating
target = q3_df['Avg Score'].values
X1     = q3_df[["Popularity", "Members", "Watching", "Rating_encoded"]].values
model1 = LinearRegression().fit(X1, target)
mse_1  = mean_squared_error(target, model1.predict(X1))
r2_1   = r2_score(target, model1.predict(X1))

# Model 2: drop encoded Rating
X2     = q3_df[["Popularity", "Members", "Watching"]].values
model2 = LinearRegression().fit(X2, target)
mse_2  = mean_squared_error(target, model2.predict(X2))
r2_2   = r2_score(target, model2.predict(X2))
```

**Key code choices:**
- Filters out "Unknown" ratings to produce a clean categorical column with ≤ 10 valid categories
- Uses `OrdinalEncoder` with explicitly listed categories in the correct age-progression order
- Uses `col_encode` variable name throughout (not the hardcoded string `"Rating"`)
- Both models trained on the same filtered dataset — consistent comparison
- `copy()` after filtering — avoids SettingWithCopyWarning

### What the Exemplary Writeup Said

#### "Why that DataFrame?"
> "I used `anime` (not `df_merged`) because df_merged duplicates anime rows by genre. If I used df_merged, an anime with 5 genres would contribute 5 rows to my regression, artificially inflating n and biasing the model toward animes with many genre labels. Using `anime` keeps each anime as one observation."

**Why this is exemplary:**
- Directly names the duplication problem in df_merged
- Explains the concrete statistical consequence (bias, inflated n)
- Contrasts the two DataFrames to make the reasoning clear

#### "Why those columns?"
> - **Popularity**: Inversely related to rank — more popular anime tend to score higher.  
> - **Members**: Captures total audience size — large member counts signal prestige.  
> - **Watching**: Active engagement today signals the anime is still compelling.  
> - **Rating (col_encode)**: Content rating (G/PG/R) may correlate with genre maturity and audience expectations, which plausibly affect scores.

**Why this is exemplary:**
- One sentence of reasoning per column
- Reasoning is causal/plausible, not just "it seemed useful"
- All four columns are connected to the target (Avg Score)

#### "Why that encoder?"
> "I used `OrdinalEncoder` because the Rating categories have a natural age-progression ordering: G < PG < PG-13 < R < R+ < Rx. OneHotEncoder would create 6 binary columns and lose this ordering information. Since the progression is meaningful (stricter ratings correspond to more mature content which may affect scores differently), ordinal encoding preserves that signal."

**Why this is exemplary:**
- States the specific ordering used and justifies *why* that order is correct
- Explicitly contrasts with OneHotEncoder and explains why OHE would be worse here
- Ties the encoding choice to the feature's semantic meaning

#### "Model comparison"
> "Model 1 (with Rating encoded): MSE = 2.595, R² = 0.668  
> Model 2 (without Rating): MSE = 2.596, R² = 0.668  
>
> The models perform nearly identically. Rating adds essentially zero predictive power for Avg Score beyond what Popularity, Members, and Watching already capture. This suggests content rating is not a strong independent predictor of audience score — plausibly because viewership patterns (captured in the other columns) are downstream of rating anyway."

**Why this is exemplary:**
- Reports both metrics (MSE and R²) for both models — not just one
- Does not just say "similar" — quantifies *how* similar
- Draws a substantive conclusion: col_encode adds no marginal predictive power
- Offers an interpretation *why* (not just states the result)
- Does not overstate: "not a strong independent predictor" not "rating has no effect on scores"

### Why Satisfactory Was Not Exemplary
Satisfactory correctly built both models but the encoder choice rationale was vague ("OneHotEncoder because the categories don't have order") without contrasting with OrdinalEncoder. The model comparison mentioned "similar MSE" but didn't discuss what that means about the held-out column's predictive value.

### Why Not Yet Failed
Not Yet hardcoded category lists and didn't filter the data first, producing models with leaked/incorrect category encodings. The writeup didn't explain the encoding logic at all.

---

## Question 4: Classification (20 pts — 3 autograded, 17 manual)

This question has the most writing and the most complex rubric. It breaks into two written sections.

### What the Exemplary Code Did

```python
col_cat   = 'Type'
all_cols5 = [col_cat, 'Source', 'Members', 'Watching']

# Encode Source with OneHotEncoder (many unordered categories)
categorical = OneHotEncoder().fit_transform(df[['Source']].values).toarray()
numerical   = df[['Members', 'Watching']].values
X           = np.append(categorical, numerical, axis=1)
y           = df[col_cat].values

# Train/test split — 25% test, random_state=216
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=216)

# Scale after split (no data leakage)
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)      # transform only, not fit_transform

# Model 1: all features
model1 = LogisticRegression(max_iter=10000)
model1.fit(X_train, y_train)
accuracy_model1 = accuracy_score(y_test, model1.predict(X_test))

# Model 2: remove Source columns (all OHE columns for Source)
# Compute which column indices belong to Source
n_source_cols = categorical.shape[1]
X_train_2     = X_train[:, n_source_cols:]
X_test_2      = X_test[:,  n_source_cols:]
model2 = LogisticRegression(max_iter=10000)
model2.fit(X_train_2, y_train)
accuracy_model2 = accuracy_score(y_test, model2.predict(X_test_2))

# Confusion matrices
ConfusionMatrixDisplay.from_estimator(model1, X_test,   y_test)
ConfusionMatrixDisplay.from_estimator(model2, X_test_2, y_test)
```

**Key code choices:**
- `col_cat` is the target variable consistently used — not a separate hardcoded string
- `train_test_split` applied once, same split for both models (not re-split per model)
- `StandardScaler.fit_transform` on train, `.transform` only on test — no data leakage
- `max_iter=10000` for LogisticRegression — avoids convergence warnings
- Confusion matrices use test data only, not train data
- Model 2 correctly removes *all* columns associated with `Source` (OHE expands to multiple columns)

### Written Portion 4.1: Data and Model

#### "Why that DataFrame and how did you filter?"
> "I used `anime` to avoid the duplicate-row problem in df_merged (each genre adds a row per anime, distorting the feature distributions). The `Type` column has 7 unique values: TV, OVA, Movie, Special, ONA, Music, and Unknown. I dropped the Unknown category (which has few samples and isn't a genuine type), leaving 6 clean categories, all with ≥ 300 rows."

**Why this is exemplary:**
- Names the df_merged duplication problem explicitly
- Lists the actual categories and states why Unknown was dropped
- Confirms the ≤ 7 category and ≥ 300 row constraints are met

#### "Which classifier and why?"
> "I used Logistic Regression.  
>  
> vs. **Naive Bayes**: NB assumes all features are conditionally independent given the class. Members and Watching are likely correlated (popular anime have more current viewers), so this assumption is violated. LR makes no such independence assumption.  
>  
> vs. **KNN**: KNN requires computing distances to all training points at prediction time, which scales poorly with dataset size. LR produces a linear decision boundary that generalizes more efficiently to unseen data."

**Why this is exemplary:**
- Compares *all three* required model types (LR, NB, KNN)
- For each comparison, states a specific, technical reason — not just "LR is more accurate"
- Addresses dataset-specific reasoning (Members/Watching correlation, dataset size)

#### "Which columns and why?"
> - **Type** (col_cat / target): The class label — what we're predicting.  
> - **Source**: Original source material (Manga, Original, Game, etc.) likely predicts format — e.g., long-running manga adapt to TV series while short stories become Movies.  
> - **Members**: Total audience size; TV shows tend to have large audiences while specials/ONAs are more niche.  
> - **Watching**: Active current viewers; Movie/OVA viewership patterns differ from TV's episodic model.

**Why this is exemplary:**
- Explains the target column (col_cat) in addition to features
- Every feature has a plausible causal argument for predicting `Type`
- Arguments are specific to the target variable, not generic "this column seemed useful"

### Written Portion 4.2: Results

#### "Is there a clear ranking between models?"
> "There is no clear ranking. Model 1 (accuracy = 43.1%) and Model 2 (accuracy = 43.2%) perform virtually identically. The difference is 0.1 percentage point, which is within noise. This suggests that the Source column — the feature held out in Model 2 — adds no meaningful predictive power for classifying anime Type, at least not beyond what Members and Watching already capture."

**Why this is exemplary:**
- Reports actual accuracies (numbers, not just "similar")
- Quantifies the difference (0.1 pp)
- Concludes specifically about the *held-out column* (Source), not the model in general
- Notes the "no marginal value" interpretation correctly

#### "Compare 2+ values WITHIN one confusion matrix"
> "Looking at Model 1's confusion matrix:  
> - The model correctly predicts **TV** 672 times (the single largest true-positive count), unsurprisingly since TV is the most common Type in the dataset.  
> - It predicts **OVA** correctly 237 times but confuses 294 OVA instances as TV. OVAs and TV shows share overlapping features (similar Members ranges) which likely causes this bleed.  
> - The model almost never correctly identifies **Music** (0–2 correct predictions) because Music-type anime are extremely rare and underrepresented in training."

**Why this is exemplary:**
- Cites specific cell values (not just "the model does well on some classes")
- Interprets *why* the confusion happens (feature overlap between OVA and TV)
- Notes a class imbalance issue (Music) and its effect on model behavior
- Compares diagonal vs off-diagonal cells

#### "Compare 2+ values BETWEEN the two confusion matrices"
> "Comparing Model 1 and Model 2:  
> - Both models classify **TV** correctly at nearly the same rate (672 vs 674). Removing Source doesn't hurt TV classification, suggesting Members/Watching are sufficient for that class.  
> - Both models fail equally on **Music** (near-zero correct predictions). Source information provides no rescue here — the problem is sample imbalance, not missing features.  
> - **ONA** classification degrades slightly in Model 2 (43 → 38 correct). ONAs are newer online formats often originating from specific sources (game adaptations, web manga), so removing Source reduces the model's ability to identify them."

**Why this is exemplary:**
- Compares specific classes across both matrices (TV, Music, ONA)
- Draws different conclusions per class — not one sweeping statement
- Explains the ONA degradation mechanistically (ONAs tied to source type)
- Confirms the Music finding is about imbalance, not features — sophisticated reasoning

### Why Satisfactory Was Not Exemplary
Satisfactory compared model accuracies and plotted confusion matrices, but the within-matrix analysis only noted "TV has the most correct predictions" without explaining *why* misclassifications occur. The between-matrix analysis said "both models perform similarly" without citing specific cell changes.

### Why Not Yet Failed
Not Yet set `col_cat = 'Type'` but then predicted `'Rating'` as the target — the variable and the model were inconsistent. The writeup didn't compare all three classifier types and the confusion matrix analysis was missing entirely.

---

## Checklists for the Actual Practicum

### Every Question — Before Submitting
- [ ] Does my writeup mirror exactly what my code does? (Same columns, same method, same numbers)
- [ ] Did I use variable names (`col_encode`, `col_cat`) everywhere downstream — no raw string literals?
- [ ] Did I explain `df_merged` duplication if I used that dataset?
- [ ] Are all required output variables set (`rows`, `columns`, `statistic`, `n1`, `n2`, etc.)?
- [ ] Are there any for/while loops I can replace with a vectorized operation?

### Q1 Specific
- [ ] Merge explicitly uses `on='MAL_ID'` and `how='left'`
- [ ] Row representation explains: one row = one anime + one genre (with NaN for no-genre case)
- [ ] Statistic is described with its units and an interpretation, not just printed

### Q2 Specific
- [ ] Bootstrap is 10,000 resamples, no loop, 2D `np.random.choice`
- [ ] Interval stored as `(lower, upper)` — lower at index 0
- [ ] Hypotheses are testable with CIs (about means, not general "difference")
- [ ] Overlap discussion includes actual CI numbers
- [ ] Writeup addresses `left`/`both`/`right` and explains what non-zero `both` means for independence

### Q3 Specific
- [ ] Encoder choice explained by contrasting with the other option (OHE vs Ordinal)
- [ ] If OrdinalEncoder: categories listed in explicit correct order with justification
- [ ] If OHE: explicitly say why order doesn't matter
- [ ] Model comparison reports both MSE and R² for both models
- [ ] Conclusion about what model difference (or non-difference) implies about col_encode's predictive value

### Q4 Specific
- [ ] col_cat is the target variable in the model (not a different column)
- [ ] Same train/test split used for both models
- [ ] Scaler `.fit_transform` on train, `.transform` only on test
- [ ] Model choice writeup compares ALL THREE types (LR, NB, KNN)
- [ ] Confusion matrix within-analysis names specific cells + explains misclassification patterns
- [ ] Confusion matrix between-analysis names specific cells that change between models + explains why

---

## The Single Most Common Drop from Exemplary to Satisfactory

Based on the rubric patterns, the most frequent reason students fall from Exemplary to Satisfactory is:

> **Incomplete hypothesis/overlap analysis in Q2** — addressing whether CIs overlap but not discussing what the `left`/`both`/`right` values mean for the independence assumption.

And the most frequent reason students fall from Satisfactory to Not Yet is:

> **Code/writeup mismatch** — writing an explanation that doesn't match the actual code execution (wrong column names, wrong methods, wrong conclusions from the numbers printed).

Always re-read your writeup after writing it and verify every claim against the actual code output.
