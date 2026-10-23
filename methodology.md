# Methodology

## 1. Refined research question
On the ULB credit card dataset, does a non-linear ensemble model (Random Forest) detect
fraud better than a regularised linear baseline (Logistic Regression) when both are
trained on the same imbalanced data with class weighting, and is the comparison still
the same when judged by precision-recall metrics instead of accuracy?

This narrows the broad topic ("fraud detection with CNNs") to something testable with
a small experiment: the literature reports very high accuracy, so I want to know how
much of that holds up under metrics that respect the imbalance.

## 2. Dataset
- **Source:** "Credit Card Fraud Detection", ULB Machine Learning Group (Kaggle,
  `mlg-ulb/creditcardfraud`). European card transactions over two days in September 2013.
- **Size:** 284,807 transactions, 31 columns. Only 492 are fraud (0.172%).
- **Features:**
  - `V1`-`V28`: numeric components produced by PCA, anonymised for confidentiality.
  - `Time`: seconds elapsed since the first transaction in the data.
  - `Amount`: transaction amount.
- **Target:** `Class` (1 = fraud, 0 = legitimate).
- **Limitations:**
  - Extreme imbalance, so the test set will contain only about 100 frauds and metrics will be noisy.
  - The V-features are PCA outputs, so they cannot be interpreted or engineered on in a meaningful way.
  - Only two days of data from one region, so it says little about seasonal or long-term fraud drift.
  - `Time` is relative, not a clock time, so "hour of day" is only an approximation.
  - Some duplicate rows exist and PCA hides what the raw fields were.

## 3. Data cleaning plan
1. Check for missing values (none expected) and confirm dtypes.
2. Drop exact duplicate rows, keeping the first copy, and report how many were removed.
3. Check `Amount` for negatives or impossible values.
4. Split into train and test **before** any fitting step (stratified 80/20, fixed seed)
   so scaling and any resampling cannot leak test information.

## 4. Feature engineering plan
- `Amount_log = log1p(Amount)` to reduce the heavy right skew, then standardise using a
  scaler fitted on the training set only.
- `Hour = (Time // 3600) % 24`, encoded as `Hour_sin` and `Hour_cos` so that 23:00 and
  00:00 are close; then drop the raw `Time` and `Amount` columns.
- Keep `V1`-`V28` unchanged (already PCA-scaled).
- No resampling in the main comparison; imbalance is handled with class weights. This
  avoids the synthetic-data leakage risk noted in the literature review.

## 5. Models and why
| Model | Reason |
|---|---|
| Logistic Regression (`class_weight="balanced"`) | Simple, fast, interpretable baseline. If it is nearly as good as the complex model, extra complexity is not justified. |
| Random Forest (`class_weight="balanced_subsample"`) | Captures non-linear interactions between the V-features, is robust to scale, and is a standard strong performer on this dataset. The reviewed paper by Mizher & Nassif also uses it as a comparison point. |

The CNNs in the reviewed papers are not used here. The inputs are anonymised PCA values
with no spatial or sequential order, so convolutions have little structure to exploit,
and the papers themselves compare against classical models. A CNN would be the natural
next step with more time.

## 6. Evaluation metrics and why
- **Average precision (area under the precision-recall curve), primary metric.** It
  summarises the precision/recall trade-off across all thresholds and is not inflated by the
  huge number of true negatives, unlike accuracy or ROC-AUC.
- **Precision, recall and F1 for the fraud class** at the default 0.5 threshold. Recall
  tells us how many frauds are caught; precision tells us how many alerts are false alarms,
  which costs analyst time and annoys customers.
- **ROC-AUC** reported for comparison with the literature only.
- **Confusion matrix** to make the actual counts visible.
- **Accuracy is deliberately not used for decisions:** predicting "legitimate" for every
  transaction already scores about 99.8%.
