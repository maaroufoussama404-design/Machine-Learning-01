# Lab 1: Environment and First System Measurements

## 1. Goal
Train and benchmark Logistic Regression and Random Forest on the Breast Cancer Wisconsin dataset.

## 2. Method
- Dataset: sklearn Breast Cancer Wisconsin.
- Split: 70/30 stratified, random_state=42.
- LogisticRegression(max_iter=1000, random_state=42).
- RandomForestClassifier(n_estimators=100, random_state=42).
- Training: one warm-up and five measured runs; median reported.
- Inference: 100 single-sample predictions; median reported.
- Memory: peak process memory increase sampled using psutil in a background thread every 10 ms.
- Model serialization: joblib.

## 3. Results

| Model | Accuracy | Training median (s) | Inference median (ms) | Size (KB) | Peak memory increase (MB) |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9415 | 0.377662 | 0.084950 | 1.03 | 0.125 |
| Random Forest | 0.9357 | 0.173176 | 3.675400 | 284.07 | 0.102 |

### Deployment budget assessment

| Model | Cloud | Edge | Mobile | TinyML |
|---|---|---|---|---|
| Logistic Regression | YES* | YES | YES | YES |
| Random Forest | YES* | YES | YES | NO |

*Cloud YES assumes a host with at least 1 GB RAM, as specified in the manual. Other assessments use the measured process memory increase, latency and serialized model size. These are estimates, not hardware deployment tests.*

## 4. Conclusions
1. Logistic Regression achieved slightly better accuracy than Random Forest on this dataset.
2. Logistic Regression was much faster when making predictions, while Random Forest took more time.
3. Logistic Regression also had a much smaller model size, making it easier to store and more suitable for devices with limited resources.

## 5. Reproducibility
Random seeds were set to 42. See results/versions.txt for package versions and results/ for raw measurements.

IMPORTANT: Rewrite the three conclusions in your own words before submission. Review the measurements and make sure you can explain them.