# 📰 Fake News Detection using Machine Learning

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kushalsh4h/fake-news-detection/blob/main/Fake_News_Detection_using_machine_learning.ipynb)
![Python](https://img.shields.io/badge/Python-3.x-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Task](https://img.shields.io/badge/Task-Binary%20Text%20Classification-green)

A machine learning project that reads the text of a news article and predicts whether it is **Fake** or **Real**. Four classifiers are trained on TF-IDF features and then vote on any new article you paste in.

📄 **[Read the full study guide (PDF)](Fake_News_Detection_Study_Guide.pdf)**: a plain-English walkthrough of every notebook cell, the maths behind TF-IDF, results, limitations and likely viva questions.

---

## 👥 Team

| Name | Roll No. |
|---|---|
| Kushal Shah | C064 |
| Varun Pitale | C080 |
| Kanishk Modi | C119 |

> Based on the open-source notebook by [kapilsinghnegi](https://github.com/kapilsinghnegi/Fake-News-Detection). All figures marked "our re-run" in the study guide were recomputed by us on the same dataset.

---

## 🎯 Overview

This is **supervised learning** (every training article comes with a label) and, more specifically, **binary text classification**.

> Load labelled articles → clean the text → turn words into numbers → train four classifiers on 75% of the data → measure them on the other 25% → let them vote on any new article.

```
New article → wordopt() → TF-IDF vectorizer → 4 trained models → Verdict
 (pasted)    (cleaning)    (numbers)            (each votes)      Fake / Not fake
```

---

## 📊 Dataset

Two CSV files, `Fake.csv` and `True.csv`, which appear to be the widely used **ISOT Fake News** dataset: English news from roughly 2015 to 2018, mostly US politics and world affairs.

| Column | Meaning | Used for training? |
|---|---|---|
| `title` | Headline | No (dropped) |
| `text` | Full article body | **Yes**, the only input |
| `subject` | Topic label | No (dropped, it would leak the answer) |
| `date` | Publication date | No (dropped) |
| `class` | Label added by the notebook: `0` = Fake, `1` = Real | **Yes**, the target |

- 23,481 fake and 21,417 real articles in the files
- 10 rows per file are held back for manual testing, leaving **44,878** articles
- Split: **33,658 train / 11,220 test** (75% / 25%)

> ⚠️ The datasets are **not included** in this repo. See [Running the project](#-running-the-project) for how to get them.

---

## 🔄 Pipeline

1. **Add labels and merge**: Fake = 0, Real = 1
2. **Drop columns and shuffle**: keep only `text` and `class`
3. **Clean the text** with `wordopt()`: lowercase, remove `[brackets]`, links, punctuation and words containing digits
4. **Split** 75% train / 25% test
5. **TF-IDF vectorizer**: vocabulary learned from training text only (about 95,000 features), then applied to the test set
6. **Train four models** and score them on the test set

The vectorizer is fitted on training data only, so the test set stays a fair "unseen data" check.

---

## 🤖 Models and results

| Model | Accuracy |
|---|---|
| Logistic Regression | 98.78% |
| Decision Tree | **99.53%** |
| Gradient Boosting | 99.45% |
| Random Forest | 98.70% |

Precision, recall and F1 are about 0.99 for each model. The split is random and unseeded, so numbers shift slightly between runs and the ranking should not be over-interpreted. Logistic Regression is the fastest and easiest to explain.

**Confusion matrix** (Logistic Regression, our re-run with a fixed split, 98.63% accuracy):

| | Predicted Fake | Predicted Real |
|---|---|---|
| **Actually Fake** | 5,818 | 82 |
| **Actually Real** | 72 | 5,248 |

---

## 🔍 What does the 99% really mean?

Scores near 99% look great, but they come with caveats:

- **The source is a giveaway.** 99.8% of real articles mention "Reuters" versus 1.4% of fake ones. The model partly learns wire-service style and source markers (`reuters`, `said`, weekday names) versus blog-style wording (`via`, `read`, `image`, `featured`).
- **It is not only Reuters.** Removing the word and retraining dropped Logistic Regression only from 98.63% to 98.12%, so many other style clues remain.
- **Narrow data.** All articles come from a few sources about US politics around 2015 to 2018. 99% describes performance on *more articles like these*, not on news from any outlet.
- **Not a fact-checker.** The models compare writing style and vocabulary. They do not know whether any claim is true.

---

## 🚀 Running the project

### Option 1: Google Colab (easiest)

Click the **Open in Colab** badge at the top, then run:

```python
!git clone https://github.com/kapilsinghnegi/Fake-News-Detection.git
%cd /content/Fake-News-Detection
!unzip -o Datasets/datasets.zip -d Datasets
!find Datasets -name "*.csv" -exec cp {} . \;
```

Then use **Runtime → Run all**. The Decision Tree and Gradient Boosting cells are the slowest, so allow several minutes.

### Option 2: Locally

```bash
git clone https://github.com/kushalsh4h/fake-news-detection.git
cd fake-news-detection
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
# place Fake.csv and True.csv in this folder
jupyter notebook Fake_News_Detection_using_machine_learning.ipynb
```

### Try it on a new article

The last cells of the notebook define `manual_testing()`. Run the final cell, paste an article into the text box, and all four models print their verdict.

---

## 🛠️ Known issues and fixes

| Where | Issue | Fix |
|---|---|---|
| Decision Tree report | Prints `classification_report(y_test, pred_lr)`, so it shows Logistic Regression results | Use `pred_dt` |
| `wordopt()` | `re.sub('<.*?>+', b'', text)` uses a bytes replacement with a string pattern | Use `''` instead of `b''` |
| `train_test_split` | No `random_state`, so results change each run | Add `random_state=42` (and optionally `stratify=y`) |
| Imports | `matplotlib`, `numpy`, `seaborn`, `accuracy_score` are never used | Remove them or add charts |
| Hold-out rows | The 20 held-back articles are never fed to the models | Use them as an extra sanity check |
| Data | Source markers such as "Reuters" stay in the text | Strip them before cleaning |

---

## 🔮 Future improvements

- Remove source markers and train on several datasets from different outlets and years
- Use n-grams, sentence embeddings, or a transformer such as BERT to capture context
- Fix the random seed and use cross-validation for stable numbers
- Serve the best model through a small web app with a probability score instead of a hard label

---

## 📁 Repository structure

```
fake-news-detection/
├── Fake_News_Detection_using_machine_learning.ipynb   # main notebook
├── Fake_News_Detection_Study_Guide.pdf                # full study guide
└── README.md
```

---

## 🙏 Acknowledgements

- Original notebook and project by [kapilsinghnegi](https://github.com/kapilsinghnegi/Fake-News-Detection)
- Dataset: ISOT Fake News dataset (Fake.csv / True.csv)
- Built with [pandas](https://pandas.pydata.org/), [scikit-learn](https://scikit-learn.org/), and Google Colab
