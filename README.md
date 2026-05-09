# Decision Tree Explorer

A small educational project: a **Streamlit** web app that explains **decision trees** for binary classification and demonstrates two standard split criteria—**Gini impurity** and **entropy**—on a **sample loan-approval dataset**. The app is suitable for teaching, demos, and deployment to **[Render](https://render.com)** from **GitHub**.

**Source repository:** [github.com/siddhartha4u2c/Decision_tree](https://github.com/siddhartha4u2c/Decision_tree)

---

## Push this project to GitHub (first time)

Use these commands from the project root (the folder that contains `app.py`). The remote below matches [siddhartha4u2c/Decision_tree](https://github.com/siddhartha4u2c/Decision_tree).

```bash
git init
git branch -M main
git remote add origin https://github.com/siddhartha4u2c/Decision_tree.git
```

If `origin` already exists, point it at this repo instead:

```bash
git remote set-url origin https://github.com/siddhartha4u2c/Decision_tree.git
```

Then commit and push:

```bash
git add .
git commit -m "Initial commit: Streamlit decision tree demo (Gini + entropy)"
git push -u origin main
```

**Authentication:** GitHub no longer accepts account passwords for HTTPS Git. Sign in with a **[personal access token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens)** (treat it like a password when Git prompts you), use **[SSH](https://docs.github.com/en/authentication/connecting-to-github-with-ssh)** (`git@github.com:siddhartha4u2c/Decision_tree.git` as the remote URL), or run **`gh auth login`** if you use the [GitHub CLI](https://cli.github.com/).

On Windows you can also run **`push-to-github.ps1`** from this folder (same steps as above).

---

## What this project demonstrates

### Decision trees (classification)

A decision tree predicts a class by following a path from the **root** to a **leaf**. At each **internal node**, a rule compares one feature to a threshold (for example, “is `credit_score ≤ 650`?”). Leaves output a predicted label (here: loan **not approved** vs **approved**).

Training chooses splits that make the child nodes **more pure** (more homogeneous with respect to the target class). You control complexity with hyperparameters such as **maximum depth** and **minimum samples per leaf** to limit **overfitting**.

### Gini vs entropy (both used in this app)

This repository trains **two** trees on the **same** training data and hyperparameters:

| Criterion | scikit-learn value | Idea |
|-----------|---------------------|------|
| **Gini impurity** | `criterion='gini'` | Measures how often a randomly chosen label would be mislabeled if we sampled according to class proportions in the node. For binary classes with proportions $p_0, p_1$: $G = 1 - p_0^2 - p_1^2$. |
| **Entropy** | `criterion='entropy'` | Based on Shannon entropy $H = -\sum_k p_k \log_2(p_k)$. Splits maximize **information gain** (reduction in entropy). |

In practice, trees from Gini and entropy often look similar; side-by-side plots and metrics in the app make differences easy to see on this dataset.

### Feature importances

After fitting, scikit-learn reports **feature importances** derived from total impurity decrease contributed by each feature. These are **heuristic**, not causal measures of real-world impact.

---

## Sample dataset

File: **`data/loan_sample.csv`**

| Column | Description |
|--------|-------------|
| `age` | Applicant age |
| `annual_income` | Annual income (fictional currency) |
| `credit_score` | Credit score–style number |
| `debt_ratio` | Debt-to-income–style ratio (0–1 scale in the file) |
| `approved` | Target: `0` = not approved, `1` = approved |

The data is **synthetic / illustrative** for learning—not real financial or personal data.

---

## App features (Streamlit)

- Short **concept** text: trees, splitting, regularization, importances.
- **Expandable** section with **Gini** and **entropy** definitions and formulas.
- **Data preview** from the CSV.
- **Controls**: train/test fraction, random seed, max depth, min samples per leaf.
- **Two models** trained every run: Gini and entropy.
- **Metrics**: test accuracy, fitted depth, and leaf count for each criterion.
- **Tabs**:
  - **Tree diagrams** — both trees plotted with `sklearn.tree.plot_tree`.
  - **Feature importance** — bar charts for Gini and entropy models.
  - **Confusion matrices** — test-set matrices and classification reports.
  - **Try a prediction** — numeric inputs; predicted class and probabilities for **both** trees.

Main entrypoint: **`app.py`**.

---

## Local setup

**Requirements:** Python **3.10+** (3.11 recommended).

```bash
git clone https://github.com/siddhartha4u2c/Decision_tree.git

cd Decision_tree

python -m venv .venv


# Windows

.venv\Scripts\activate


# macOS / Linux

source .venv/bin/activate



pip install -r requirements.txt

streamlit run app.py

```

Open the URL shown in the terminal (usually `http://localhost:8501`).

---

## Deploy on Render via GitHub

### Prerequisites

- Code pushed to a **GitHub** repository (this folder as the repo root, or adjust paths).
- A [Render](https://render.com) account.

### Option A — Blueprint (`render.yaml`)

1. In Render: **New** → **Blueprint**.
2. Connect the GitHub repo and select **`render.yaml`**.
3. Apply the blueprint. Render will create a **Web Service** with the configured build and start commands.

The included **`render.yaml`** installs dependencies and starts Streamlit bound to Render’s port and host:

- **Build:** `pip install -r requirements.txt`
- **Start:** `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true`

`--server.address=0.0.0.0` is required so the service accepts external connections. `$PORT` is set by Render.

### Option B — Manual Web Service

1. **New** → **Web Service** → connect the repo.
2. **Runtime:** Python (or Native Python, depending on Render’s UI).
3. **Build command:** `pip install -r requirements.txt`
4. **Start command:**  
   `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true`
5. Deploy. Optionally set **Python version** in the service settings (e.g. **3.11**) if you remove or adjust environment variables in `render.yaml`.

### Troubleshooting

- **Blank page / connection refused:** Confirm the start command uses `$PORT` and `0.0.0.0`.
- **Missing data:** Ensure **`data/loan_sample.csv`** is committed; the app resolves the CSV relative to `app.py`.
- **Slow cold starts (free tier):** Normal on Render’s free plan; first load after idle may take longer.

---

## Project layout

```text

Decision_tree/

├── app.py                 # Streamlit application

├── data/

│   └── loan_sample.csv    # Sample classification dataset

├── requirements.txt       # Python dependencies

├── render.yaml            # Render Blueprint (optional)

├── .gitignore

├── push-to-github.ps1    # Optional: first-time push helper (Windows)

└── README.md

```

---

## Dependencies

Listed in **`requirements.txt`**:

- **streamlit** — web UI
- **scikit-learn** — `DecisionTreeClassifier`, metrics, `plot_tree`
- **pandas** — load CSV
- **numpy** — prediction arrays
- **matplotlib** — tree figures for Streamlit

---

## License and disclaimer

Use this project for **education and demonstration** only. The dataset is fictional; do not use it as financial advice or for production credit decisions.

If you extend the app (cross-validation, pruning, random forests, SHAP, etc.), keep the same deployment pattern: bind Streamlit to `$PORT` and `0.0.0.0` on Render.
