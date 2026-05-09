"""
Streamlit demo: decision trees for binary classification on a small loan-approval sample.
Trains and compares trees using Gini impurity and entropy (information-theoretic) criteria.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

DATA_PATH = Path(__file__).resolve().parent / "data" / "loan_sample.csv"

FEATURE_NAMES = ["age", "annual_income", "credit_score", "debt_ratio"]
TARGET = "approved"


@st.cache_data
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing dataset at {DATA_PATH}")
    return pd.read_csv(DATA_PATH)


def fit_tree(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    criterion: str,
    max_depth: int,
    min_samples_leaf: int,
    random_state: int,
) -> DecisionTreeClassifier:
    clf = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    clf.fit(X_train, y_train)
    return clf


def plot_tree_figure(clf: DecisionTreeClassifier) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 6.5))
    plot_tree(
        clf,
        feature_names=FEATURE_NAMES,
        class_names=["not approved", "approved"],
        filled=True,
        rounded=True,
        fontsize=8,
        ax=ax,
    )
    return fig


def main() -> None:
    st.set_page_config(
        page_title="Decision Tree Explorer",
        page_icon="🌳",
        layout="wide",
    )

    st.title("Decision trees: concepts and interactive demo")
    st.caption(
        "Binary classification on a fictional loan-approval sample (CSV in `data/loan_sample.csv`). "
        "Each run trains **two** trees: **Gini** and **entropy**."
    )

    with st.expander("What is a decision tree?", expanded=False):
        st.markdown(
            """
            A **decision tree** predicts a label by asking a sequence of yes/no questions about
            the features. Each **internal node** is one rule (for example, “is `credit_score ≤ 650`?”).
            Each **leaf** is a predicted class (here: **not approved** vs **approved**).

            **Splitting:** At each step the algorithm picks the feature and threshold that best
            separate the classes. scikit-learn’s `DecisionTreeClassifier` scores splits using either
            **Gini impurity** or **entropy** (see the next expander). Lower impurity means more
            homogeneous child nodes.

            **Depth and regularization:** A deeper tree can fit training noise (**overfitting**).
            Limits like **max depth** and **min samples per leaf** keep the tree simpler and often
            generalize better on new data.

            **Feature importance:** After training, the model estimates how much each feature
            contributed to reducing impurity across all splits (a useful summary, not a causal story).
            """
        )

    with st.expander("Gini impurity vs entropy (split criteria)", expanded=True):
        st.markdown(
            r"""
            Both criteria measure **node impurity** for classification. Splits are chosen to
            **reduce** weighted impurity in the children (equivalently, maximize **information gain**
            when using entropy).

            **Gini impurity** (`criterion='gini'`)

            For a node with class proportions \(p_k\) (here \(k \in \{0,1\}\)):

            \[
            G = 1 - \sum_k p_k^2
            \]

            Ranges from **0** (one class only) to **0.5** for binary data with a 50/50 mix.
            Fast to compute; often produces trees similar to entropy.

            **Entropy** (`criterion='entropy'`)

            \[
            H = - \sum_k p_k \log_2(p_k)
            \]

            (with \(0 \log 0 := 0\)). Also **0** for a pure node; highest for a balanced mix.
            Tied to **information gain**: gain = parent entropy minus weighted child entropy.

            **Practical note:** On many datasets, Gini and entropy trees differ only slightly.
            This app trains **both** with the same hyperparameters so you can compare structure,
            test accuracy, and importances side by side.
            """
        )

    df = load_data()

    col_data, col_controls = st.columns((1.1, 1.0), gap="large")

    with col_data:
        st.subheader("Sample dataset")
        st.markdown(
            f"**{len(df)}** rows · Target column: **`{TARGET}`** (0 = not approved, 1 = approved)."
        )
        st.dataframe(df.head(12), use_container_width=True)

    with col_controls:
        st.subheader("Train your trees")
        test_size = st.slider("Test set fraction", 0.1, 0.4, 0.25, 0.05)
        random_state = st.number_input("Random seed", 0, 9999, 42, 1)
        max_depth = st.slider("Max depth", 1, 10, 3)
        min_samples_leaf = st.slider("Min samples per leaf", 1, 25, 5)

    X = df[FEATURE_NAMES]
    y = df[TARGET]

    rs = int(random_state)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=rs,
        stratify=y,
    )

    clf_gini = fit_tree(
        X_train,
        y_train,
        criterion="gini",
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=rs,
    )
    clf_entropy = fit_tree(
        X_train,
        y_train,
        criterion="entropy",
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=rs,
    )

    pred_g = clf_gini.predict(X_test)
    pred_e = clf_entropy.predict(X_test)
    acc_g = accuracy_score(y_test, pred_g)
    acc_e = accuracy_score(y_test, pred_e)

    st.subheader("Gini vs entropy — test-set summary")
    g1, g2, g3, e1, e2, e3 = st.columns(6)
    g1.metric("Gini — accuracy", f"{acc_g:.3f}")
    g2.metric("Gini — depth", clf_gini.get_depth())
    g3.metric("Gini — leaves", clf_gini.get_n_leaves())
    e1.metric("Entropy — accuracy", f"{acc_e:.3f}")
    e2.metric("Entropy — depth", clf_entropy.get_depth())
    e3.metric("Entropy — leaves", clf_entropy.get_n_leaves())

    tab_tree, tab_importance, tab_matrix, tab_predict = st.tabs(
        ["Tree diagrams", "Feature importance", "Confusion matrices", "Try a prediction"]
    )

    with tab_tree:
        st.markdown(
            "Rectangles are **decision nodes**; rounded boxes are **leaves** with "
            "`value = [not approved, approved]` counts on the training split."
        )
        c_left, c_right = st.columns(2, gap="medium")
        with c_left:
            st.markdown("**Criterion: Gini**")
            fig_g = plot_tree_figure(clf_gini)
            st.pyplot(fig_g)
            plt.close(fig_g)
        with c_right:
            st.markdown("**Criterion: Entropy**")
            fig_e = plot_tree_figure(clf_entropy)
            st.pyplot(fig_e)
            plt.close(fig_e)

    with tab_importance:
        i1, i2 = st.columns(2, gap="medium")
        with i1:
            st.markdown("**Gini — feature importances**")
            imp_g = pd.Series(clf_gini.feature_importances_, index=FEATURE_NAMES).sort_values(
                ascending=True
            )
            st.bar_chart(imp_g)
        with i2:
            st.markdown("**Entropy — feature importances**")
            imp_e = pd.Series(clf_entropy.feature_importances_, index=FEATURE_NAMES).sort_values(
                ascending=True
            )
            st.bar_chart(imp_e)

    with tab_matrix:
        m_left, m_right = st.columns(2, gap="medium")
        labels = [0, 1]
        idx = ["Actual 0 (not approved)", "Actual 1 (approved)"]
        cols = ["Pred 0", "Pred 1"]

        with m_left:
            st.markdown("**Gini — confusion matrix (test)**")
            cm_g = confusion_matrix(y_test, pred_g, labels=labels)
            st.dataframe(
                pd.DataFrame(cm_g, index=idx, columns=cols),
                use_container_width=True,
            )
            with st.expander("Gini — classification report"):
                st.text(
                    classification_report(
                        y_test, pred_g, target_names=["not approved", "approved"]
                    )
                )

        with m_right:
            st.markdown("**Entropy — confusion matrix (test)**")
            cm_e = confusion_matrix(y_test, pred_e, labels=labels)
            st.dataframe(
                pd.DataFrame(cm_e, index=idx, columns=cols),
                use_container_width=True,
            )
            with st.expander("Entropy — classification report"):
                st.text(
                    classification_report(
                        y_test, pred_e, target_names=["not approved", "approved"]
                    )
                )

    with tab_predict:
        st.markdown("Adjust inputs; both trees classify the same feature vector.")
        c1, c2, c3, c4 = st.columns(4)
        age = c1.number_input("Age", 18, 80, 35)
        income = c2.number_input("Annual income", 20000, 120000, 55000, 1000)
        credit = c3.number_input("Credit score", 500, 850, 680)
        debt = c4.number_input("Debt ratio", 0.15, 0.60, 0.30, 0.01)

        sample = np.array([[age, income, credit, debt]])
        pg = clf_gini.predict_proba(sample)[0]
        pe = clf_entropy.predict_proba(sample)[0]
        pred_g_i = int(clf_gini.predict(sample)[0])
        pred_e_i = int(clf_entropy.predict(sample)[0])

        p1, p2 = st.columns(2)
        with p1:
            st.info(
                f"**Gini tree** — class: "
                f"{'approved (1)' if pred_g_i == 1 else 'not approved (0)'}  \n"
                f"P(not approved) = {pg[0]:.2%}, P(approved) = {pg[1]:.2%}"
            )
        with p2:
            st.info(
                f"**Entropy tree** — class: "
                f"{'approved (1)' if pred_e_i == 1 else 'not approved (0)'}  \n"
                f"P(not approved) = {pe[0]:.2%}, P(approved) = {pe[1]:.2%}"
            )

    st.divider()
    st.caption("Deployment: see **README.md** (Render + GitHub).")


if __name__ == "__main__":
    main()
