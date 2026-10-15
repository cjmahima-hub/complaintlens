"""Step 23: Streamlit demo for complaintlens.

Type or paste a complaint narrative and see the predicted product category,
the probability for each class, and the words that pushed the prediction.
Nothing is stored or logged.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
VEC_PATH = ROOT / "models" / "tfidf_vectorizer.joblib"
CLF_PATH = ROOT / "models" / "logreg_classifier.joblib"

# Invented examples written for this demo, not real complaints
EXAMPLES = {
    "Mortgage (invented)": (
        "My mortgage servicer raised my escrow payment twice this year "
        "without a clear explanation, and I cannot get anyone on the phone "
        "to review the escrow analysis."
    ),
    "Debt collection (invented)": (
        "A collection agency keeps calling me about a medical bill I already "
        "paid. I sent proof of payment but they still call every day."
    ),
    "Credit reporting (invented)": (
        "Equifax still shows an account that I disputed three months ago. "
        "They said they would investigate but nothing was removed from my "
        "credit report."
    ),
}
NO_EXAMPLE = "(write your own)"


@st.cache_resource
def load_model():
    vec = joblib.load(VEC_PATH)
    clf = joblib.load(CLF_PATH)
    terms = np.array(vec.get_feature_names_out())
    return vec, clf, terms


def top_words(x, clf, terms, k, n=5):
    coo = x.tocoo()
    contrib = coo.data * clf.coef_[k][coo.col]
    order = np.argsort(contrib)[::-1][:n]
    return [
        (str(terms[coo.col[i]]), round(float(contrib[i]), 3))
        for i in order
        if contrib[i] > 0
    ]


def fill_example():
    choice = st.session_state["example"]
    if choice != NO_EXAMPLE:
        st.session_state["text"] = EXAMPLES[choice]


st.set_page_config(page_title="ComplaintLens", layout="centered")
st.title("ComplaintLens")
st.caption(
    "Predicts which of 9 product categories a consumer financial complaint "
    "belongs to. Nothing you type is stored."
)

if not (VEC_PATH.exists() and CLF_PATH.exists()):
    st.error("Model files not found. Run: python scripts\\top_words.py")
    st.stop()

vec, clf, terms = load_model()

st.selectbox(
    "Try an example",
    [NO_EXAMPLE] + list(EXAMPLES),
    key="example",
    on_change=fill_example,
)
text = st.text_area("Complaint text", key="text", height=200)

if st.button("Classify", type="primary"):
    if not text.strip():
        st.warning("Please type or paste a complaint first.")
    else:
        x = vec.transform([text])
        if x.nnz == 0:
            st.warning(
                "None of these words are in the model's vocabulary, so it "
                "cannot make a meaningful prediction."
            )
        else:
            probs = clf.predict_proba(x)[0]
            k = int(probs.argmax())
            st.subheader(str(clf.classes_[k]))
            st.write(f"Model probability: {probs[k] * 100:.1f}%")

            table = (
                pd.DataFrame(
                    {
                        "category": clf.classes_,
                        "probability (%)": probs * 100,
                    }
                )
                .sort_values("probability (%)", ascending=False)
                .reset_index(drop=True)
            )
            st.dataframe(
                table,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "probability (%)": st.column_config.ProgressColumn(
                        "probability (%)",
                        min_value=0.0,
                        max_value=100.0,
                        format="%.1f",
                    )
                },
            )

            words = top_words(x, clf, terms, k)
            if words:
                st.write("Words that pushed toward this category:")
                st.dataframe(
                    pd.DataFrame(words, columns=["word or phrase", "push"]),
                    hide_index=True,
                    use_container_width=True,
                )

with st.expander("About this model"):
    st.markdown(
        """
- **Model:** TF-IDF (1-2 word phrases) + Logistic Regression, trained on
  CFPB complaint narratives from 2017 to 2022.
- **Measured on 100,000 complaints from 2024:** macro-F1 0.7446, accuracy
  0.8498. Weaker classes: payday, title or personal loan (F1 0.55) and
  vehicle loan or lease (F1 0.61).
- **Probabilities are not calibrated.** We did not check whether 90% means
  right 90% of the time.
- **Company names carry a lot of the signal**, so the model may do worse on
  companies that were rare or absent in the training years.
- It knows only these 9 categories. It is a portfolio demo, not advice.
        """
    )