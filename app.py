import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Spam Detector", page_icon="📩")
st.title("📩 Spam Email/SMS Detector")
st.caption("Beginner NLP demo: TF-IDF + Logistic Regression.")
data = pd.read_csv("data/messages.csv")
model = Pipeline([("tfidf", TfidfVectorizer()), ("classifier", LogisticRegression(max_iter=1000))])
model.fit(data["text"], data["label"])
message = st.text_area("Enter a message", placeholder="Paste an SMS or email message...")
if st.button("Classify message"):
    if not message.strip():
        st.warning("Enter a message first.")
    else:
        prediction = model.predict([message])[0]
        st.subheader(f"Prediction: {prediction.upper()}")
        st.write("This is a tiny educational dataset; predictions are not reliable for real-world filtering.")
with st.expander("View demo training data"):
    st.dataframe(data, use_container_width=True)
