
import streamlit as st
import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# =====================================
# 1. PAGE CONFIGURATION
# =====================================

st.set_page_config(
    page_title="Smart Email Spam Detection",
    page_icon="📧",
    layout="wide"
)

# =====================================
# 2. LOAD DATASET
# =====================================

data = pd.read_csv(
    "dataset/SMSSpamCollection",
    sep="\t",
    names=["label", "message"]
)

data["label"] = data["label"].map({
    "ham": 0,
    "spam": 1
})

X = data["message"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =====================================
# 3. CREATE MACHINE LEARNING PIPELINE
# =====================================

def create_model():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english"
        )),
        ("classifier", MultinomialNB())
    ])

# =====================================
# 4. SESSION STATE
# =====================================

if "model" not in st.session_state:
    if os.path.exists("model/spam_classifier.pkl"):
        st.session_state.model = joblib.load(
            "model/spam_classifier.pkl"
        )
    else:
        st.session_state.model = None

if "trained_this_session" not in st.session_state:
    st.session_state.trained_this_session = False

# =====================================
# 5. LEFT SIDE NAVIGATION
# =====================================

st.sidebar.title("📧 Project Menu")
st.sidebar.write("Smart Email Spam Detection")

st.sidebar.markdown("### MODULE 1")
st.sidebar.caption("Email Data Preprocessing")

st.sidebar.markdown("### MODULE 2")
st.sidebar.caption("Spam Classification and Evaluation")

page = st.sidebar.radio(
    "Select Implementation",
    [
        "1. Dataset Collection",
        "2. Text Preprocessing",
        "3. TF-IDF Feature Extraction",
        "4. Naive Bayes Algorithm",
        "5. Model Training",
        "6. Model Evaluation",
        "7. Live Spam Detection"
    ]
)

st.sidebar.divider()
st.sidebar.write("**Technologies Used**")
st.sidebar.write("Python")
st.sidebar.write("Pandas")
st.sidebar.write("Scikit-learn")
st.sidebar.write("Streamlit")

# =====================================
# 6. MAIN HEADING
# =====================================

st.title("📧 Smart Email Spam Detection")
st.write(
    "Spam Classification Using TF-IDF and Naive Bayes"
)
st.divider()

# =====================================
# PAGE 1: DATASET COLLECTION
# =====================================

if page == "1. Dataset Collection":

    st.header("Module 1: Dataset Collection")

    st.write(
        "We collected a labelled dataset containing "
        "Spam and Legitimate messages."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Messages", len(data))
    col2.metric("Legitimate Messages", (y == 0).sum())
    col3.metric("Spam Messages", (y == 1).sum())

    st.subheader("Dataset Sample")

    sample = data.head(20).copy()

    sample["label"] = sample["label"].map({
        0: "Legitimate",
        1: "Spam"
    })

    st.dataframe(sample, use_container_width=True)

    st.success("Dataset loaded successfully!")

# =====================================
# PAGE 2: TEXT PREPROCESSING
# =====================================

elif page == "2. Text Preprocessing":

    st.header("Module 1: Text Preprocessing")

    st.write(
        "Text preprocessing prepares messages "
        "before machine learning."
    )

    message = st.selectbox(
        "Choose a sample message",
        data["message"].head(50).tolist()
    )

    st.subheader("Original Message")
    st.code(message)

    # Use the same analyzer settings as the model
    analyzer = TfidfVectorizer(
        lowercase=True,
        stop_words="english"
    ).build_analyzer()

    words = analyzer(message)

    st.subheader("Processed Words")

    st.write(words)

    st.info(
        "The TF-IDF text analyzer converts text to lowercase, "
        "splits it into words, ignores punctuation and removes "
        "English stop words."
    )

    st.subheader("Preprocessing Output")

    st.write("Original message:", message)
    st.write("Number of processed words:", len(words))

# =====================================
# PAGE 3: TF-IDF FEATURE EXTRACTION
# =====================================

elif page == "3. TF-IDF Feature Extraction":

    st.header("Module 1: TF-IDF Feature Extraction")

    st.write(
        "TF-IDF converts text messages into numerical "
        "features that the machine learning model can understand."
    )

    if st.session_state.model is None:

        st.warning("Please train the model first.")

    else:

        message = st.selectbox(
            "Select a message",
            data["message"].head(50).tolist(),
            key="tfidf_message"
        )

        tfidf = st.session_state.model.named_steps["tfidf"]

        transformed = tfidf.transform([message])

        feature_names = tfidf.get_feature_names_out()

        feature_table = pd.DataFrame({
            "Word": feature_names,
            "TF-IDF Value": transformed.toarray()[0]
        })

        feature_table = feature_table[
            feature_table["TF-IDF Value"] > 0
        ].sort_values(
            by="TF-IDF Value",
            ascending=False
        )

        st.metric(
            "Total Vocabulary Features",
            len(feature_names)
        )

        st.subheader("Numerical Feature Output")

        st.dataframe(
            feature_table,
            use_container_width=True
        )

        st.success("Text successfully converted into numerical features!")

# =====================================
# PAGE 4: NAIVE BAYES ALGORITHM
# =====================================

elif page == "4. Naive Bayes Algorithm":

    st.header("Module 2: Naive Bayes Algorithm")

    st.write(
        "Multinomial Naive Bayes is the machine learning "
        "algorithm used to classify messages."
    )

    st.subheader("How it works")

    st.markdown("""
    1. Receives numerical features from TF-IDF.
    2. Learns patterns from labelled messages.
    3. Calculates the probability of each class.
    4. Predicts Spam or Legitimate.
    """)

    st.info("Algorithm: Multinomial Naive Bayes")

    if st.session_state.model is not None:

        nb = st.session_state.model.named_steps["classifier"]

        st.write("Classes learned by the model:")

        st.write({
            "0": "Legitimate",
            "1": "Spam"
        })

        st.write("Learned class probabilities:")

        st.write({
            "Legitimate": round(float(nb.class_log_prior_[0]), 4),
            "Spam": round(float(nb.class_log_prior_[1]), 4)
        })

    else:
        st.warning("Train the model to view its learned information.")

# =====================================
# PAGE 5: MODEL TRAINING
# =====================================

elif page == "5. Model Training":

    st.header("Module 2: Model Training")

    st.write(
        "The dataset is divided into 80% training data "
        "and 20% testing data."
    )

    col1, col2 = st.columns(2)

    col1.metric("Training Records", len(X_train))
    col2.metric("Testing Records", len(X_test))

    st.subheader("Training Process")

    st.markdown("""
    1. Load the labelled dataset.
    2. Split the dataset into training and testing data.
    3. Convert text into TF-IDF features.
    4. Train the Multinomial Naive Bayes algorithm.
    5. Save the trained model.
    """)

    if st.button("🚀 Train Model", type="primary"):

        with st.spinner("Training the Naive Bayes model..."):

            new_model = create_model()

            new_model.fit(X_train, y_train)

            os.makedirs("model", exist_ok=True)

            joblib.dump(
                new_model,
                "model/spam_classifier.pkl"
            )

            st.session_state.model = new_model
            st.session_state.trained_this_session = True

        st.success("Model training completed successfully!")

    if st.session_state.model is not None:

        if st.session_state.trained_this_session:
            st.success("Model trained during this session.")
        else:
            st.info("Previously trained model loaded successfully.")

        st.write("Saved model location:")
        st.code("model/spam_classifier.pkl")

    else:
        st.warning("Click Train Model to start training.")

# =====================================
# PAGE 6: MODEL EVALUATION
# =====================================

elif page == "6. Model Evaluation":

    st.header("Module 2: Model Evaluation")

    st.write(
        "The trained model is tested using unseen testing data."
    )

    if st.session_state.model is None:

        st.warning("Please train the model first.")

    else:

        model = st.session_state.model

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(
            y_test, predictions, zero_division=0
        )
        recall = recall_score(
            y_test, predictions, zero_division=0
        )
        f1 = f1_score(
            y_test, predictions, zero_division=0
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Accuracy", f"{accuracy * 100:.2f}%")
        c2.metric("Precision", f"{precision * 100:.2f}%")
        c3.metric("Recall", f"{recall * 100:.2f}%")
        c4.metric("F1 Score", f"{f1 * 100:.2f}%")

        st.subheader("Classification Report")

        report = classification_report(
            y_test,
            predictions,
            target_names=["Legitimate", "Spam"],
            output_dict=True,
            zero_division=0
        )

        st.dataframe(
            pd.DataFrame(report).transpose().round(3),
            use_container_width=True
        )

        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y_test, predictions)

        cm_table = pd.DataFrame(
            cm,
            index=["Actual Legitimate", "Actual Spam"],
            columns=["Predicted Legitimate", "Predicted Spam"]
        )

        st.dataframe(cm_table, use_container_width=True)

        st.success("Model evaluation completed!")

# =====================================
# PAGE 7: LIVE SPAM DETECTION
# =====================================

elif page == "7. Live Spam Detection":

    st.header("Live Spam Detection")

    st.write(
        "Enter a message and check whether it is Spam or Legitimate."
    )

    message = st.text_area(
        "Enter your message:",
        height=150,
        placeholder="Type your message here..."
    )

    if st.button("Check Message", type="primary"):

        if message.strip() == "":

            st.warning("Please enter a message.")

        elif st.session_state.model is None:

            st.warning("Please train the model first.")

        else:

            model = st.session_state.model

            prediction = model.predict([message])[0]

            if prediction == 1:

                st.error("🚨 Result: This message is SPAM!")

            else:

                st.success("✅ Result: This message is LEGITIMATE!")

            st.subheader("TF-IDF Features")

            tfidf = model.named_steps["tfidf"]

            values = tfidf.transform([message])

            feature_names = tfidf.get_feature_names_out()

            feature_table = pd.DataFrame({
                "Word": feature_names,
                "TF-IDF Value": values.toarray()[0]
            })

            feature_table = feature_table[
                feature_table["TF-IDF Value"] > 0
            ].sort_values(
                by="TF-IDF Value",
                ascending=False
            )

            st.dataframe(
                feature_table,
                use_container_width=True
            )

st.divider()

st.caption(
    "Smart Email Spam Detection | Python | Scikit-learn | Streamlit"
)
