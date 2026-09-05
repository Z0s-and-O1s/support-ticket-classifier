import streamlit as st
import joblib

MODEL_PATH = "support_ticket_model.pkl"


def predict_urgency(text):
    normalized = (text or "").lower()

    high_keywords = [
        "fraud",
        "fraudulent",
        "unauthorized",
        "accessed my account",
        "hacked",
        "breach",
        "security",
        "compromised",
    ]
    medium_keywords = [
        "login",
        "password",
        "charged",
        "transaction",
        "billing",
        "crashing",
        "error",
        "not working",
        "cannot",
        "issue",
        "account",
        "service",
    ]

    if any(keyword in normalized for keyword in high_keywords):
        return "High"
    if any(keyword in normalized for keyword in medium_keywords):
        return "Medium"
    return "Low"


def load_model():
    try:
        model_bundle = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        st.error(f"Model file not found at {MODEL_PATH}.")
        return None
    except Exception as exc:
        st.error(f"Failed to load model from {MODEL_PATH}: {exc}")
        return None

    if not isinstance(model_bundle, dict):
        st.error("The saved model file does not contain the expected dictionary format.")
        return None

    required_keys = ["tfidf_vectorizer", "ticket_classifier"]
    missing_keys = [key for key in required_keys if key not in model_bundle]
    if missing_keys:
        st.error(f"Missing required model keys: {', '.join(missing_keys)}")
        return None

    return model_bundle


st.set_page_config(page_title="Support-Ticket Category Classifier", page_icon="📩")

st.title("Support-Ticket Category Classifier")
st.write(
    "The app predicts the support-ticket category and urgency from ticket text."
)

example_tickets = [
    "I cannot log into my account and my password is not working.",
    "I was charged twice for the same transaction.",
    "The application keeps crashing when I open it.",
    "I want to know more about your services and available plans.",
    "Someone accessed my account and I believe there are fraudulent transactions.",
]

for example in example_tickets:
    if st.button(example, key=f"example_{example[:20]}"):
        st.session_state.ticket_text = example

default_ticket = st.session_state.get("ticket_text", "")
ticket_text = st.text_area(
    "Ticket Details",
    value=default_ticket,
    height=220,
    placeholder="Paste or type a support ticket here...",
)

if st.button("Classify Ticket", type="primary"):
    if not ticket_text or not ticket_text.strip():
        st.warning("Please enter a support ticket before classifying.")
    else:
        model_bundle = load_model()
        if model_bundle is None:
            st.stop()

        try:
            tfidf_vectorizer = model_bundle["tfidf_vectorizer"]
            ticket_classifier = model_bundle["ticket_classifier"]

            features = tfidf_vectorizer.transform([ticket_text])
            predicted_category = ticket_classifier.predict(features)[0]
            predicted_urgency = predict_urgency(ticket_text)
        except Exception as exc:
            st.error(f"An error occurred during classification: {exc}")
            st.stop()

        st.subheader("Predicted Category")
        st.write(predicted_category)

        st.subheader("Predicted Urgency")
        st.write(predicted_urgency)
