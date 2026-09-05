# Support-Ticket Category Classifier

A Natural Language Processing (NLP) application that automatically classifies customer support tickets into predefined categories and assigns a simple urgency level.

## Project Overview

Customer support teams receive large numbers of tickets that need to be categorized and prioritized. This project uses machine learning to automatically predict the category of an incoming support ticket from its subject and description.

The application provides:

- **Ticket Category Prediction** using TF-IDF + Logistic Regression
- **Urgency Prediction** using keyword-based rules
- **Interactive Streamlit Web App**
- Saved trained model for making predictions on new tickets

## Categories

The classifier predicts one of five categories:

- Technical
- Billing
- Account
- General Inquiry
- Fraud

## Dataset

The project uses a customer support ticket dataset containing:

- **20,000 records**
- **12 original columns**
- **5 Issue Categories**
- No missing values
- No duplicate rows

The main text fields used for classification are:

- `Ticket_Subject`
- `Ticket_Description`

These are combined into a single `text` feature.

The target variable is:

- `Issue_Category`

Other customer and ticket-management fields were not used as model input.

## Machine Learning Pipeline

The classification pipeline is:

```text
Ticket Subject
      +
Ticket Description
      ↓
Combined Text
      ↓
Train/Test Split
      ↓
TF-IDF Vectorization
      ↓
Logistic Regression
      ↓
Predicted Issue Category


Access the webpage at : https://support-ticket-classifier-kvyw6yerncy2kkmup2j728.streamlit.app/