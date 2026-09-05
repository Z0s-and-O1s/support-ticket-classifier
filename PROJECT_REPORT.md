# PROJECT 01 --- Support-Ticket Category Classifier

## 1. Introduction

Customer support teams receive large numbers of tickets that must be
categorized and prioritized. This project develops an NLP-based
classifier that reads a support ticket subject and description and
predicts its category. A separate keyword-based component assigns Low,
Medium, or High urgency. A Streamlit application demonstrates the
complete system.

## 2. Dataset

The selected customer support dataset contains 20,000 ticket records and
12 original columns. The target variable is `Issue_Category`, with five
classes: Technical, Billing, Account, General Inquiry, and Fraud.
`Ticket_Subject` and `Ticket_Description` were combined into one text
feature. Customer and operational fields were not used as model inputs.

The dataset contained no missing values or duplicate rows. An 80/20
stratified train/test split with `random_state=42` produced 16,000
training samples and 4,000 test samples.

## 3. Methodology

The combined ticket text was converted into numerical features using
TF-IDF. The vectorizer used English stop-word removal, unigrams and
bigrams (`ngram_range=(1,2)`), and `min_df=2`. It was fitted only on
training text before transforming the test text.

Logistic Regression and Multinomial Naive Bayes were evaluated. Both
achieved 1.0000 accuracy and 1.0000 Macro F1 on the 4,000-ticket test
set. Logistic Regression was selected as the final classifier because it
matched the best performance and is a straightforward model for sparse
TF-IDF features.

  Model                       Accuracy   Macro F1
  ------------------------- ---------- ----------
  Logistic Regression             100%       100%
  Multinomial Naive Bayes         100%       100%

## 4. Model Validation

Because the perfect score was unusually high, additional checks were
performed. There were zero exact text values shared between training and
testing sets, and rechecking the predictions confirmed 4,000 correct
predictions out of 4,000.

The text was also checked for direct target-category terms. A
verification experiment removed `Technical`, `Billing`, `Account`,
`General Inquiry`, and `Fraud` from the text and retrained Logistic
Regression. Accuracy and Macro F1 remained at 100%. This suggests that
the result was not dependent on the explicit category names alone.
However, the unusually perfect score should be interpreted in the
context of this dataset and not assumed to represent all real-world
support tickets.

## 5. Urgency Classification

Urgency was implemented independently using transparent keyword rules.
High urgency includes terms such as urgent, emergency, immediately,
ASAP, critical, fraud, hacked, compromised, security breach, and locked
out. Medium includes terms such as soon, quickly, problem, issue, error,
failed, unable, not working, and delay. If no High or Medium keyword is
found, the result is Low.

## 6. Streamlit Demonstration

The Streamlit application loads the saved TF-IDF vectorizer and Logistic
Regression classifier from `support_ticket_model.pkl`. Users can paste a
ticket and select **Classify Ticket** to receive the predicted Issue
Category and Urgency. Example tickets are provided for quick testing.
The application was successfully tested locally.

## 7. Conclusion

The project demonstrates an end-to-end support-ticket classification
workflow covering text preparation, TF-IDF feature extraction,
supervised classification, evaluation, urgency rules, model packaging,
and an interactive web demo. The deliverables include the trained
classifier, saved model artifact, Streamlit application, requirements
file, README documentation, and this report.
