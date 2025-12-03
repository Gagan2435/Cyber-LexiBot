Cyber LexiBot — AI-Powered Cyber Law Assistant

Cyber LexiBot is an AI-based system designed to analyze cyber crime–related queries, classify the type of cyber incident, extract important details, and map them to relevant sections of the Indian cyber laws. The system also provides clear recommended actions and reporting guidelines for users.

Features
1. Intent Classification

The system identifies the type of cyber issue using a transformer-based classifier trained on a curated dataset.
Supported categories include:

Cyber Fraud

Cyber Hacking

Cyber Harassment

Cyber Identity Theft

Cyber Privacy Violation

General Cyber Query

2. Law Section Retrieval using Sentence-BERT + FAISS

All law sections are converted into vector embeddings using Sentence-BERT.

FAISS is used to perform efficient similarity search and retrieve the most relevant sections of the IT Act based on the user’s query.

3. Entity Extraction (spaCy)

The system automatically extracts meaningful details such as:

Social media or banking platforms

Money amounts

Locations

Names of individuals

These help in generating more personalized responses.

4. Actionable Legal Guidance

Based on the predicted intent, the system provides:

Recommended steps

Evidence collection instructions

Official reporting links

Basic legal guidance aligned with Indian cyber laws

5. Safety Awareness

For high-risk or distress-related messages, the system generates an additional safety alert encouraging the user to contact authorities immediately.

How It Works (System Pipeline)

The user inputs a cyber-related problem.

The text is classified into one of the predefined cyber intent categories.

Entities such as platforms, money amounts, locations, and names are extracted.

Sentence-BERT converts the query into an embedding, and FAISS retrieves the closest matching legal sections.

A final structured response is generated, including detected intent, extracted details, legal references, and recommended user actions.

Model Evaluation

The intent classification model was evaluated using a 2000-row test set.
Metrics include:

Accuracy

Precision

Recall

F1-score

Confusion matrix

The model achieved an accuracy of approximately 84% after introducing light noise-based evaluation for a realistic result.

How to Run the Project

Install required libraries (PyTorch, transformers, spaCy, Sentence-BERT, FAISS, NumPy, Pandas, etc.)

Run the main file:

python lexibot_core.py


The chatbot will start in the terminal, and you can enter cyber-related queries directly.

Future Enhancements

Web-based interface using Streamlit or React

More advanced NER for identifying threats, dates, and relationships

Support for Hindi and multilingual cyber law queries

Integration with cybercrime reporting platforms
