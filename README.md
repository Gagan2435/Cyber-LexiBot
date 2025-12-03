Cyber LexiBot — AI-Based Cyber Law Assistant

Cyber LexiBot is an intelligent system that analyzes user-reported cyber incidents, classifies the type of issue, extracts key information, retrieves relevant Indian cyber-law sections, and provides structured guidance for further action.

Key Capabilities
Intent Classification

A fine-tuned transformer model categorizes user queries into:
Cyber Fraud, Cyber Hacking, Cyber Harassment, Cyber Identity Theft, Cyber Privacy Violation, or General Cyber Queries.

Legal Section Retrieval

Sentence-BERT embeddings combined with a FAISS similarity index retrieve the most relevant sections from the Information Technology Act based on the user’s query.

Information Extraction

Using spaCy, the system identifies essential details such as platforms involved, monetary amounts, locations, and person names to refine the final response.

Actionable Cyber-Guidance

The system provides practical next steps, evidence-collection advice, and official reporting links based on the identified incident type.

System Workflow (Summary)

User input is processed.

The intent classifier predicts the type of incident.

Entities and other useful details are extracted.

Similar legal sections are retrieved using Sentence-BERT + FAISS.

A structured and legally aligned response is generated.

Model Evaluation

The classification model was evaluated on a 2000-record test set.
Metrics include accuracy, precision, recall, F1-score, and a confusion matrix.
The final model achieved approximately 84% accuracy after applying realistic noise-based evaluation.

Running the Project

Install all required Python libraries.

Start the system using:

python lexibot_core.py


The chatbot will open in the terminal and accept user queries.

Future Improvements

Web-based interface

Enhanced named-entity recognition

Support for multilingual queries

Integration with automated complaint-reporting workflows
