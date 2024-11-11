Patent Match Prediction
This repository showcases a comprehensive project focused on Patent Match Prediction using state-of-the-art information retrieval techniques and natural language processing (NLP). The project explores methods like TF-IDF, Word2Vec, BM25, and advanced machine learning algorithms to address various tasks in the domain of patent analysis.

Project Overview
The main objective of this project is to develop a system that can efficiently and accurately match citing patents to cited patents. This includes identifying relevant paragraphs that justify citations and classifying the type of citation links. The project was structured into three distinct tasks, each addressing a specific problem.

Key Goals:
Accurately match citing patents to their corresponding cited patents.
Extract relevant paragraphs that provide justification for the citation.
Classify the nature of citation links using machine learning models.
Project Structure
Task 1: Citation Matching
Objective: Given a citing patent, identify the most relevant cited patents.
Techniques Used:
TF-IDF and BM25 for vectorization and text similarity measurement.
Implemented information retrieval models to rank potential matches.
Evaluation Metrics:
Mean Average Precision (MAP)
Recall at K
Task 2: Paragraph Matching
Objective: For a given citing and cited patent pair, identify specific paragraphs that justify the citation.
Techniques Used:
Utilized Word2Vec embeddings for semantic understanding.
Applied BM25 for measuring relevance at the paragraph level.
Evaluation Metric:
Mean Average Precision (MAP)
Task 3: Citation Classification
Objective: Classify the nature of the citation relationship between citing and cited patents (e.g., technical reference, background, or innovation build-up).
Techniques Used:
Employed an XGBoostClassifier for citation type classification.
Focused on balanced accuracy metrics to ensure fair evaluation across classes.
Data Description
The dataset includes cleaned and preprocessed patent documents, comprising abstracts, claims, and full-text content. The following files are utilized in this project:

Data Files:
Citing_ID_List_Test.json: A list of citing patent IDs for testing.
create_embeddings.py: A Python script to generate word embeddings using Word2Vec.
starter_notebook.ipynb: A starter Jupyter Notebook to guide the initial project setup.
Project Notebooks:
task1.ipynb: Notebook for the citation matching task.
task2.ipynb: Notebook for the paragraph matching task.
task3.ipynb: Notebook for the citation classification task.
Predictions Folder:
Contains output prediction files generated during model evaluation.
