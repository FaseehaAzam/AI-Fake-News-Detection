

\# 📰 AI Fake News Detection \& Analysis



An interactive machine learning web application that analyzes news article text and classifies it as \*\*Real-like\*\* or \*\*Fake-like\*\* using Natural Language Processing (NLP).



Built with Python, Scikit-learn, TF-IDF, and Streamlit.



\## Project Overview



This project demonstrates an end-to-end NLP classification workflow, from data preprocessing and model training to evaluation and deployment through an interactive web application.



Users can paste news article text into the app and receive a prediction, model confidence score, article statistics, and an explanation of influential words.



\*\*Important:\*\* This application does not independently verify the factual accuracy of news. It identifies patterns learned from labeled training articles.



\## Features



\- Interactive Streamlit web application

\- Real-like and Fake-like text classification

\- Model confidence display

\- Word count and character count

\- Influential word analysis

\- Logistic Regression and Naive Bayes comparison

\- Confusion matrix visualization

\- Dataset distribution visualization



\## Technologies Used



\- Python

\- Pandas and NumPy

\- Scikit-learn

\- TF-IDF Vectorization

\- Logistic Regression

\- Multinomial Naive Bayes

\- Matplotlib and Seaborn

\- Streamlit

\- Joblib

\- Google Colab



\## Dataset



\*\*Dataset:\*\* Fake and Real News Dataset



\*\*Source:\*\* Kaggle



https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset



\### Data Preparation



\- Loaded fake and real news articles

\- Assigned binary classification labels

\- Cleaned and normalized article text

\- Removed empty articles

\- Removed exact duplicates based on cleaned text

\- Split the deduplicated dataset into training and testing sets

\- Applied TF-IDF vectorization



\### Dataset Statistics



| Description | Count |

|---|---:|

| Articles after cleaning | 44,183 |

| Duplicate articles removed | 5,870 |

| Unique articles | 38,313 |

| Training articles | 30,650 |

| Testing articles | 7,663 |

| TF-IDF features | 5,000 |



\## Model Performance



| Model | Test Accuracy |

|---|---:|

| Logistic Regression | 98.58% |

| Multinomial Naive Bayes | 93.63% |



Logistic Regression was selected for the application because it achieved the higher accuracy on the deduplicated random test split.



These results may not represent performance on unfamiliar news sources or newer articles because of potential dataset and source-specific patterns.



\## Visualizations



\### Confusion Matrix



!\[Confusion Matrix](confusion\_matrix.png)



\### Dataset Distribution



!\[News Distribution](news\_distribution.png)



\## How to Run the Application



\### 1. Install Python



Install a compatible version of Python.



\### 2. Install Dependencies



Open Command Prompt inside the project folder and run:



```bash

python -m pip install -r requirements.txt

```



\### 3. Launch the Application



```bash

python -m streamlit run app.py

```



\### 4. Open in Browser



Visit:



http://localhost:8501



\### 5. Analyze News



Paste a news article into the text box and click \*\*Analyze News\*\*.



The application will display the predicted category, model confidence, article statistics, and influential words.



\## Project Structure



```text

Fake\_News\_Detection\_App/

├── app.py

├── fake\_news\_model.pkl

├── tfidf\_vectorizer.pkl

├── confusion\_matrix.png

├── news\_distribution.png

├── requirements.txt

└── README.md

```



\## Limitations



\- The model is not a fact-checking service.

\- Predictions may be influenced by writing style and source-specific patterns.

\- Confidence scores are not guarantees of factual accuracy.

\- Performance on new articles may differ from the reported test results.

\- The current version supports English-language text.



\## Future Improvements



\- Evaluate on independent news sources

\- Detect near-duplicate articles

\- Add more detailed explainability charts

\- Improve the user interface

\- Deploy the application online



\## Author



Developed as an AI/ML portfolio project.



