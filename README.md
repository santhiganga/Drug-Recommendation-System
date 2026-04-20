# Patient-Aware Clinical Intelligence System

A high-performance medical diagnostic tool combining traditional Machine Learning classifiers with a Large-Parameter Neural Reasoning Engine for personalized medication protocols.

## 🏗️ Technical Architecture

This system demonstrates a multi-stage clinical inference pipeline:

1.  **Stage 1: NLP Pre-Processing**
    - Text normalization, tokenization, and vectorization using `NLTK` and `TF-IDF`.
2.  **Stage 2: ML Classification**
    - Logistic Regression model classifies raw symptoms into potential medical categories.
3.  **Stage 3: Neural Refinement (NRE)**
    - A proprietary Neural Reasoning Engine (NRE) cross-references the ML prediction with patient metadata (age, gender, allergies) to synthesize high-precision medication recommendations.

## 📁 System Structure
- `app.py`: Main system entry point (Streamlit).
- `models/`: ML Model artifacts and training logic.
- `nlp/`: Core NLP preprocessing pipeline.
- `services/`: Neural inference engine logic.
- `data/`: Source datasets for model training.

project-name/
│
├── app.py                  # Main system entry point (Streamlit)
│
├── models/                 # ML model artifacts & training logic
│   ├── train_model.py
│   ├── predict.py
│   ├── model.pkl
│
├── nlp/                    # NLP preprocessing pipeline
│   ├── text_cleaning.py
│   ├── tokenizer.py
│   ├── vectorizer.py
│
├── services/               # Inference / prediction logic
│   ├── recommendation.py
│   ├── sentiment_analysis.py
│   ├── api_service.py
│
├── data/                   # Datasets for training/testing
│   ├── raw_data.csv
│   ├── processed_data.csv
│   ├── test_data.csv
  
  ## 🚀 Features
- 💊 Drug Recommendation System (ML-based)
- 🧠 Clinical Review Sentiment Analysis (NLP)
- 📊 Interactive Dashboard (Streamlit)
- 🗂️ Patient Condition-Based Filtering
- 📈 Data Visualization (charts & insights)
- 🔍 Text preprocessing & cleaning pipeline

## 🚀 Deployment Instructions

### 1. Prerequisites
Ensure your environment variables are configured. The system requires an active connection to the `Neural Engine`. Add your credentials to a `.env` file:
```bash
GROQ_API_KEY=your_connection_key
```

### 2. Install Engine Dependencies
```bash
pip install -r requirements.txt
```

### 3. Model Synchronization
Synchronize the ML classifier with the local dataset:
```bash
python models/train_model.py
```

### 4. Launch System
```bash
streamlit run app.py
```
### Future Enhancements
Deep learning-based recommendation system
Real-time API integration
User authentication system
Deployment on cloud (AWS/Streamlit Cloud)


## ⚖️ Clinical Disclaimer
**For Research & Educational Demonstration Only.** This software is a prototype for simulating Clinical Decision Support Systems (CDSS). It does not provide medical diagnosis and should never be used as a substitute for professional medical consultation.

### Author - Santhi Ganga
