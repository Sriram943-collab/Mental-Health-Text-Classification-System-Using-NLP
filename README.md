# 🧠 Mental Health Text Classification System

An advanced **Natural Language Processing (NLP) system** that analyzes user text and predicts mental health conditions such as **Anxiety, Depression, and Normal** using **Word Embeddings (Word2Vec & FastText)** and Machine Learning models.

---

## 📌 Problem Statement

Understanding mental health from textual data is complex due to:

- Unstructured nature of language  
- Overlapping emotional expressions  
- Lack of clear boundaries between mental states  

This project addresses a **multi-class classification problem**, where the objective is to classify user text into:

- **Anxiety**
- **Depression**
- **Normal**

---

## 🧠 Solution Approach

This system transforms raw text into meaningful insights through:

1. Text preprocessing and cleaning  
2. Semantic representation using embeddings  
3. Feature extraction using sentence vectors  
4. Model training using multiple ML algorithms  
5. Balanced classification to avoid bias  
6. Real-time prediction using Streamlit  

---

## 🏗️ System Architecture

🔹 **User Input (Text)**  
⬇️  
🔹 **Text Preprocessing**  
(Cleaning + Tokenization + Stopword Removal)  
⬇️  
🔹 **Word Embeddings**  
(Word2Vec / FastText)  
⬇️  
🔹 **Sentence Vector Construction**  
(Averaging Word Vectors)  
⬇️  
🔹 **Machine Learning Models**  
(Logistic Regression / Random Forest / Naive Bayes)  
⬇️  
🔹 **Balanced Classification Engine**  
⬇️  
🔹 **Prediction Output (Mental State)**  
⬇️  
🔹 **Interactive Streamlit Interface**

---

## ⚙️ Data Preprocessing

- Handling missing and empty text values  
- Text normalization (lowercasing)  
- Removal of special characters  
- Tokenization using NLTK  
- Stopword removal for meaningful feature extraction  
- Dataset balancing to prevent model bias  

---

## 🧠 Word Embedding Techniques

### 🔹 Word2Vec
- Captures contextual similarity between words  
- Learns relationships based on surrounding words  

### 🔹 FastText
- Uses subword information for better representation  
- Handles unseen and rare words effectively  
- Provides improved generalization  

---

## 🤖 Model Development

Multiple models were implemented and evaluated:

- **Logistic Regression**
  - Works well with embedding features  
  - Provides stable and balanced predictions  

- **Random Forest**
  - Captures non-linear relationships  
  - Handles complex feature interactions  

- **Naive Bayes**
  - Fast baseline model  
  - Limited due to feature independence assumption  

---

## ⚡ Model Optimization

- Dataset balancing to reduce bias toward dominant class  
- Improved sentence vector normalization  
- Comparison of multiple models for best performance  
- Selection of optimal model based on accuracy and stability  

---

## 📊 Output System

The system provides:

- 🎯 Predicted mental health category  
- 📊 Confidence score  
- ⚡ Real-time prediction  
- 🖥️ Interactive user interface  

---

## 💡 Key Learnings

- Built a complete **end-to-end NLP pipeline**  
- Understood **semantic relationships using embeddings**  
- Compared multiple ML models  
- Handled real-world challenges like:
  - Class imbalance  
  - Misclassification  
  - Context loss  
- Deployed ML model as a **real-time application**  

---

## ⚠️ Limitations

- Long text may dilute semantic meaning due to averaging  
- Mixed emotional sentences may lead to misclassification  
- Performance depends on dataset quality  

---

## 🚀 Future Improvements

- Use transformer-based models (BERT)  
- Improve dataset size and labeling  
- Add explainable AI features  
- Enhance UI with visual analytics  

---

## 🏆 Conclusion

This project demonstrates how **NLP and Machine Learning** can be applied to analyze human emotions from text. It highlights the importance of **data preprocessing, embeddings, and model selection** in solving real-world problems.

---

## 👨‍💻 Author

**Krishnasagarapu Sri Ram**

---

## ⭐ If you found this useful

Give it a ⭐ on GitHub!
