# FAKE-NEWS_DETECTION_SYSTEM

## Overview
The **Fake News Detection System** is a Natural Language Processing (NLP) project designed to identify whether a news article is **real** or **fake** using deep learning.  
This project combines the power of **transformer-based language models** (RoBERTa) with a custom-built neural network for binary classification.

By leveraging contextual embeddings from RoBERTa, the system can capture nuanced linguistic patterns and subtle cues often found in deceptive or biased writing.

---

## Objective
Misinformation poses a significant challenge in the digital age.  
The goal of this project is to build an automated system capable of distinguishing fake news from real news, assisting journalists, researchers, and online platforms in promoting credible information.

---

## Key Features
- Utilizes **RoBERTa (a variant of BERT)** for contextual text understanding  
- Includes **advanced text preprocessing** (stopword removal, lemmatization, cleaning)  
- Employs a **custom neural network classifier** built in PyTorch  
- Achieves **96.63% validation accuracy** on benchmark data  
- Ready for **deployment with Streamlit** for real-time prediction  

---

## Practical Applications
This model can be deployed in several real-world scenarios:

- **Media Verification:** Help editors and journalists filter unreliable articles before publication  
- **Social Media Monitoring:** Flag potentially misleading posts for review  
- **Academic Research:** Serve as a foundation for misinformation detection studies  
- **Public Awareness Tools:** Integrate with browser extensions or APIs to indicate article credibility  
- **Education:** Demonstrate transformer-based NLP systems in data science or AI courses  

---

## Dataset Description
The dataset consists of two labeled CSV files:

- `Fake.csv` — containing fabricated or misleading articles  
- `True.csv` — containing verified and authentic articles  

Each file contains the article **title**, **text**, and **publication metadata**.  
A new label column is added where:
- `1` represents *Fake News*  
- `0` represents *True News*  

After labeling, both datasets are merged and randomly split into **training (70%)** and **testing (30%)** sets.

---

## Data Preprocessing Pipeline
To ensure consistent and meaningful input to the model, each article undergoes a structured cleaning process:

1. **Text Normalization:** Converts all text to lowercase  
2. **Noise Removal:** Removes HTML tags, URLs, hashtags, and non-alphabetic symbols  
3. **Tokenization:** Splits text into meaningful tokens using NLTK  
4. **Stopword Removal:** Eliminates common but uninformative words (e.g., “the”, “and”)  
5. **Lemmatization:** Converts words to their root forms for better semantic consistency  
6. **Reconstruction:** Joins cleaned tokens back into sentences  

The result is a new `cleaned_text` column ready for tokenization.

---

## Model Architecture

### 1. RoBERTa Encoder
- Base Model: `roberta-base` (a robustly optimized version of BERT)
- Pretrained on large text corpora to understand linguistic context  
- The model outputs **contextual embeddings** for each token  
- The `[CLS]` token representation serves as the overall sentence embedding  

### 2. Custom Classification Head
Built using **PyTorch**, this lightweight network classifies the news as fake or real.

Architecture summary:
- Input: 768-dimensional embedding from RoBERTa  
- Hidden Layers:
  - Dense layer with 256 neurons (ReLU activation, Dropout 0.3)  
  - Dense layer with 128 neurons (ReLU activation, Dropout 0.2)  
- Output Layer:
  - Single neuron with **Sigmoid activation** for binary prediction (0 = Real, 1 = Fake)

### 3. Training Strategy
- **Optimizer:** Adam  
- **Loss Function:** Binary Cross Entropy (BCELoss)  
- **Learning Rate:** 0.001  
- **Batch Size:** 64  
- **Epochs:** 3  
- **Computation:** GPU acceleration (if available)

During training, RoBERTa’s parameters are **frozen**, allowing the model to act as a feature extractor.  
This ensures faster convergence and prevents overfitting on smaller datasets.

---

## Model Performance

| Metric              | Value      |
|---------------------|------------|
| Validation Loss      | 0.0923     |
| Validation Accuracy  | 96.63%     |

The model demonstrates strong generalization capability and balanced classification accuracy.  
For production-grade analysis, it can be extended to include **F1 score**, **precision**, **recall**, and **confusion matrices**.

---

## System Workflow
1. **Data Collection:** Load and merge labeled news datasets.  
2. **Preprocessing:** Clean and prepare text data.  
3. **Tokenization:** Convert text into numerical tokens using RoBERTa tokenizer.  
4. **Model Training:** Fine-tune the classification layers on top of RoBERTa features.  
5. **Evaluation:** Test the model on unseen data to measure accuracy.  
6. **Deployment:** Use Streamlit to provide a web interface for real-time predictions.  

---

## How the Model Works
When a user inputs a news article through the Streamlit interface:
The text is preprocessed using the same cleaning function used during training.
It is tokenized with the RoBERTa tokenizer, ensuring compatibility with the model.
The model predicts a probability between 0 and 1:
Closer to 1 → Fake News
Closer to 0 → Real News
The prediction is displayed instantly along with a confidence score.

## Limitations and Ethical Use
While the model performs well, it is not a factual verification system.
It identifies linguistic patterns rather than verifying claims.
Therefore:
Predictions should support, not replace, human judgment.
Biases in the dataset may influence model outcomes.
Results should not be used for censorship or punitive actions without manual review.

## Future Work
Fine-tune all RoBERTa layers for further accuracy improvement
Expand dataset diversity for better generalization
Add interpretability modules (e.g., SHAP, LIME)
Support multilingual news classification
Develop APIs for large-scale deployment

## License
This project is released under the MIT License.
You are free to use, modify, and distribute it with appropriate attribution.

## Author

Developed by: [Shiwan Mangate]
GitHub: https://github.com/<shiwan-mangate>

Contact: mangateshiwan@gmail.com



