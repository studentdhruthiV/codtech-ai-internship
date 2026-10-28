# Codtech AI Internship

This repository contains my projects for the Codtech AI Internship.

## Task 4: Stock Price Forecasting (NIFTY-50)

**Problem:** Predict the next day's closing price of RELIANCE using the previous 60 days of closing prices.

**Dataset:** NIFTY-50 Stock Market Data (2000-2021) from Kaggle: https://www.kaggle.com/datasets/rohanrao/nifty50-stock-market-data
(The dataset is not included in this repo because of its size.)

**Approach:**
- Used the Close price of RELIANCE
- Split the data by time: 80% training, 20% testing (no shuffling)
- Scaled the data with MinMaxScaler (fitted on training data only)
- Created sequences of 60 days to predict the next day
- Built a 2-layer LSTM model with Dropout and EarlyStopping
- Compared the results with a naive baseline (tomorrow's price = today's price)

**Results:**
LSTM  RMSE: 78.83 | MAE: 48.00 | MAPE: 3.76%
Naive baseline RMSE: 37.91

**CONCLUSION **: Overall project is completed with succesfull results.with all the outputs graph..which is represented in the ipynb file.

**Tools used:** Python, Pandas, NumPy, Scikit-learn, TensorFlow/Keras, Matplotlib

**File:** `stock_price_forecasting_nifty50.ipynb`





















## Task 5: Toxic Comment Classification

**Problem:** Classify comments into six toxicity categories using machine learning and natural language processing.

**Dataset:** Jigsaw Toxic Comment Classification Challenge.

**Technologies:** Python, Pandas, Scikit-learn, TF-IDF, Logistic Regression, Matplotlib, and Seaborn.

**Methodology:**
1. Load and explore the dataset.
2. Prepare the comment text and toxicity labels.
3. Extract text features using TF-IDF.
4. Train a multi-label Logistic Regression classifier.
5. Evaluate the model using F1 score, precision, recall, Hamming loss, and ROC-AUC.
6. Save the trained model and test predictions.


Label	Probability	Prediction
0	toxic	0.027028	False
1	severe_toxic	0.009513	False
2	obscene	0.023669	False
3	threat	0.019980	False
4	insult	0.059543	False
5	identity_hate	0.014216	False


label	probability	predicted_at_threshold
0	insult	0.999267	True
1	toxic	0.998389	True
2	obscene	0.939361	True
3	severe_toxic	0.368133	False
4	identity_hate	0.311768	False
5	threat	0.025918	False


Per-label classification report:
               precision    recall  f1-score   support

        toxic       0.66      0.85      0.74      3056
 severe_toxic       0.30      0.81      0.44       321
      obscene       0.73      0.87      0.79      1715
       threat       0.30      0.76      0.43        74
       insult       0.60      0.86      0.70      1614
identity_hate       0.28      0.71      0.40       294

    micro avg       0.59      0.85      0.70      7074
    macro avg       0.48      0.81      0.59      7074
 weighted avg       0.63      0.85      0.72      7074
  samples avg       0.06      0.08      0.07      7074





















## Task 1: Medical Image Classification (Chest X-Ray: Pneumonia vs Normal)

**Problem:** Classify chest X-ray images as NORMAL or PNEUMONIA using deep learning.

**Dataset:** Chest X-Ray Images (Pneumonia) from Kaggle: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

**Note on the dataset:** The dataset is very large (about 2 GB), so it could not be attached to this repository (GitHub does not allow files over 100 MB). It can be downloaded from the Kaggle link above.

**Approach:**
- Transfer learning with MobileNetV2 (pre-trained on ImageNet)
- Images resized to 224x224, with data augmentation (flip, rotation, zoom)
- 15% of the training images used for validation; the official test folder used only for final evaluation
- Class weights to handle the imbalance between NORMAL and PNEUMONIA images
- Two-phase training: top layers first, then fine-tuning the last layers with a small learning rate
- Evaluation with accuracy, precision, recall, F1, confusion matrix and ROC-AUC

**Results:** Because of the large size of the dataset (several GB) and limited computing resources, the model could not be trained for enough epochs, so fully accurate final results could not be obtained. The complete code and pipeline are in the notebook, and with more compute time the notebook can be run end to end to get the final accuracy, confusion matrix and ROC-AUC.

**Conclusion:** The full pipeline (data loading, augmentation, transfer learning, fine-tuning and evaluation) has been implemented. Final accuracy depends on training on the full dataset with enough compute.

**Tools used:** Python, Pandas, NumPy, Scikit-learn, TensorFlow/Keras, Matplotlib, Seaborn

**File:** `task1_medical_image_classification.ipynb`



