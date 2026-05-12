## Loan Default Risk Predictor

A machine learning web application that predicts whether a loan applicant is likely to default, built with Python, Scikit-learn, and Flask.



##  Overview

This project uses a **Random Forest classifier** trained on LendingClub loan data to predict loan default risk in real time. Applicants enter their financial details through a web form and instantly receive an approval or rejection decision with a default probability score.



##  Tech Stack

- Python, Flask
- Scikit-learn, Random Forest
- Pandas, NumPy
- HTML, CSS
- Gunicorn, Render



##  Model Performance

| Metric | Score |
|--------|-------|
| ROC AUC | 0.69 |
| Prediction Threshold | 0.30 |
| Estimators | 300 |
| Max Depth | 10 |

##  Why ROC AUC 0.69?

The model achieves a ROC AUC of 0.69 due to the following real-world reasons:

- The model achieves 0.69 ROC AUC because loan default behavior depends on real-life events 
- that 13 financial features alone cannot fully capture. The dataset is also imbalanced — 
- fully paid loans heavily outnumber defaults — making it harder for the model to learn default patterns. This is expected  with real-world noisy financial data.

### How it can be improved:
- Use `class_weight='balanced'` in Random Forest
- Try XGBoost or Gradient Boosting
- Add feature engineering (income to installment ratio etc.)
- Collect more features




##  Run Locally

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Train the model: `python train.py`
4. Run the app: `python app.py`
5. Visit `http://localhost:5000`

---

## 👤 Author

** NANDALA ANIRUDH **
- GitHub: (https://github.com/anirudhnandala8-ctrl)
- LinkedIn: (https://www.linkedin.com/in/anirudh-nandala-974194408/)