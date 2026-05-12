import joblib
import pandas as pd
model = joblib.load("loan_model.pkl")
columns = joblib.load("columns.pkl")

#sample input
Sample={
    "credit.policy":1,
    "int.rate":0.12,
    "installment":300,
    "log.annual.inc":11,
    "dti":15,
    "fico":700,
    "days.with.cr.line":3000,
    "revol.bal": 5000,
    "revol.util": 40,
    "inq.last.6mths": 1,
    "delinq.2yrs": 0,
    "pub.rec": 0
}

#convert to model format
df = pd.DataFrame([Sample])
df = pd.get_dummies(df)
df = df.reindex(columns=columns,fill_value=0)

prob = model.predict_proba(df)[0][1]

if prob>0.3:
    print("❌ Loan Rejected")
else:
    print("✅ Loan Approved")