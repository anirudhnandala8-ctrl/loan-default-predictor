import os
from flask import Flask, render_template, request
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("loan_model.pkl")
columns = joblib.load("columns.pkl")

# Single threshold — change here to update everywhere
THRESHOLD = 0.3


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = []
        purpose_selected = request.form["purpose"]

        for col in columns:

            # One-hot encoded purpose columns
            if col.startswith("purpose_"):
                input_data.append(1 if col == f"purpose_{purpose_selected}" else 0)

            # log.annual.inc: CSV already has log values, so user enters raw income
            # and we log-transform it here before feeding the model
            elif col == "log.annual.inc":
                income = float(request.form["log.annual.inc"])
                if income <= 0:
                    raise ValueError("Annual income must be a positive number.")
                input_data.append(np.log(income))

            # int.rate: CSV has decimals (0.12), user enters percentage (12),
            # so we divide by 100
            elif col == "int.rate":
                rate = float(request.form["int.rate"])
                input_data.append(rate / 100)

            # All other numeric inputs passed through directly
            else:
                value = float(request.form[col])
                input_data.append(value)

        input_df = pd.DataFrame([input_data], columns=columns)
        prob = model.predict_proba(input_df)[0][1]

        if prob > THRESHOLD:
            result = "Loan Rejected"
            result_type = "rejected"
        else:
            result = "Loan Approved"
            result_type = "approved"

        return render_template(
            "index.html",
            prediction=result,
            result_type=result_type,
            prob=round(prob * 100, 1)
        )

    except ValueError as e:
        return render_template(
            "index.html",
            error=str(e) if str(e) else "Please check all fields are filled with valid numbers."
        )
    except Exception:
        return render_template(
            "index.html",
            error="Something went wrong. Please check your inputs and try again."
        )


if __name__ == "__main__": 
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)