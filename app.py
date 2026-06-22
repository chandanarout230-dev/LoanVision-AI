from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Load model and scaler
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

    app.logger.info("Model and scaler loaded successfully")

except Exception as e:
    app.logger.error(f"Error loading model files: {e}")
    model = None
    scaler = None


@app.route("/", methods=["GET", "POST"])
def predict_loan_status():

    prediction_result = ""

    if request.method == "POST":
        try:
            if model is None or scaler is None:
                return render_template(
                    "index.html",
                    prediction_text="Error: Model not loaded."
                )

            features = [
                float(request.form["Gender"]),
                float(request.form["Married"]),
                float(request.form["Dependents"]),
                float(request.form["Education"]),
                float(request.form["Self_Employed"]),
                float(request.form["ApplicantIncome"]),
                float(request.form["CoapplicantIncome"]),
                float(request.form["LoanAmount"]),
                float(request.form["Loan_Amount_Term"]),
                float(request.form["Credit_History"]),
                float(request.form["Property_Area"])
            ]

            features_array = np.array(features).reshape(1, -1)
            features_scaled = scaler.transform(features_array)

            prediction = model.predict(features_scaled)

            if prediction[0] == 1:
                prediction_result = "Congratulations! Your loan application is likely to be approved."
            else:
                prediction_result = "We regret to inform you that your loan application may be rejected."

        except ValueError:
            prediction_result = "Please enter valid numeric values."

        except Exception as e:
            app.logger.error(str(e))
            prediction_result = "An unexpected error occurred."

    return render_template(
        "index.html",
        prediction_text=prediction_result
    )


@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=False)