from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load model & encoders
model = joblib.load("model/svm_model.pkl")
features = joblib.load("model/features.pkl")
encoders = joblib.load("model/encoders.pkl")
target_encoder = joblib.load("model/target_encoder.pkl")
dropdown_options = joblib.load("model/dropdown_options.pkl")

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_label = None

    if request.method == 'POST':
        try:
            input_data = {}
            for feature in features:
                value = request.form[feature]
                value = encoders[feature].transform([value])[0]
                input_data[feature] = value

            input_df = pd.DataFrame([input_data])
            prediction = model.predict(input_df)[0]
            prediction_label = target_encoder.inverse_transform([prediction])[0]

        except Exception as e:
            prediction_label = f"Error: {e}"

    return render_template("index.html", features=features, dropdown_options=dropdown_options, prediction=prediction_label)

if __name__ == '__main__':
    app.run(debug=True)

