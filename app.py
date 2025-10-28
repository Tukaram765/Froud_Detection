import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from datetime import datetime
from flask_cors import CORS 

app = Flask(__name__)
CORS(app) 

# --- Model Loading ---
current_script_directory = os.path.dirname(os.path.abspath(__file__))
# Construct the full path to the model file
model_file_name = "fraud_detection_model.pkl" 
model_path = os.path.join(current_script_directory, model_file_name)

pipeline = None 
try:
    pipeline = joblib.load(model_path)
    print(f"Model pipeline loaded successfully from: {model_path}")
except FileNotFoundError:
    print(f"Error: Model file not found at {model_path}. Please ensure 'fraud_detection_model.pkl' is in the same directory as app.py, or check the file name/path.")
    print("The server will start, but predictions will return an error until the model is found.")
except Exception as e:
    print(f"Error loading model from {model_path}: {e}")
    print("The server will start, but predictions will return an error until the model is loaded.")

@app.route('/predict', methods=['POST'])
def predict():
    if pipeline is None:
        return jsonify({'error': 'Model is not loaded. Cannot make predictions.'}), 500

    try:
        data = request.get_json(force=True)

        if not isinstance(data, dict):
            return jsonify({'error': 'Invalid input format. Expected a JSON object with claim data.'}), 400

        # Convert the input dictionary to a DataFrame
        input_df = pd.DataFrame([data])


        # 1. Convert AdmissionDt and DischargeDt to datetime and calculate StayDuration
        if 'AdmissionDt' in input_df.columns and 'DischargeDt' in input_df.columns:
            input_df['AdmissionDt'] = pd.to_datetime(input_df['AdmissionDt'], errors='coerce')
            input_df['DischargeDt'] = pd.to_datetime(input_df['DischargeDt'], errors='coerce')
            input_df['StayDuration'] = (input_df['DischargeDt'] - input_df['AdmissionDt']).dt.days
        else:
            input_df['StayDuration'] = 0 

        # 2. 'is_inpatient' should be provided directly by the frontend as 0 or 1
        if 'is_inpatient' in input_df.columns:
            input_df['is_inpatient'] = pd.to_numeric(input_df['is_inpatient'], errors='coerce').fillna(0).astype(int)
        else:
            input_df['is_inpatient'] = 0 

        # 3. 'NumClaims' calculation/provision.
        if 'NumClaims' not in input_df.columns or pd.isna(input_df['NumClaims']).all():
            input_df['NumClaims'] = 1 

        # 4. Select the exact features used for training and handle NaNs
        features_used_in_training = [
            'InscClaimAmtReimbursed',
            'DeductibleAmtPaid',
            'is_inpatient',
            'StayDuration',
            'NumClaims'
        ]

        # Create a DataFrame with only the required features, ensuring all are present
        X_predict_raw = pd.DataFrame(columns=features_used_in_training)
        for feature in features_used_in_training:
            if feature in input_df.columns:
                X_predict_raw[feature] = input_df[feature]
            else:
                X_predict_raw[feature] = 0 

        # Apply fillna(0) to the selected features, as done in your notebook
        X_predict_processed = X_predict_raw.fillna(0)


        # Make prediction using the loaded pipeline
        prediction = pipeline.predict(X_predict_processed)
        prediction_proba = pipeline.predict_proba(X_predict_processed)[:, 1]

        # Convert prediction to human-readable format
        result = 'Potential Fraud' if prediction[0] == 1 else 'No Fraud'
        probability = float(prediction_proba[0]) 

        return jsonify({
            'prediction': result,
            'probability_of_fraud': probability
        })

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # For development, run with debug=True.
    app.run(debug=True, port=5000)