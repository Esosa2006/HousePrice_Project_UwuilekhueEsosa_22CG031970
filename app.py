"""
House Price Prediction Flask Application
Predicts house prices using a trained machine learning model.
"""
from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
MODEL_PATH = 'model/house_price_model.pkl'

try:
    model_data = joblib.load(MODEL_PATH)
    model = model_data['model']
    label_encoder = model_data['label_encoder']
    feature_columns = model_data['feature_columns']
    neighborhood_options = list(label_encoder.classes_)
except FileNotFoundError:
    raise RuntimeError(f"Model file not found at {MODEL_PATH}. Please ensure the model is trained and saved.")
except Exception as e:
    raise RuntimeError(f"Error loading model: {str(e)}")


@app.route('/')
def home():
    """Render the home page with neighborhood options and model metrics"""
    try:
        return render_template('index.html',
                               neighborhoods=neighborhood_options,
                               metrics=model_data.get('metrics', {}))
    except Exception as e:
        return f"Error loading page: {str(e)}", 500


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests from the form"""
    try:
        # Get and validate form data
        overall_qual = int(request.form.get('overall_qual', 0))
        gr_liv_area = float(request.form.get('gr_liv_area', 0))
        total_bsmt_sf = float(request.form.get('total_bsmt_sf', 0))
        garage_cars = float(request.form.get('garage_cars', 0))
        year_built = int(request.form.get('year_built', 0))
        neighborhood = request.form.get('neighborhood', '')

        # Validate required fields
        if not all([overall_qual, gr_liv_area, neighborhood, year_built]):
            return jsonify({
                'success': False,
                'error': 'Missing required fields'
            }), 400

        # Encode neighborhood
        neighborhood_encoded = label_encoder.transform([neighborhood])[0]

        # Create feature array in correct order
        features = np.array([[
            overall_qual,
            gr_liv_area,
            total_bsmt_sf,
            garage_cars,
            year_built,
            neighborhood_encoded
        ]])

        # Make prediction
        prediction = model.predict(features)[0]

        # Return result
        return jsonify({
            'success': True,
            'predicted_price': f'${prediction:,.2f}',
            'price_value': float(prediction)
        })

    except ValueError as e:
        return jsonify({
            'success': False,
            'error': f'Invalid input: {str(e)}'
        }), 400
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Prediction error: {str(e)}'
        }), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    }), 200


if __name__ == '__main__':
    # Use environment variable for port (required for deployment)
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug_mode)