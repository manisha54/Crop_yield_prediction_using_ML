from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import joblib  # Add this line

app = Flask(__name__)
# Load your dataset
df = pd.read_csv("C:/Users/dell/Downloads/new_dataset.csv") 
# Load your trained model and preprocessor
model = GradientBoostingRegressor(n_estimators=100, random_state=42)

# Define categorical and numerical columns
categorical_columns = ['state_names', 'district_names', 'season_names', 'crop_names', 'soil_type']
numerical_columns = [col for col in df.columns if col not in categorical_columns + ['crop_yield']]

# Create transformers
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))
])

# Create preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_columns),
        ('cat', categorical_transformer, categorical_columns)
    ])

# Create and evaluate the pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                             ('regressor', model)])

# Load your pre-trained model
model_path = "C:/Users/dell/Downloads/gradient_boosting_model2.joblib"  # replace with the actual path to your saved model
pipeline = joblib.load(model_path)

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for the prediction page
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the HTML form
        crop_year = int(request.form['crop_year'])
        area = float(request.form['area'])
        temperature = float(request.form['temperature'])
        wind_speed = float(request.form['wind_speed'])
        precipitation = float(request.form['precipitation'])
        humidity = float(request.form['humidity'])
        soil_type = request.form['soil_type']
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        pressure = float(request.form['pressure'])
        state_names = request.form['state_names']
        district_names = request.form['district_names']
        season_names = request.form['season_names']
        crop_names = request.form['crop_names']

        # Create a DataFrame from the input data
        input_data = pd.DataFrame({
            'crop_year': [crop_year],
            'area': [area],
            'temperature': [temperature],
            'wind_speed': [wind_speed],
            'precipitation': [precipitation],
            'humidity': [humidity],
            'soil_type': [soil_type],
            'N': [N],
            'P': [P],
            'K': [K],
            'pressure': [pressure],
            'state_names': [state_names],
            'district_names': [district_names],
            'season_names': [season_names],
            'crop_names': [crop_names],
        })

        # Transform the input data
        preprocessor_transformed_data = pipeline.named_steps['preprocessor'].transform(input_data)

        # Make predictions using the model
        prediction = pipeline.named_steps['regressor'].predict(preprocessor_transformed_data)[0]

        return render_template('index.html', prediction=f'Predicted Crop Yield: {prediction:.2f}')

    except Exception as e:
        return render_template('index.html', prediction=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
