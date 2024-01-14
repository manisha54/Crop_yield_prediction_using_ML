from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Length
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import joblib


app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'  # Replace with a secret key for secure sessions

# Flask-Login setup
login_manager = LoginManager(app)
login_manager.login_view = 'login'  # Set the login view

# Simple in-memory user database (replace with a real database in a production environment)
users = {'user1': {'password': 'password123'}}


class User(UserMixin):
    def __init__(self, username):
        self.id = username

@login_manager.user_loader
def load_user(username):
    return User(username)

# Define a registration form
class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=4, max=25)])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=6)])
    submit = SubmitField('Register')

# Define a login form
class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=4, max=25)])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=6)])
    submit = SubmitField('Login')

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
model_path = "C:/Users/dell/Downloads/gradient_boosting_model2.joblib"
pipeline = joblib.load(model_path)

# Define a route for the input page
@app.route('/')
def default():
    return redirect(url_for('register'))

# Add a registration route
@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('login'))  # Redirect logged-in users to the login page

    form = RegistrationForm()

    if form.validate_on_submit():
        username = form.username.data
        password = form.password.data

        # Check if the username is already taken
        if username in users:
            flash('Username is already taken. Choose a different one.', 'danger')
        else:
            # Store the user in the database (replace with database logic)
            users[username] = {'password': password}
            flash('Registration successful. You can now log in.', 'success')
            return redirect(url_for('login'))  # Redirect to the login page after successful registration

    return render_template('register.html', form=form)

        


# Add a login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('input_page'))  # Redirect logged-in users to the input page

    form = LoginForm()

    if form.validate_on_submit():
        username = form.username.data
        password = form.password.data

        # Check if the username and password are correct
        if username in users and users[username]['password'] == password:
            user = User(username)
            login_user(user)
            flash('Login successful.', 'success')
            return redirect(url_for('input_page'))
        else:
            flash('Invalid username or password. Please try again.', 'danger')

    return render_template('login.html', form=form)


# Define a route for the input page
@app.route('/input_page')
@login_required
def input_page():
    return render_template('input.html')

# Define a route for the input page
@app.route('/aboutus_page')
@login_required
def aboutus_page():
    return render_template('aboutus.html')

# Define a route for processing the input and displaying the result
@app.route('/result', methods=['POST'])
@login_required  # Add login_required decorator to protect this route


def result():
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

        # Pass the prediction result to the result page
        return render_template('result.html', prediction=prediction)

    except Exception as e:
        # Handle errors
        return render_template('input.html', prediction=f'Error: {str(e)}')



# Add a logout route
@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logout successful.', 'success')
    return redirect(url_for('login'))
    # return render_template('register.html', form=form)








if __name__ == '__main__':
    app.run(debug=True)
