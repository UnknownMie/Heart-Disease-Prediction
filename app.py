from flask import Flask, render_template, request, flash, redirect, url_for
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
app = Flask(__name__)
app.secret_key = 'some_secret_key'

# Load the dataset and preprocess it
df = pd.read_csv('heart.csv')

# Selecting correlated features using Heatmap
# Get correlation of all the features of the dataset
corr_matrix = df.corr()
top_corr_features = corr_matrix.index

dataset = pd.get_dummies(df, columns=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])
dataset.columns
from sklearn.preprocessing import StandardScaler
standScaler = StandardScaler()
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
dataset[columns_to_scale] = standScaler.fit_transform(dataset[columns_to_scale])
dataset.head()
# Splitting the dataset into dependent and independent features
X = dataset.drop('target', axis=1)
y = dataset['target']
# Finding the best accuracy for knn algorithm using cross_val_score
knn_scores = []
for i in range(1, 21):
  knn_classifier = KNeighborsClassifier(n_neighbors=i)
  cvs_scores = cross_val_score(knn_classifier, X, y, cv=10)
  knn_scores.append(round(cvs_scores.mean(),3))
# Training the knn classifier model with k value as 12
knn_classifier = KNeighborsClassifier(n_neighbors=12)
knn_classifier.fit(X, y)

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/details', methods=['POST'])
def save_details():
    # Here you can save the details to a database or session if needed
    # For now, we'll just redirect to the prediction form
    return redirect(url_for('index'))

@app.route('/index')
def index():
    return render_template('index.html', show_form=True)

@app.route('/predict', methods=['POST'])
def predict():
    result = None
    show_form = False

    try:
        # Extract input values
        age = float(request.form['age'])
        sex = int(request.form['sex'])
        cp = int(request.form['cp'])
        trestbps = float(request.form['trestbps'])
        chol = float(request.form['chol'])
        fbs = int(request.form['fbs'])
        restecg = int(request.form['restecg'])
        thalach = float(request.form['thalach'])
        exang = int(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        slope = int(request.form['slope'])
        ca = int(request.form['ca'])
        thal = int(request.form['thal'])

        # Input validation
        if age < 0 or age > 120:
            flash('Please enter a valid age between 0 and 120.')
            return render_template('index.html')
        if sex not in [0, 1]:
            flash('Invalid value for sex.')
            return render_template('index.html')
        # ... [similar checks for other fields]

        # One-hot encoding
        sex_0 = 1 if sex == 0 else 0
        sex_1 = 1 if sex == 1 else 0

        cp_0 = 1 if cp == 0 else 0
        cp_1 = 1 if cp == 1 else 0
        cp_2 = 1 if cp == 2 else 0
        cp_3 = 1 if cp == 3 else 0

        fbs_0 = 1 if fbs == 0 else 0
        fbs_1 = 1 if fbs == 1 else 0

        restecg_0 = 1 if restecg == 0 else 0
        restecg_1 = 1 if restecg == 1 else 0
        restecg_2 = 1 if restecg == 2 else 0

        exang_0 = 1 if exang == 0 else 0
        exang_1 = 1 if exang == 1 else 0

        slope_0 = 1 if slope == 0 else 0
        slope_1 = 1 if slope == 1 else 0
        slope_2 = 1 if slope == 2 else 0

        ca_0 = 1 if ca == 0 else 0
        ca_1 = 1 if ca == 1 else 0
        ca_2 = 1 if ca == 2 else 0
        ca_3 = 1 if ca == 3 else 0
        ca_4 = 1 if ca == 4 else 0

        thal_0 = 1 if thal == 0 else 0
        thal_1 = 1 if thal == 1 else 0
        thal_2 = 1 if thal == 2 else 0
        thal_3 = 1 if thal == 3 else 0

        input_data = [age, trestbps, chol, thalach, oldpeak, sex_0, sex_1, cp_0, cp_1, cp_2, cp_3, fbs_0, fbs_1,
                      restecg_0, restecg_1, restecg_2, exang_0, exang_1, slope_0, slope_1, slope_2, ca_0, ca_1, ca_2,
                      ca_3, ca_4, thal_0, thal_1, thal_2, thal_3]

        # Ensure the input data is scaled
        input_data = np.array(input_data).reshape(1, -1)
        # Extract the columns that need scaling
        data_to_scale = np.array(input_data[0][:5]).reshape(1, -1)

        # Scale those columns
        scaled_data = standScaler.transform(data_to_scale)

        # Concatenate the scaled data with the one-hot encoded columns
        input_data = np.concatenate((scaled_data, input_data[0][5:]), axis=None).reshape(1, -1)

        prediction = knn_classifier.predict(input_data)
        result = f'Prediction: {"Has Heart Disease" if prediction[0] == 1 else "No Heart Disease"}'
    except Exception as e:
        flash(f"An error occurred: {str(e)}")
        show_form = True

    return render_template('index.html', result=result, show_form=show_form)

if __name__ == '__main__':
    app.run(debug=True)