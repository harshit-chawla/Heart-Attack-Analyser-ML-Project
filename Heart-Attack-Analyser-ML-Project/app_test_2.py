from flask import Flask, render_template, request, redirect
import csv
from  heart_analyser import *


app = Flask(__name__)

# Function to read columns from dataset
def get_columns_from_dataset(dataset_path):
    # Assume dataset is in CSV format
    import pandas as pd
    df = pd.read_csv(dataset_path)
    df = df.drop("result", axis=1)
    return df.columns.tolist()

@app.route('/')
def home():
    return render_template('input_test.html')

@app.route('/predict', methods=['POST'])
def predict():
    dataset_path = r"heart.csv"  
    columns = get_columns_from_dataset(dataset_path)
    return render_template('input_test2.html', columns=columns)

def save_values_to_csv(values):
    with open('user_input.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([int(value) for value in values.values()])

@app.route('/result', methods=['POST'])
def result():
    # Capture the entered values
    entered_values = {key: request.form[key] for key in request.form}
    
    # Save the entered values to a CSV file
    save_values_to_csv(entered_values)

    predicted_value, a=report()

    # Define the risk level message based on U_pred
    if predicted_value == 0:
        risk_level = "You are at low risk"
        
    elif predicted_value == 1:
        risk_level = "You are at high risk"
        
    else:
        risk_level = "Prediction result not available"
        
    # Render a template or redirect to another page
    return render_template('result.html', entered_values=entered_values,risk_level=risk_level)

@app.route('/analyse_report', methods=['POST'])
def analyse_report():

    value=genrate_report()
    return render_template('report_analyse.html', value=value)

if __name__ == '__main__':
    app.run(debug=False)

