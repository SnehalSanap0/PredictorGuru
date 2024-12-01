from flask import Flask, render_template, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import pickle
import os

app = Flask(__name__)
CORS(app)

# Dictionary to store loaded models and data for each round
round_data = {}
ROUNDS = ['Round 1', 'Round 2', 'Round 3', 'Round 4', 'Round 5']
CSV_FILES = {
    'Round 1': 'JoSAA1f.csv',
    'Round 2': 'JoSAA2f.csv',
    'Round 3': 'JoSAA3f.csv',
    'Round 4': 'JoSAA4f.csv',
    'Round 5': 'JoSAA5f.csv'
}

def clean_seat_type(seat_type):
    if pd.isna(seat_type):
        return 'OPEN'
    if 'PwD' in seat_type:
        return seat_type.split(' ')[0]
    return seat_type

def encode_categorical(df, column):
    le = LabelEncoder()
    df[column] = df[column].fillna('Unknown')
    df[f'ENCODED_{column}'] = le.fit_transform(df[column])
    return le

def load_and_train_round_data(round_name):
    """Load and train model for a specific round"""
    try:
        if round_name in round_data:
            return round_data[round_name]

        if round_name not in CSV_FILES:
            raise ValueError(f"Invalid round name: {round_name}")

        # Load the CSV file for this round
        file_path = CSV_FILES[round_name]
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV file not found for round {round_name}: {file_path}")

        df = pd.read_csv(file_path)

        # Create institute codes
        df['INSTITUTE CODE'] = df['INSTITUTE NAME'].apply(lambda x: ''.join([word[0] for word in x.split()]))

        # Handle missing values in the target variable
        df['CLOSING RANK'] = df.groupby(['INSTITUTE NAME', 'BRANCH NAME'])['CLOSING RANK'].transform(
            lambda x: x.fillna(x.median())
        )
        median_rank = df['CLOSING RANK'].median()
        df['CLOSING RANK'] = df['CLOSING RANK'].fillna(median_rank)

        # Encode categorical variables
        institute_le = encode_categorical(df, 'INSTITUTE NAME')
        branch_le = encode_categorical(df, 'BRANCH NAME')
        seat_type_le = encode_categorical(df, 'MODIFIED SEAT TYPE')
        district_le = encode_categorical(df, 'DISTRICT')
        gender_le = encode_categorical(df, 'GENDER')

        # Prepare features and target
        X = df[['ENCODED_INSTITUTE NAME', 'ENCODED_BRANCH NAME', 'ENCODED_MODIFIED SEAT TYPE', 
                'ENCODED_DISTRICT', 'ENCODED_GENDER']]
        y = df['CLOSING RANK']

        # Train the model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        round_data[round_name] = {
            'df': df,
            'model': model,
            'institute_le': institute_le,
            'branch_le': branch_le,
            'seat_type_le': seat_type_le,
            'district_le': district_le,
            'gender_le': gender_le,
            'options': {
                'seat_type_options': ['All'] + sorted(df['MODIFIED SEAT TYPE'].unique().tolist()),
                'branch_options': ['All'] + sorted(df['BRANCH NAME'].unique().tolist()),
                'district_options': ['All'] + sorted(df['DISTRICT'].unique().tolist()),
                'college_options': ['All'] + sorted(df['INSTITUTE NAME'].unique().tolist()),
                'gender_options': ['All'] + sorted(df['GENDER'].unique().tolist())
            }
        }

        return round_data[round_name]

    except Exception as e:
        print(f"Error loading data for round {round_name}: {str(e)}")
        raise

def search_colleges(round_name, rank, seat_types, branches, districts, college_name, gender, range_=300):
    try:
        data = round_data[round_name]
        df_filtered = data['df'].copy()
        
        if seat_types and 'All' not in seat_types:
            df_filtered = df_filtered[df_filtered['MODIFIED SEAT TYPE'].isin(seat_types)]
        
        if branches and 'All' not in branches:
            df_filtered = df_filtered[df_filtered['BRANCH NAME'].isin(branches)]
        
        if districts and 'All' not in districts:
            df_filtered = df_filtered[df_filtered['DISTRICT'].isin(districts)]
            
        if gender and gender != 'All':
            df_filtered = df_filtered[df_filtered['GENDER'] == gender]
        
        if college_name and college_name != 'All':
            df_filtered = df_filtered[df_filtered['INSTITUTE NAME'].str.contains(college_name, case=False, na=False)]
        
        rank = int(rank)
        result_df = df_filtered[(df_filtered['CLOSING RANK'] >= rank - range_) & 
                               (df_filtered['CLOSING RANK'] <= rank + range_)]
        
        if len(result_df) < 10:
            result_df = df_filtered.head(10)
        
        return result_df

    except Exception as e:
        print(f"Error in search_colleges: {str(e)}")
        raise

@app.route('/')
def home():
    try:
        return render_template('home.html', rounds=ROUNDS)
    except Exception as e:
        return render_template('error.html', error=str(e), rounds=ROUNDS)

@app.route('/jee')
def cet():
    try:
        selected_round = request.args.get('round', 'Round 1')
        data = load_and_train_round_data(selected_round)
        return render_template('index1.html',
                             rounds=ROUNDS,
                             selected_round=selected_round,
                             seat_type_options=data['options']['seat_type_options'],
                             branch_options=data['options']['branch_options'],
                             district_options=data['options']['district_options'],
                             college_options=data['options']['college_options'],
                             gender_options=data['options']['gender_options'])
    except Exception as e:
        return render_template('error.html', error=str(e), rounds=ROUNDS)

@app.route('/search', methods=['POST'])
def search():
    try:
        selected_round = request.form.get('round', 'Round 1')
        data = load_and_train_round_data(selected_round)
        
        rank = int(request.form['rank'])
        seat_types = request.form.getlist('seat_types')
        branches = request.form.getlist('branches')
        districts = request.form.getlist('districts')
        college_name = request.form.get('college_name', 'All')
        gender = request.form.get('gender', 'All')
        
        result_df = search_colleges(selected_round, rank, seat_types, branches, districts, college_name, gender)
        
        predictions = []
        for _, row in result_df.iterrows():
            features = np.array([[
                data['institute_le'].transform([row['INSTITUTE NAME']])[0],
                data['branch_le'].transform([row['BRANCH NAME']])[0],
                data['seat_type_le'].transform([row['MODIFIED SEAT TYPE']])[0],
                data['district_le'].transform([row['DISTRICT']])[0],
                data['gender_le'].transform([row['GENDER']])[0]
            ]])
            
            predicted_rank = data['model'].predict(features)[0]
            predictions.append(round(predicted_rank))
        
        result_df['Predicted Rank'] = predictions
        
        columns_to_display = ['INSTITUTE CODE', 'INSTITUTE NAME', 'BRANCH NAME', 'MODIFIED SEAT TYPE', 
                             'DISTRICT', 'GENDER', 'CLOSING RANK', 'Predicted Rank']
        result_df = result_df[columns_to_display]
        
        result_table = result_df.to_html(index=False,
                                       classes='table table-striped table-bordered',
                                       justify='center',
                                       escape=False)
        
        return render_template('results1.html', table=result_table, selected_round=selected_round, rounds=ROUNDS)
        
    except Exception as e:
        return render_template('error.html', error=str(e), rounds=ROUNDS)

@app.route('/about')
def about():
    try:
        return render_template('about.html', rounds=ROUNDS)
    except Exception as e:
        return render_template('error.html', error=str(e), rounds=ROUNDS)

@app.route('/instructions')
def instructions():
    try:
        return render_template('instructions.html', rounds=ROUNDS)
    except Exception as e:
        return render_template('error.html', error=str(e), rounds=ROUNDS)

if __name__ == '__main__':
    app.run(debug=True, port=8000)
