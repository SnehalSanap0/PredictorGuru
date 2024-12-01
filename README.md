# PredictorGuru
PredictorGuru is a college prediction tool designed to assist students in forecasting their chances of admission into various engineering colleges based on their entrance exam ranks. It supports predictions for both the Joint Entrance Examination (JEE) and the Common Entrance Test (CET).

## Features
- **Dual Exam Support:** Provides predictions for both JEE and CET aspirants.
- **User-Friendly Interface:** Interactive web interface for inputting exam details and viewing results.
- **Data-Driven Predictions:** Utilizes historical data and machine learning models to estimate admission probabilities.
- **Separate Data Sources:**
  - **JEE:** Uses historical data from JoSAA files (JoSAA1f.csv, JoSAA2f.csv, etc.).
  - **CET:** Uses the output_file.csv for CET predictions.

## Installation
Clone the Repository:
```bash
git clone https://github.com/SnehalSanap0/PredictorGuru.git
cd PredictorGuru
```

## Usage
### Prepare Data:

- **For JEE:** Ensure that the JoSAA1f.csv, JoSAA2f.csv, etc., files are in the root directory.
- **For CET:** Ensure that output_file.csv is in the root directory.

### Run the Application:
- randomforest3.py - Runs for CET Data
- randomforest3(2).py - Runs for JEE Data

### Navigate the Interface:
- Use the navigation bar to switch between JEE and CET prediction tools.
- Input your exam rank and select relevant filters to view predicted admission chances.

### File Structure:
- /templates/: HTML templates for rendering web pages.
- /static/: Static files like CSS and images.
- JoSAA1f.csv, JoSAA2f.csv, ...: CSV files for JEE predictions.
- output_file.csv: CSV file for CET predictions.

## How It Works

- **JEE Predictions:**
  - Uses JoSAA data files (JoSAA1f.csv, JoSAA2f.csv, etc.) to train a machine learning model.
  - Provides insights based on your rank, category, branch, and other filters.

- **CET Predictions:**
  - Utilizes output_file.csv to train a separate machine learning model.
  - Predicts college options based on rank and other input parameters.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your changes.
