# Dota 2 Match Duration Predictor

A Streamlit web application that predicts Dota 2 match duration based on hero selections for both teams.

## Setup

### Virtual Environment

This project uses a Python virtual environment to manage dependencies.

#### Option 1: Use the activation script (Recommended)
```bash
./activate_venv.sh
```

#### Option 2: Manual activation
```bash
source venv/bin/activate
```

#### Option 3: Create a new virtual environment
If you need to recreate the virtual environment:
```bash
# Remove existing venv
rm -rf venv

# Create new virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Running the Application

1. Activate the virtual environment (see options above)
2. Run the Streamlit app:
   ```bash
   streamlit run dota2_predictor.py
   ```
3. Open your browser and navigate to the URL shown in the terminal (usually http://localhost:8501)

## Features

### 🎮 Match ID Prediction
- Enter a Dota 2 match ID from OpenDota
- Automatically fetches hero selections from the match
- Predicts match duration using the loaded model
- Displays detailed results with probability distribution

### 🎯 Manual Hero Selection
- Manually select 5 heroes for Radiant team
- Manually select 5 heroes for Dire team
- Get predictions for custom hero combinations

### 📊 Results Display
- Predicted duration with confidence score
- Probability distribution chart
- Detailed probability table
- Hero team composition display

## Dependencies

The main dependencies are listed in `requirements.txt`:
- streamlit: Web framework for the application
- torch: PyTorch for the ML model
- matplotlib: For creating charts and visualizations
- numpy: For numerical operations
- pandas: For data manipulation
- requests: For fetching match data from OpenDota API

## Project Structure

- `dota2_predictor.py`: Main Streamlit application
- `poc_02_scripted_model.pt`: Trained PyTorch model
- `idx_to_name_dataset_02.pkl`: Hero index to name mapping
- `name_to_idx_dataset_02.pkl`: Hero name to index mapping
- `venv/`: Virtual environment directory
- `requirements.txt`: Python package dependencies
- `activate_venv.sh`: Script to activate virtual environment

## Deactivating

To deactivate the virtual environment:
```bash
deactivate
``` 