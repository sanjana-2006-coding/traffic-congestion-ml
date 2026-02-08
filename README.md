# Traffic Congestion Prediction & Intelligent Route Selection 🚦

This project predicts traffic congestion at multiple junctions using machine learning
and recommends the optimal route based on predicted vehicle count and contextual factors.

## Problem Statement
Urban traffic congestion causes delays, fuel waste, and stress. This system predicts
traffic intensity for different junctions at a given time and suggests the least
congested route.

## Tech Stack
- Python
- Pandas, NumPy
- XGBoost Regressor
- Scikit-learn
- Streamlit

## Features
- Time-based traffic prediction (hour, peak hours, weekends)
- Seasonal impact modeling
- Intelligent route scoring using congestion penalties
- Interactive Streamlit UI for real-time predictions

## Project Structure
traffic-prediction/
├── app.py
├── train_model.py
├── preprocessor.py
├── requirements.txt
├── README.md
├── data/
│ └── traffic_data.csv
├── model/
│ └── traffic_model.pkl

## How to Run
1. Create virtual environment
   python -m venv venv
   venv\Scripts\activate

2. Install dependencies
   pip install -r requirements.txt

3. Preprocess data
   python preprocessor.py

4. Train model
   python train_model.py

5. Run application
   streamlit run app.py

## Output
- Predicts vehicle count for each junction
- Classifies congestion as Low / Medium / High
- Recommends optimal route

## Future Enhancements
- Real-time traffic data integration
- Map-based visualization
- Deep learning models (LSTM)
- Deployment on cloud
