# Traffic Congestion Prediction & Route Recommendation

This project predicts traffic congestion at multiple junctions using machine learning and provides an intelligent route recommendation.

## Problem Statement
Urban traffic congestion causes delays, fuel loss, and stress. Static routing systems fail to adapt to time-based traffic variations.

## Solution
An ML-based system that:
- Predicts traffic volume using XGBoost
- Adjusts predictions with rule-based traffic heuristics
- Recommends the least congested route
- Estimates expected travel delay

## Dataset
- Historical traffic counts
- Timestamp-based features (hour, weekday, season)
- Junction identifiers

## Feature Engineering
- Peak hour detection
- Time blocks
- Hour squared (non-linearity)
- Weekend and seasonal indicators

## Model
- XGBoost Regressor
- Compared against mean baseline
- Feature importance extracted for interpretability

## Results
- XGBoost outperforms baseline prediction
- Peak hour and time-based features dominate importance

## Limitations
- Random train-validation split (not time-aware)
- No lag-based traffic features
- Junction interactions are heuristic-based

## Future Improvements
- Time-series validation
- Lag and rolling traffic features
- Graph-based route optimization

## How to Run
```bash
pip install -r requirements.txt
python preprocessor.py
python train_model.py
streamlit run app.py
