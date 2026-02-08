import pandas as pd

data = pd.read_csv("data/traffic_data.csv")
data['DateTime'] = pd.to_datetime(data['DateTime'])

data['Hour'] = data['DateTime'].dt.hour
data['DayOfWeek'] = data['DateTime'].dt.weekday
data['IsWeekend'] = data['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
data['Month'] = data['DateTime'].dt.month

def time_block(hour):
    if 0 <= hour <= 5:
        return 0
    elif 6 <= hour <= 9:
        return 1
    elif 10 <= hour <= 16:
        return 2
    elif 17 <= hour <= 20:
        return 3
    else:
        return 4

data['TimeBlock'] = data['Hour'].apply(time_block)

def season(month):
    if month in [3,4,5,6]:
        return 0
    elif month in [7,8,9]:
        return 1
    else:
        return 2

data['Season'] = data['Month'].apply(season)

data.to_csv("data/processed_data.csv", index=False)
print("Preprocessing completed successfully")