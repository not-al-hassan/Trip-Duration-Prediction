import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_log_error, r2_score
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.neighbors import KNeighborsClassifier
from haversine import haversine
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import  OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import lightgbm as lgb
from matplotlib import rcParams



# Random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)




if __name__ == '__main__':


    df_train = pd.read_csv('train.csv')
    df_test = pd.read_csv('test.csv')
    df_sample = pd.read_csv('sample_submission.csv')

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', 100)


    # Some information about my data
    print(df_test.columns)
    print(df_train.columns)
    print(df_train.head())
    print(df_test.head())
    print(df_train.info())
    print(df_test.info())
    print(df_train.describe())
    print(df_train.isnull().sum())

    # return the distance Between Pick-up and Drop-off
    def Distance(row):
        return haversine((row['pickup_latitude'], row['pickup_longitude']),
                         (row['dropoff_latitude'], row['dropoff_longitude']))


    # Add a new Feature called Dis
    df_train['Dis'] = df_train.apply(Distance, axis = 1)

    #Add a new Feature called Speed
    df_train['Speed'] = df_train['Dis'] / df_train['trip_duration'].replace(0, np.nan)
    df_train['Speed'] = df_train['Speed'].replace([np.inf, -np.inf], np.nan).fillna(0)

    print(df_train.columns)

    # Convert pickup_datetime from text into datetime that pandas can deal with.
    df_train['pickup_datetime'] = pd.to_datetime(df_train['pickup_datetime'])

    # Extract the day of year of the trip (1 → Jan 1st, 365 → Dec 31st).
    df_train['pickup_day_of_year'] = df_train['pickup_datetime'].dt.dayofyear

    # Extract day of week of the trip
    df_train['pickup_day_of_week'] = df_train['pickup_datetime'].dt.dayofweek + 1

    #Extract Hour of the trip
    df_train['pickup_hour_of_day'] = df_train['pickup_datetime'].dt.hour
    # The Extraction is usefull it help the model to deal with time-based pattern


    # plt.hist(df_train['trip_duration'].clip(None, 7000), bins=100) --->
    plt.show()

    df_train.dropna(inplace = True)
    df_train.drop_duplicates(inplace = True)

    # We notice from the histogram that most of trip duration is less than 5000
    df_train = df_train[df_train['trip_duration'] <= 5000]
    df_train = df_train[df_train['passenger_count'] > 0]  # First, exclude trips with no passengers
    df_train = df_train[df_train['passenger_count'] <= 6]  # Considering the capacity of a taxi, we temporarily assume that more than 6 passengers is invalid data

    plt.hist(df_train['Dis'].clip(None, 200), bins=100)
    plt.show()
    # We notice that max Distance is 25
    df_train = df_train[df_train['Dis'] > 0]
    df_train = df_train[df_train['Dis'] < 100]

    # Special case to check if the given trip is close to one of the popular airports in NewYork city.
    def air(row):
        jfk_coords = (40.645730, -73.784467)
        lag_coords = (40.776863, -73.874069)
        nwk_coords = (40.6925, -74.168611)

        if (haversine((row['dropoff_latitude'], row['dropoff_longitude']),
                      jfk_coords) < 2):
            return True
        elif (haversine((row['pickup_latitude'], row['pickup_longitude']), jfk_coords) < 2):
            return True
        elif (haversine((row['dropoff_latitude'], row['dropoff_longitude']), lag_coords) < 2):
            return True
        elif (haversine((row['pickup_latitude'], row['pickup_longitude']), lag_coords) < 2):
            return True
        elif (haversine((row['dropoff_latitude'], row['dropoff_longitude']), nwk_coords) < 2):
            return True
        elif (haversine((row['pickup_latitude'], row['pickup_longitude']), nwk_coords) < 2):
            return True
        return False

    df_train['air_port'] = df_train.apply(air, axis = 1)


    print(df_train[df_train['air_port'] == True])

    # Take Weather into consideration
    weather_data = pd.read_csv('weather_data_nyc_centralpark_2016(1).csv')
    weather_data = weather_data.replace('T', 0)
    weather_data['date_column'] = pd.to_datetime(weather_data['date'], format='%d-%m-%Y')

    print(weather_data)

    df_train['date'] = df_train.pickup_datetime.dt.date
    df_train['date'] = pd.to_datetime(df_train['date'])
    df_train = pd.merge(df_train, weather_data, left_on='date', right_on='date_column')

    df_train['precipitation'] = df_train['precipitation'].astype('float64')
    df_train['snow fall'] = df_train['snow fall'].astype('float64')
    df_train['snow depth'] = df_train['snow depth'].astype('float64')
    df_train['air_port'] = df_train['air_port'].astype(int)

    df_train.rename(columns=lambda c: c.strip().replace(" ", "_"), inplace=True)
    x = df_train[
        [
            'vendor_id',
            'passenger_count',
            'Dis',
            'pickup_day_of_year',
            'pickup_day_of_week',
            'pickup_hour_of_day',
            'air_port',
            'average_temperature',
            'precipitation',
            'snow_fall',
            'snow_depth'
        ]
    ]

    y = np.log1p(df_train['trip_duration'])  # log(1 + duration)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=42)

    model = lgb.LGBMRegressor(
        learning_rate=0.05,
        n_estimators=500,
        num_leaves=64,
        max_depth=-1,
        random_state=42
    )

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("Mean Squared Error : ", mse)
    print("Mean Absolute Error : ", mae)
    print("R² Score:", r2)
