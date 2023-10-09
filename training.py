# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h2o
from autots import AutoTS


# reading data and preprocessing

train_a = pd.read_parquet('data/A/train_targets.parquet')
train_b = pd.read_parquet('data/B/train_targets.parquet')
train_c = pd.read_parquet('data/C/train_targets.parquet')

X_train_estimated_a = pd.read_parquet('data/A/X_train_estimated.parquet')
X_train_estimated_b = pd.read_parquet('data/B/X_train_estimated.parquet')
X_train_estimated_c = pd.read_parquet('data/C/X_train_estimated.parquet')

X_train_observed_a = pd.read_parquet('data/A/X_train_observed.parquet')
X_train_observed_b = pd.read_parquet('data/B/X_train_observed.parquet')
X_train_observed_c = pd.read_parquet('data/C/X_train_observed.parquet')

X_test_estimated_a = pd.read_parquet('data/A/X_test_estimated.parquet')
X_test_estimated_b = pd.read_parquet('data/B/X_test_estimated.parquet')
X_test_estimated_c = pd.read_parquet('data/C/X_test_estimated.parquet')

# X_train_estimated_a.to_csv('train_est_a.csv')


def preprocessing(data):
    if 'date_calc' in data.columns:
        data.drop('date_calc', axis=1, inplace=True)

    data['hour'] = data['date_forecast'].dt.hour
    data['month'] = data['date_forecast'].dt.month

    data.set_index('date_forecast', inplace=True)

    data = data.groupby(pd.Grouper(freq='1H')).mean()
    data.dropna(how='all', inplace=True)

    data['snow_density:kgm3'].fillna(0.0, inplace=True)

    """lag_window_size = 1  # Example lag window size
    feature_columns = data.columns  # Get a list of all feature columns

    for feature in feature_columns:
        for i in range(1, lag_window_size + 1):
            data[f'{feature}_lag_{i}'] = data[feature].shift(i)"""

    data.interpolate(method='time', inplace=True)

    data.rename_axis('time', inplace=True)

    return data


# combining observed and estimated data
x_train_a = pd.concat([X_train_observed_a, X_train_estimated_a], axis=0)
x_train_a = preprocessing(x_train_a)

x_train_b = pd.concat([X_train_observed_b, X_train_estimated_b], axis=0)
x_train_b = preprocessing(x_train_b)

x_train_c = pd.concat([X_train_observed_c, X_train_estimated_c], axis=0)
x_train_c = preprocessing(x_train_c)

# adding labels
# train_a.set_index('time', inplace=True)
train_a.fillna(0.0, inplace=True)
x_train_a = x_train_a.merge(train_a, how='inner', on='time')

train_b.set_index('time', inplace=True)
train_b.fillna(0.0, inplace=True)
x_train_b = x_train_b.merge(train_b, how='inner', on='time')

train_c.set_index('time', inplace=True)
train_c.fillna(0.0, inplace=True)
x_train_c = x_train_c.merge(train_c, how='inner', on='time')

# x_train = pd.concat([x_train_a, x_train_b, x_train_c], axis=0, ignore_index=True)


# AutoTS

model = AutoTS(forecast_length=14, frequency='infer',
               ensemble='simple', max_generations=10, num_validations=2)

model_a = model.fit(x_train_a, date_col='time', value_col='pv_measurement')

print(model_a)
