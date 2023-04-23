import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from math import sqrt

df = pd.read_csv("data/TIDY_Monthly Electricity Statistics.csv")

electricity_df = df[df['Product'] == 'Electricity']
electricity_df = electricity_df[electricity_df['Balance'] == 'Net Electricity Production']
electricity_df = electricity_df.iloc[:,1:]
electricity_df.iloc[:,3:].columns = pd.to_datetime(electricity_df.iloc[:,3:].columns, format='%B-%Y')
electricity_df.index = pd.to_datetime(electricity_df.index)
df_monthly = electricity_df.groupby([electricity_df.index.year, electricity_df.index.month]).sum()
df_monthly = df_monthly.melt( var_name='Date', value_name='Electricity Production')

# df_monthly.reset_index()
df_monthly.set_index('Date', inplace=True)
train_data = df_monthly[:int(len(df_monthly)*0.75)]
test_data = df_monthly[int(len(df_monthly)*0.75):]

# Fitting the ARIMA model
model = ARIMA(train_data, order=(6, 0, 3))
fit_model = model.fit()

# Forecasting the demand for the testing period
forecast = fit_model.forecast(steps=len(test_data))

future_predictions = fit_model.predict(start='2023-02-01', end='2024-01-01', dynamic=True)
future_dates = ['2023-02-01', '2023-03-01','2023-04-01','2023-05-01','2023-06-01','2023-07-01',
                '2023-08-01','2023-09-01', '2023-10-01', '2023-11-01', '2023-12-01', '2024-01-01']

# Visualizing the forecasted results
plt.figure(figsize=(10, 6))
plt.plot(train_data.index, train_data['Electricity Production'], label='Train')
plt.plot(test_data.index, test_data['Electricity Production'], label='Test')
plt.plot(test_data.index, forecast, label='Forecast')
plt.plot(future_dates, future_predictions, label="Future Predictions")
plt.xticks(rotation=90)
plt.legend()
plt.title('Electricity Production Forecast using ARIMA')
plt.xlabel('Date (Month-Year)')
plt.ylabel('Electricity Production')


# summary of fit model
print(fit_model.summary())
# line plot of residuals
residuals = pd.DataFrame(fit_model.resid)
residuals.plot(ylabel="Residual")
# density plot of residuals
residuals.plot(kind='kde', xlabel="Residual")
plt.show()
# summary stats of residuals
print(residuals.describe())

# evaluate forecasts
rmse = sqrt(mean_squared_error(test_data, forecast))
print('Test RMSE: %.3f' % rmse)
print('Test MSE: %.3f' % mean_squared_error(test_data, forecast))
print('Test MAE: %.3f' % mean_absolute_error(test_data, forecast))
print('Test MAPE: %.3f' % mean_absolute_percentage_error(test_data["Electricity Production"].values, forecast))



