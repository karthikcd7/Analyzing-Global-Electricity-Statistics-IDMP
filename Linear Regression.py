import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from math import sqrt
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv("data/TIDY_Monthly Electricity Statistics.csv")

electricity_df = df[df['Product'] == 'Electricity']
electricity_df = electricity_df[electricity_df['Balance'] == 'Net Electricity Production']
electricity_df = electricity_df.iloc[:,1:]
electricity_df.iloc[:,3:].columns = pd.to_datetime(electricity_df.iloc[:,3:].columns, format='%B-%Y')
electricity_df.index = pd.to_datetime(electricity_df.index)
df_monthly = electricity_df.groupby([electricity_df.index.year, electricity_df.index.month]).sum()
df_monthly = df_monthly.melt( var_name='Date', value_name='Electricity Production')

df_monthly.reset_index()
df_monthly["Date"] = pd.to_datetime(df_monthly['Date'], format='%B-%Y')
df_monthly['Date_Num'] = df_monthly['Date'].dt.month + df_monthly['Date'].dt.year * 100

train_data = df_monthly[:int(len(df_monthly)*0.75)]
test_data = df_monthly[int(len(df_monthly)*0.75):]

X = np.array(train_data["Date_Num"]).reshape(-1, 1) # use time as predictor variable
y = np.array(train_data['Electricity Production'])

model = LinearRegression()
fit_model = model.fit(X, y)

pred_X = np.array(test_data["Date_Num"]).reshape(-1,1)
predictions = fit_model.predict(pred_X)

# Visualizing the forecasted results
plt.figure(figsize=(10, 6))
plt.plot(train_data["Date_Num"].astype(str), train_data['Electricity Production'], label='Train')
plt.plot(test_data["Date_Num"].astype(str), test_data['Electricity Production'], label='Test')
plt.plot(test_data["Date_Num"].astype(str), predictions, label='Forecast')
plt.xticks(rotation=90)
plt.legend()
plt.title('Electricity Production Forecast using Linear Regression')
plt.xlabel('Date (Month-Year)')
plt.ylabel('Electricity Production (GWh)')
plt.show()


# evaluate forecasts
rmse = sqrt(mean_squared_error(test_data["Electricity Production"].values, predictions))
print('Test RMSE: %.3f' % rmse)
print('Test MSE: %.3f' % mean_squared_error(test_data["Electricity Production"].values, predictions))
print('Test MAE: %.3f' % mean_absolute_error(test_data["Electricity Production"].values, predictions))
print('Test MAPE: %.3f' % mean_absolute_percentage_error(test_data["Electricity Production"].values, predictions))


