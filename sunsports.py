
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX


dta = pd.read_csv("/content/drive/MyDrive/sunspots.csv")
dta.index = pd.Index(sm.tsa.datetools.dates_from_range('1700', '2008'))
del dta["YEAR"]


dta.head(10)
dta.describe()


fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(dta.values.squeeze(), lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(dta, lags=40, ax=ax2)

# Fitting SARIMA model
sarima_mod = SARIMAX(dta, order=(2, 0, 0), seasonal_order=(1, 0, 1, 12)).fit()
print(sarima_mod.summary())

print(sarima_mod.aic, sarima_mod.bic, sarima_mod.hqic)

# Plotting diagnostics
sarima_mod.plot_diagnostics(figsize=(10, 8))
plt.show()

# Making predictions
predict_sunspots_sarima = sarima_mod.predict('1990', '2012', dynamic=False)

# Plotting the original data and predictions
ax = dta.loc['1950':].plot(figsize=(12, 8))
ax = predict_sunspots_sarima.plot(ax=ax, style='r--', label='SARIMA Prediction')
ax.legend()
ax.axis((-20.0, 38.0, -4.0, 200.0))

# Calculating (MFE) and (MAE)
def mean_forecast_err(y, yhat):
    return y.sub(yhat).mean()

def mean_absolute_err(y, yhat):
    return np.mean((np.abs(y.sub(yhat).mean()) / yhat))

print("MFE = ", mean_forecast_err(dta.SUNACTIVITY, predict_sunspots_sarima))
print("MAE = ", mean_absolute_err(dta.SUNACTIVITY, predict_sunspots_sarima))
