


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


df = pd.read_csv("train.csv")

print("First 5 Rows")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())


X = df[['GrLivArea','BedroomAbvGr','FullBath']]
y = df['SalePrice']


X = X.fillna(X.mean())


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


model = LinearRegression()

model.fit(X_train, y_train)


y_pred = model.predict(X_test)


print("\nModel Evaluation")

print("R2 Score :", r2_score(y_test, y_pred))

print("MAE :", mean_absolute_error(y_test, y_pred))

print("MSE :", mean_squared_error(y_test, y_pred))

print("RMSE :", np.sqrt(mean_squared_error(y_test, y_pred)))


print("\nIntercept")

print(model.intercept_)

print("\nCoefficients")

coef = pd.DataFrame(model.coef_,
                    X.columns,
                    columns=['Coefficient'])

print(coef)

plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")

plt.show()


plt.figure(figsize=(8,6))

sns.histplot(y_test-y_pred,
             bins=30,
             kde=True)

plt.title("Residual Distribution")

plt.show()


plt.figure(figsize=(8,6))

plt.scatter(df['GrLivArea'],
            df['SalePrice'])

plt.xlabel("Ground Living Area")
plt.ylabel("Sale Price")
plt.title("Area vs Sale Price")

plt.show()



print("\nPredicting New House Price")

sqft = float(input("Enter Square Footage : "))

bed = int(input("Enter Bedrooms : "))

bath = int(input("Enter Bathrooms : "))

new_house = [[sqft, bed, bath]]

prediction = model.predict(new_house)

print("\nEstimated House Price = ${:,.2f}".format(prediction[0]))