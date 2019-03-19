# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 22:20:53 2019

@author: tala
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("polynomial-regression.csv", sep = ";")

y = df.araba_max_hiz.values.reshape(-1,1)
x = df.araba_fiyat.values.reshape(-1,1)

plt.scatter(x,y)
plt.show()

# %% linear regression

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(x,y)

# %% predict 

y_head = lr.predict(x)

plt.plot(x,y_head, color = "red")
plt.show()

print("10m tl'lik araba hızı: ", lr.predict(10000))

# %% polynomial regression
# polynomial reg => y = b0 + b1*x + b2*x^2 + ... + bn*x^n

from sklearn.preprocessing import PolynomialFeatures
polynomial_regression = PolynomialFeatures(degree = 4) #4.dereceden poly

x_polynomial = polynomial_regression.fit_transform(x) # x^2 feature oluşturdu

# %% fit
linear_regression2 = LinearRegression()
linear_regression2.fit(x_polynomial,y)

# predict

y_head2 = linear_regression2.predict(x_polynomial)
plt.plot(x, y_head2, color = "orange", label="poly")
plt.legend()
plt.show()

