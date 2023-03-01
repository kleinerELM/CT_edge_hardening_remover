#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import scipy.optimize

def fit_function_tanh_exp(x, x0, b, c):
    return c + (0.5 * np.tanh((x-x0)/b) + 1.0) * np.exp((x-x0)/b)

def fit_function_tanh_exp_v2(x, x0, b, c):
    return c + (0.5 * np.tanh((x-x0)/b) + 1.0) * 3.0**(3.0**((x-x0)/b))

def fit_function_tanh_exp_v3(x, x0, b, c, d):
    return c + (0.5 * np.tanh((x-x0)/b) + 1.0) * d**(d**((x-x0)/b))

def fit_function_exp_v4(x, x0, b, c, d):
    return c + d**(d**((x-x0)/b))

class FitCurveProfile:
    def __init__(self, csv, fit_function):
        self.csv = csv
        self.df = pd.read_csv(csv, header=None)
        self.y_data = np.array(self.df.iloc[:, 0])
        self.x_data = np.arange(0, len(self.y_data))
        self.fit_function = fit_function

    # deletes all values after the maximum
    def filter_values(self):
        max_y_value = self.y_data.max()
        filtered_values = np.zeros(0)
        for i in range(0, len(self.y_data)):
            filtered_values = np.append(filtered_values, self.y_data[i])
            if ( self.y_data[i] == max_y_value ):
                break
        self.y_data = filtered_values
        self.x_data = np.arange(0, len(self.y_data))

    def fit_data(self, p0=(800, 400, 100)):
        self.params, self.cv = scipy.optimize.curve_fit(self.fit_function, self.x_data, self.y_data, p0, maxfev=10000)

    def fit_data_v2(self, p0=(1000, 700, 100, 1.5)):
        bounds = ((0, 0, 0, 1.0), (1e+8, 1e+5, 1e+4, 1.75) )
        self.params, self.cv = scipy.optimize.curve_fit(self.fit_function, self.x_data, self.y_data, p0, bounds=bounds, maxfev=10000)

    def plot_data(self):
        data = []
        for i in range(0, len(self.y_data)):
            fitted_y = 0
            if ( len(self.params) == 3 ):
                fitted_y = self.fit_function(self.x_data[i], self.params[0], self.params[1], self.params[2] )
            elif ( len(self.params) == 4 ):
                fitted_y = self.fit_function(self.x_data[i], self.params[0], self.params[1], self.params[2], self.params[3] )
            data.append([self.x_data[i], self.y_data[i], fitted_y])
        self.fit_df = pd.DataFrame(data, columns=['x', 'y', 'y_fit'])
        self.fit_df.plot(x='x', y=['y', 'y_fit'], title=self.csv)

if __name__ == "__main__":

    print('nothing to see here. Use in Jupyter notebook.')
    """
    do_4_param_fit = True

    if do_4_param_fit:
        ff = fit_function_exp_v4
    else:
        ff = fit_function_tanh_exp

    foo = FitCurveProfile('foo.csv', fit_function=ff)
    foo.filter_values()
    foo.fit_data_v2() if do_4_param_fit else foo.fit_data()
    foo.plot_data()
    print(foo.params)
    """