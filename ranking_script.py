# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 18:40:18 2020

@author: ValentinC
"""

import pandas as pd

from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import requests
import numpy as np
import operator

def main():
    errors = ''

    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    
    countries_population_df = pd.read_csv("countries_population.csv", header=None)
    countries_population = {}    
        
    for index, row in countries_population_df.iterrows():
        countries_population[row[0]] = row[2]
    
    url = "https://api.covid19api.com/"
    countryUrl = url + "countries"
    byCountryUrl = url + "total/country/{0}"
    
    payload = {}
    headers= {}
    
    response = requests.request("GET", countryUrl, headers=headers, data = payload)
    
    country_ranking = {}
    
    df = pd.DataFrame(response.json())
    
    for index, row in df.iterrows():
        try:           
            print(row['Slug'])
        
            response = requests.request("GET", byCountryUrl.format(row['Slug']), headers=headers, data = payload)
            responseArray = np.array(response.json())
            
            if responseArray.size > 0: 
                df = pd.DataFrame(response.json())
                df.to_csv (row['Slug'] + '.csv', index = None)
                
                data = pd.read_csv(row['Slug'] + '.csv',header=None)
                
                T = data.iloc[1:,7:10]
                X = data.iloc[1:,0:1]
                for i in range(1, len(T)+1):
                    X.loc[i, 0] = i
                
                predict = pd.DataFrame()
                for i in range(1, 11):
                    predict.loc[i, 0] = len(T)+i
                
                net = MLPRegressor(hidden_layer_sizes=(10, 10), activation='relu', solver='lbfgs') #perfect
                
                net.fit(X, T)
                predictedData = net.predict(predict)

                if row['Slug'] == 'romania' or row['Slug'] == 'united-states' or row['Slug'] == 'italy':
                    realDataNumber = len(data.iloc[1:,7])
                    t1 = np.arange(realDataNumber)
                    t2 = np.arange(realDataNumber, realDataNumber + len(predictedData))
                    plt.figure()
                    dataForPlot = np.array(data.iloc[1:,7])
                    plt.plot(t1, dataForPlot.astype(np.int),'.-b', label='real data')
                    plt.plot(t2, predictedData[:,0],'.-r', label='predicted data')
                    plt.title("Real and Predicted data for " + row['Country'])
                    plt.xlabel('days')
                    plt.ylabel('number of cases')
                    plt.legend()


                if countries_population.get(row['Country']) > 0:
                    score = round(round(predictedData[9][0], 3) / countries_population.get(row['Country']) * 100, 4)
                    country_ranking[row['Country']] = score
            
        except Exception as e:
            errors = errors + str(e)
    print(errors)
                    
    print(country_ranking)
    sorted_country_ranking = sorted(country_ranking.items(), key=operator.itemgetter(1))
    print(sorted_country_ranking)

    gdp_and_percentage_services = pd.read_csv("gdp_and_percentage_services.csv", header=None)
    countries_gdp_services = gdp_and_percentage_services.iloc[:,0:3]
    print(countries_gdp_services)
   
    country_gdp_ranking = {};
    for index, row in countries_gdp_services.iterrows():
        try:
            if isNotNaN(row[1]) and isNotNaN(row[2]):
                score = float(row[1]) * 1/100 * float(row[2])
                country_gdp_ranking[row[0]] = score
                sorted_country_ranking[row[0]] = score
        except Exception as e:
            errors = errors + str(e)

    print(country_gdp_ranking)
    sorted_country_gdp_ranking = sorted(country_gdp_ranking.items(), key=operator.itemgetter(1))
    
    
#    print(sorted_country_ranking)

main()

def isNotNaN(num):
    return num == num

def is_float(value):
  try:
    float(value)
    return True
  except:
    return False 
