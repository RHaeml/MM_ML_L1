# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.metrics import mean_squared_error as MSE, r2_score as r2
import random
import matplotlib.colors as mcolors


def polypredict(ax, basefunction, samples_X=[], samples_Y=[], degree=2,figure=None,n_training_samples=None, show_predicted_targets=False, show_given_function=True):

    #if samples == [] or len(samples) < 10:
    #   X = [random.random()*2*np.pi for i in range(50)]
    #   if len(samples) < 10:
    #        print('"samples" abgelehnt. Bitte wähle zukünftig eine größere Anzahl Basisdaten (n>=10)\n')
    #else:
    #    X = samples
    #
    #
    #Y = [basefunction(element)*(1+(random.random()*0.1*(noise/100))) for element in X]
    X = samples_X
    Y = samples_Y  
    
    if n_training_samples == None:
        n_training_samples = round(len(X)/2)
    #else:   
    #    if len(X)-n_training_samples < 5:
    #        print('Zu hohe Anzahl an "training_targets". Die Anzahl der "training_targets" wurde auf die Hälfte des "sample"-Umfangs reduziert\n')
    #        n_training_samples = round(len(X)/2)

    
    training_features = np.array(X[:-(len(X)-n_training_samples)])[:,np.newaxis]
    training_targets = np.array(Y[:-(len(X)-n_training_samples)])[:,np.newaxis]
    testing_features = np.array(X[-(len(X)-n_training_samples):])[:,np.newaxis]
    testing_targets = np.array(Y[-(len(X)-n_training_samples):])[:,np.newaxis]
    
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    poly.fit(training_features)
    poly_training_features = poly.transform(training_features)
    poly_testing_features = poly.transform(testing_features)    
    #Daten der Polynomialisierung
    #poly_n_features_in = poly.n_features_in_
    #poly_n_features_out = poly.n_output_features_
    
    poly_regr = linear_model.LinearRegression()
    poly_regr.fit(poly_training_features, training_targets)    
    #Daten der Regression
    #n_regr_features = poly_regr.n_features_in_
    #regr_coef = poly_regr.coef_
    #intercept = poly_regr.intercept_
    print('{string:<25} {n_feat}'.format(string = 'Anzahl Regr. Merkmale', n_feat = poly_regr.n_features_in_))
    print('{string:<25} {coef}'.format(string = 'Regr. Koeffizienten', coef = poly_regr.coef_))
    print('{string:<25} {inter}'.format(string = 'Achsenabschnitt Regr.', inter = poly_regr.intercept_))
    
    predicted_targets = poly_regr.predict(poly_testing_features)
    #Daten der Vorhersagegüte    
    print('{string:<25} {MSE:.2f}'.format(string = 'MSE', MSE = MSE(testing_targets, predicted_targets)))
    print('{string:<25} {R2:.2f}\n'.format(string = 'R2', R2 = r2(testing_targets, predicted_targets)))
    
    #ax.set_xlim(-np.pi/2,np.pi*5/2)
    #ax.set_ylim(-1.5,1.5)
    
    #Wähle Farbe für Plot
    c_list = ['r', 'b', 'g', 'c', 'm', 'y']
    try:                 
        latest_c = figure.get_axes()[-1].get_lines()[-1].get_color()
        if c_list.index(latest_c) == len(c_list):
            c=c_list[0]
        else:
            c=c_list[c_list.index(latest_c)+1]
    except:
        c='r'

    lim = max(X)
    x_set = np.arange(0,lim,0.1)
     
    #Visualisierung der Trainingsdaten    
    if show_given_function == True:
        y_set = basefunction(x_set)
        ax.plot(x_set,y_set,label='true function',color='black',linewidth=1)
        ax.scatter(training_features,training_targets,label='training_targets',color='black',s=10)

    #Visualisierung der Vorhersagefunktion     
    x_set_array = np.array(x_set)[:,np.newaxis]
    poly_x_set_array = poly.transform(x_set_array)
    predicted_y_set = poly_regr.predict(poly_x_set_array)    
    ax.plot(x_set,predicted_y_set,label='degree '+str(degree),color=c,linewidth=1)    
        
    #Visualisierung der Testdaten
    if show_predicted_targets == True:
        ax.scatter(testing_features,testing_targets,label='testing_targets',color='blue',s=10)
        ax.scatter(testing_features,predicted_targets,label='predicted_targets',color=c,s=10)
       
    ax.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
    return ax

def R2MSEeval(maximum_degree, samples_X=[], samples_Y=[],):
    X = samples_X
    Y = samples_Y
    training_features = np.array(X[:-round(len(X)/2)])[:,np.newaxis]
    training_targets = np.array(Y[:-round(len(X)/2)])[:,np.newaxis]
    testing_features = np.array(X[-round(len(X)/2):])[:,np.newaxis]    
    testing_targets = np.array(Y[-round(len(X)/2):])[:,np.newaxis]
    
    MSE_training_data_prediction = []
    MSE_testing_data_prediction = []
    R2_training_data_prediction = []
    R2_testing_data_prediction = []
    dims = [x for x in range(1,maximum_degree+1)]
    
    for i in range(1,maximum_degree+1):
        poly = PolynomialFeatures(degree=i, include_bias=False)
        poly.fit(training_features)
        poly_training_features = poly.transform(training_features)
        poly_testing_features = poly.transform(testing_features)
        
        poly_regr = linear_model.LinearRegression()
        poly_regr.fit(poly_training_features, training_targets) 
        
        predicted_training_targets = poly_regr.predict(poly_training_features)
        predicted_testing_targets = poly_regr.predict(poly_testing_features)
        
        MSE_training_data_prediction.append(MSE(training_targets, predicted_training_targets))
        R2_training_data_prediction.append(r2(training_targets, predicted_training_targets))
        
        MSE_testing_data_prediction.append(MSE(testing_targets, predicted_testing_targets))       
        R2_testing_data_prediction.append(r2(testing_targets, predicted_testing_targets))
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
    #fig.suptitle('MSE and R2 evaluation')
    ax1.set_yscale('log')
    ax1.plot(dims, MSE_training_data_prediction,label='training_data',color=mcolors.CSS4_COLORS['cyan'])
    ax1.plot(dims, MSE_testing_data_prediction,label='testing_data',color=mcolors.CSS4_COLORS['lightgreen'])
    ax1.set_ylabel('MSE')
    ax1.set_xlabel('Dimensions')
    ax1.legend(bbox_to_anchor=(1, 1.02), borderaxespad=0, loc = 'lower right')
    
    #ax2.set_yscale('log')
    ax2.plot(dims, R2_training_data_prediction,label='training_data',color=mcolors.CSS4_COLORS['cyan'])
    ax2.plot(dims, R2_testing_data_prediction,label='testing_data',color=mcolors.CSS4_COLORS['lightgreen'])
    ax2.set_ylabel('R2')    
    ax2.set_xlabel('Dimensions')    
    ax2.legend(bbox_to_anchor=(1, 1.02), borderaxespad=0, loc = 'lower right') 

    
        
        
        