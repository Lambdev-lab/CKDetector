#Analisis discriminante

"""            
 | |                   | |       | |            
 | |     __ _ _ __ ___ | |__   __| | _____   __ 
 | |    / _` | '_ ` _ \| '_ \ / _` |/ _ \ \ / / 
 | |___| (_| | | | | | | |_) | (_| |  __/\ V /  
 |______\__,_|_| |_| |_|_.__/ \__,_|\___| \_/
"""
#Benemerita Universidad Autonoma de Aguascalientes

#Raul Sanchez Vazquez
#Fernando Yael Ortega Guizar
#Karla Valeria Perez Perez
#Rogelio Robledo Moreno

from scipy.spatial import distance
import pandas as pd
import numpy as np
import math
import csv


def openCSV():
  #renalX.csv es la version ajustada del dataset
  data = pd.read_csv("renalX.csv")
  return data

def entrenamiento(data):
  ckdData = data.iloc[0:249,[0,1,2,3,4,9,10,11,12,13,14,15]]
  notckdData = data.iloc[250:399,[0,1,2,3,4,9,10,11,12,13,14,15]]
  covarianza = np.cov(ckdData,notckdData)
  Sp1 = np.linalg.inv(covarianza)
  ckdC = ckdData.mean()
  notckdC = notckdData.mean()
  return [ckdC,notckdC]

def prediccion(x,centroide):
  ckdD = distance.euclidean(x,centroide[0])
  notckdD = distance.euclidean(x,centroide[1])
  media = (ckdD+notckdD)/2
  if ckdD < notckdD:
    #Paciente ckd
    error = abs(media-ckdD)
    print("Paciente notckd con error de:", error)
  elif notckdD < ckdD:
    #Paciente notckd 
    error = abs(media-notckdD)
    print("Paciente notckd con error de:", error)

def main():
  data = openCSV()
  centroide = entrenamiento(data)
  #Es el vector de variables a introducir
  #[age, blood pressure,specific gravity,albumin,sugar,blood glucose random, blood urea,serum cretinine,sodium,potassium,hemoglobin,packed cell volume]
  x = [48,80,1.02,1,0,121,36,1.2,15.4,44,7800,5.2]
  prediccion(x,centroide)

main()