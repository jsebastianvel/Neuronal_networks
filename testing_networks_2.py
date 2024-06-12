# -*- coding: utf-8 -*-
"""
Created on Fri May 17 10:14:44 2024

@author: OMEN Laptop
"""

import network2_mod
import os
import numpy as np
from tensorflow.keras.utils import to_categorical
import pickle

directorio = 'C:/Users/OMEN Laptop/Peccala'

#'input_1.npy', 'input_3.npy','input_4.npy','input_5.npy','input_6.npy','input_7.npy','input_8.npy',
# Lista de nombres de archivos NPY
nombres_archivos_input = ['input_1.npy','input_2.npy','input_3.npy','input_4.npy','input_5.npy','input_6.npy','input_7.npy','input_8.npy','input_9.npy']  # y así sucesivamente
nombres_archivos_output = ['output_1.npy','output_2.npy', 'output_3.npy','output_4.npy','output_5.npy','output_6.npy','output_7.npy','output_8.npy','output_9.npy']  # y así sucesivamente

#transformar a vector
validation_input= np.load(os.path.join(directorio,"input_0.npy"))
#validation_input = (validation_input + 1)
validation_input=validation_input.reshape(8761,48)
validation_input = [np.reshape(x, (48,1)) for x in validation_input]

validation_output= np.load(os.path.join(directorio,"output_0.npy"))
validation_output = to_categorical(validation_output + 1)
#validation_output=validation_output+1
validation_output = np.expand_dims(validation_output, axis=1)
validation_output = list(validation_output)
#validation_output.reshape(8761,3,1) 
# Inicializa una lista para almacenar los datos cargados
datos_input = []
datos_output = []


datos_input_concat=np.empty([48,8761])

# Carga cada archivo NPY y añádelo a la lista de datos
#for nombre_archivo in nombres_archivos_input:
for count, nombre_archivo in enumerate(nombres_archivos_input):
    ruta_archivo = os.path.join(directorio, nombre_archivo)
    data = np.load(ruta_archivo)
    data = np.nan_to_num(data)
    #data= data.astype(int)
    #datos_input.append(data)
    if count==0:
        datos_input_concat=data
    else:    
        datos_input_concat=np.concatenate((datos_input_concat,data),axis=1)
    
datos_input_concat= np.transpose(datos_input_concat)

datos_output_concat=np.array([])

for nombre_archivo in nombres_archivos_output:
    ruta_archivo = os.path.join(directorio, nombre_archivo)
    data = np.load(ruta_archivo)
    data = np.nan_to_num(data)
    #data= data.astype(int)
    #datos_output.append(data)
    datos_output_concat=np.concatenate((datos_output_concat,data))

training_input=datos_input_concat
#training_input = np.array(training_input)
#training_input= (training_input+1)
#training_input =training_input.reshape(8761*9,48)
training_input = [np.reshape(x, (48,1)) for x in training_input]

training_output=datos_output_concat
training_output= np.array(training_output)
training_output = to_categorical(training_output + 1)
#training_output= training_output+1
training_output= np.array(training_output)
#training_output = training_output.reshape(8761*9,3,1)
training_output = np.expand_dims(training_output, axis=2)
training_output=list(training_output)

training_data=zip(training_input,training_output)
validation_data=zip(validation_input,validation_output)

#training_data_list=list(training_data) #para comprobar que la dimensionalidad correcto
#validation_data_list=list(validation_data) #para comprobar que la dimensionalidad correcto

architecture=[48,10,3]
#primera red
net = network2_mod.Network(architecture, cost=network2_mod.CrossEntropyCost,activation_function=0)
#net.large_weight_initializer()
epochs=30
net.SGD(training_data, epochs, 10, 0.1, lmbda = 5.0,evaluation_data=validation_data,
    monitor_evaluation_accuracy=True)

architecture_str = '_'.join(map(str, architecture))
with open(f'net_object_{architecture_str}.pkl', 'wb') as file:
    pickle.dump(net, file)

print("Objeto 'net' guardado exitosamente en f'net_object_{architecture_str}.pkl'.")

"""
with open(f'resultados_red_neuronal_{architecture_str}.txt', 'w') as file:
    original_stdout = sys.stdout  # Guardar la salida estándar original
    sys.stdout = file  # Redirigir la salida estándar al archivo
print(f"Los resultados de la red neuronal se han guardado en f'resultados_red_neuronal_{architecture_str}.txt'.")
"""