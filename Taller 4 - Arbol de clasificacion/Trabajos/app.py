from flask import Flask, render_template, request
import joblib
import os
import numpy as np

app = Flask(__name__)

#Obtén el directorio actual
directorio_actual = os.path.dirname(os.path.abspath(__file__))

#Construye la ruta completa al archivo del modelo
ruta_modelo = os.path.join(directorio_actual, 'models', 'modelo_arbol.pkl')

#Cargar el modelo utilizando la ruta completa
model = joblib.load(ruta_modelo)

# Definir la ruta principal del sitio web
@app.route('/')
def index():
    return render_template('index.html')  # Renderizar la plantilla 'index.html'

# Definir la ruta para realizar la predicción
@app.route('/predict', methods=['POST'])
def predict():
    # Obtener los valores del formulario enviado
    N = int(request.form['N'])
    P = int(request.form['P'])
    K = int(request.form['K'])
    temperatura = int(request.form['temperatura'])
    humedad = int(request.form['humedad'])
    ph = int(request.form['ph'])
    precipitacion = int(request.form['precipitacion'])
    
    pred_probabilities = np.array([[N, P, K, temperatura, humedad, ph, precipitacion]])
    
    prediccion = model.predict(pred_probabilities)

    mensaje = f"La semilla perfecta para sembras en su terreno es: {prediccion}"

    # Renderizar la plantilla 'result.html' y pasar el mensaje a la plantilla
    return render_template('result.html', pred=mensaje)

# Iniciar la aplicación si este script es el punto de entrada
if __name__ == '__main__':
    app.run()  # Iniciar la aplicación Flask