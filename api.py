from flask import Flask, request, jsonify
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/classify": {"origins": "http://localhost:3000"}})

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'OPTIONS, POST')
    return response

# Carregar modelo e rótulos globalmente
model_path = os.path.join(os.path.dirname(__file__), "C:\\Users\\x\\Downloads\\converted_keras (14)\\keras_model.h5")
model = load_model(model_path, compile=False)

labels_path = os.path.join(os.path.dirname(__file__), "C:\\Users\\x\\Downloads\\converted_keras (14)\\labels.txt")
class_names = [line.strip() for line in open(labels_path, "r").readlines()]

def preprocess_image(image_path):
    # Carregar imagem
    image = Image.open(image_path).convert("RGB")

    # Redimensionar e cortar a imagem
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # Converter imagem para matriz numpy
    image_array = np.asarray(image)

    # Normalizar a imagem
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Criar a matriz com a forma correta para alimentar o modelo Keras
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    return data

@app.route('/classify', methods=['POST'])
def classify_image():
    try:
        # Verificar se 'foto1', 'foto2', 'foto3' estão na solicitação
        if 'foto1' not in request.files:
            print("Nenhuma foto1 na requisição.")
            return jsonify({'error': 'Nenhuma foto1 encontrada'}), 400

        if 'foto2' not in request.files:
            print("Nenhuma foto2 na requisição.")
            return jsonify({'error': 'Nenhuma foto2 encontrada'}), 400

        if 'foto3' not in request.files:
            print("Nenhuma foto3 na requisição.")
            return jsonify({'error': 'Nenhuma foto3 encontrada'}), 400

        foto1 = request.files['foto1']
        foto2 = request.files['foto2']
        foto3 = request.files['foto3']

        # Verificar se os nomes dos arquivos estão vazios
        if foto1.filename == '':
            print("Nenhum arquivo selecionado para foto1.")
            return jsonify({'error': 'Nenhum arquivo selecionado para foto1'}), 400

        if foto2.filename == '':
            print("Nenhum arquivo selecionado para foto2.")
            return jsonify({'error': 'Nenhum arquivo selecionado para foto2'}), 400

        if foto3.filename == '':
            print("Nenhum arquivo selecionado para foto3.")
            return jsonify({'error': 'Nenhum arquivo selecionado para foto3'}), 400

        print(f"Arquivos recebidos: {foto1.filename}, {foto2.filename}, {foto3.filename}")

        try:
            # Pré-processamento das imagens
            data1 = preprocess_image(foto1)
            data2 = preprocess_image(foto2)
            data3 = preprocess_image(foto3)

            # Lógica para a previsão do modelo
            prediction1 = model.predict(data1)
            prediction2 = model.predict(data2)
            prediction3 = model.predict(data3)

            # Obter os índices das classes previstas
            index1 = np.argmax(prediction1)
            index2 = np.argmax(prediction2)
            index3 = np.argmax(prediction3)

            # Obter os nomes das classes correspondentes
            class_name1 = class_names[index1]
            class_name2 = class_names[index2]
            class_name3 = class_names[index3]

            # Obter as probabilidades das classes previstas
            score1 = prediction1[0][index1]
            score2 = prediction2[0][index2]
            score3 = prediction3[0][index3]

            print(f"Previsão 1: Classe - {class_name1}, Score - {score1}")
            print(f"Previsão 2: Classe - {class_name2}, Score - {score2}")
            print(f"Previsão 3: Classe - {class_name3}, Score - {score3}")

            result = {
                'success': True,
                'predictions': [
                    {'class_name': class_name1, 'score': float(score1)},
                    {'class_name': class_name2, 'score': float(score2)},
                    {'class_name': class_name3, 'score': float(score3)},
                ]
            }

            return jsonify(result)

        except Exception as e:
            return jsonify({'error': f'Erro durante o processamento: {str(e)}'}), 500

    except Exception as e:
        print(f'Erro geral: {str(e)}')
        return jsonify({'error': f'Erro geral: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
