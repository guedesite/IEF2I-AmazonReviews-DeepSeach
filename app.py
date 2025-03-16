from flask import Flask, request, render_template
import numpy as np
from openvino import Core
from data_preparation import load_and_prepare_data
import os

app = Flask(__name__)

# Charger les données
# Paramètres pour limiter la mémoire
max_users = 50000  # Réduire si nécessaire
max_items = 50000  # Réduire si nécessaire
max_rows =  1000000  # Réduire si le fichier JSON est trop gros

# Charger et préparer les données
user_item_matrix, _, item_metadata = load_and_prepare_data(
    "amazon_reviews_2023.json", max_users=max_users, max_items=max_items, max_rows=max_rows
)
print(f"Shape of user_item_matrix: {user_item_matrix.shape}")

item_indices_to_asin = user_item_matrix.columns.tolist()

user_ids = user_item_matrix.index.tolist()

# Vérifier la forme des données
num_items = user_item_matrix.shape[1]
print(f"Nombre d'articles dans user_item_matrix: {num_items}")

# Charger et compiler le modèle OpenVINO
core = Core()

# Vérifier que les fichiers modèle existent
model_xml_path = "model.xml"
model_bin_path = "model.bin"
if not os.path.exists(model_xml_path) or not os.path.exists(model_bin_path):
    raise FileNotFoundError(f"Model files not found: {model_xml_path} and {model_bin_path}")

# Charger le modèle
try:
    model = core.read_model(model=model_xml_path, weights=model_bin_path)
except Exception as e:
    raise RuntimeError(f"Failed to read the model: {e}")

# Compiler le modèle pour le NPU
try:
    compiled_model = core.compile_model(model, "NPU")
    infer_request = compiled_model.create_infer_request()
except Exception as e:
    raise RuntimeError(f"Failed to compile the model on NPU: {e}")

# Vérifier les noms et formes des entrées/sorties
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)
input_name = input_layer.any_name
expected_input_shape = input_layer.shape
print(f"Input layer name: {input_name}, Expected input shape: {expected_input_shape}")

# Vérifier que la forme des données correspond à celle attendue par le modèle
if expected_input_shape[1] != num_items:
    raise ValueError(f"Shape mismatch: Model expects input shape {expected_input_shape}, but user_item_matrix has {num_items} items")

def get_recommendations(user_id):
    if user_id not in user_ids:
        return None
    user_idx = user_ids.index(user_id)
    user_data = user_item_matrix.iloc[user_idx].values / 5.0
    input_tensor = np.array([user_data], dtype=np.float32)  # Shape: (1, num_items)
    
    try:
        infer_request.infer(inputs={input_name: input_tensor})
        output = infer_request.get_output_tensor(0).data
        top_item_indices = np.argsort(output[0])[-5:][::-1]  # Top 5 indices
        
        # Convertir les indices en ASIN et récupérer les métadonnées
        recommended_items = []
        for idx in top_item_indices:
            asin = item_indices_to_asin[idx]  # Convertir l'index en ASIN
            item_info = item_metadata.loc[asin].to_dict()  # Récupérer les métadonnées
            item_info['asin'] = asin  # Ajouter l'ASIN au dictionnaire
            print(item_info);
            recommended_items.append(item_info)
        return recommended_items
    except Exception as e:
        print(f"Error during inference: {e}")
        return None

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_id = request.form['user_id']
        recommendations = get_recommendations(user_id)
        if recommendations:
            return render_template('results.html', user_id=user_id, recommendations=recommendations)
        return "User not found", 404
    return render_template('index.html', user_ids=user_ids[:1000])

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)  # Désactive le reloader