# IEF2I-AmazonReviews-DeepSeach

Ce projet est un système de recommandation basé sur un autoencodeur entraîné avec PyTorch et optimisé avec OpenVINO pour une exécution sur un NPU Intel Core Ultra. Il utilise les données Amazon Reviews 2023 pour générer des recommandations personnalisées et les affiche via une interface web Flask.

## Structure du projet
```
/IEF2I-AmazonReviews-DeepSeach/
  ├── app.py              # Application Flask principale
  ├── data_preparation.py # Préparation des données à partir du JSON
  ├── model.py            # Définition et entraînement du modèle (version simple)
  ├── train.py            # Entraînement optimisé avec CUDA et exportation ONNX
  ├── templates/
  │   ├── index.html      # Page d'accueil avec sélection d'utilisateur
  │   └── results.html    # Page des recommandations avec métadonnées
  └── requirements.txt    # Dépendances du projet (à créer manuellement)
```

**Fichiers non inclus dans ce dépôt :**
- `amazon_reviews_2023.json` : Dataset source.
- `model.onnx` : Modèle exporté au format ONNX.
- `model.xml` et `model.bin` : Modèle converti au format OpenVINO IR.

---

## Prérequis

### Matériel
- Ordinateur avec au moins 32 Go de RAM.
- GPU NVIDIA avec CUDA (optionnel, pour accélérer l'entraînement).
- Intel Core Ultra avec NPU (pour l'inférence OpenVINO).

### Logiciels
- Python 3.8 ou supérieur.
- CUDA Toolkit (si GPU disponible, version compatible avec PyTorch, par exemple CUDA 11.7).
- OpenVINO Toolkit 2023.3 ou supérieur (pour conversion et inférence sur NPU).

---

## Installation

1. **Cloner le dépôt**
   ```bash
   git clone https://github.com/guedesite/IEF2I-AmazonReviews-DeepSeach.git
   cd IEF2I-AmazonReviews-DeepSeach
   ```

2. **Créer un environnement virtuel**
   ```bash
   python -m venv env
   source env/bin/activate  # Linux/Mac
   env\Scripts\activate     # Windows
   ```

3. **Installer les dépendances**
   Créez un fichier `requirements.txt` avec le contenu suivant :
   ```
   flask
   numpy
   pandas
   torch
   openvino
   openvino-dev
   ```
   Puis installez les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

4. **Installer OpenVINO Toolkit**
   - Téléchargez et installez OpenVINO depuis le [site officiel Intel](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html).
   - Ajoutez les scripts OpenVINO au PATH (Windows) :
     ```bash
     set PATH=%PATH%;C:\Program Files (x86)\Intel\openvino_2023.3\runtime\bin
     ```

---

## Téléchargement du dataset Amazon Reviews 2023

1. **Source officielle**
   - Le dataset Amazon Reviews 2023 est disponible sur [Amazon Customer Reviews Dataset](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/) (maintenu par UCSD).
   - Rendez-vous sur le site, sélectionnez une catégorie (ex. "Electronics"), et téléchargez le fichier JSON correspondant (par exemple, `All_Amazon_Review_2023.jsonl`).

2. **Placement**
   - Placez le fichier téléchargé dans le dossier `./IEF2I-AmazonReviews-DeepSeach/` et renommez-le en `amazon_reviews_2023.json`.
   - Assurez-vous que le fichier est au format JSONL (une ligne par review) avec au moins les colonnes : `user_id`, `asin`, `rating`, et idéalement `title`, `category`, `price`, `description`.

3. **Alternative**
   - Si vous ne trouvez pas le dataset exact, utilisez un ancien dataset Amazon Reviews (ex. 2018) disponible sur le même site, mais adaptez les noms de colonnes dans `data_preparation.py` si nécessaire.

---

## Entraînement du modèle

1. **Préparer les données**
   - Le script `train.py` charge et prétraite les données via `data_preparation.py`. Assurez-vous que `amazon_reviews_2023.json` est dans le dossier `./IEF2I-AmazonReviews-DeepSeach/`.

2. **Lancer l'entraînement**
   - Exécutez le script `train.py` pour entraîner l'autoencodeur avec CUDA (si disponible) :
     ```bash
     python train.py
     ```
   - **Paramètres ajustables** :
     - `max_users` : Nombre maximum d'utilisateurs (par défaut 50 000).
     - `max_items` : Nombre maximum d'articles (par défaut 50 000).
     - `max_rows` : Nombre maximum de lignes à lire dans le JSON (par défaut 1 000 000).
     - `batch_size` : Taille des lots pour l'entraînement (par défaut 64).
     - `num_epochs` : Nombre d'époques (par défaut 20).
   - Réduisez ces valeurs si vous manquez de mémoire (RAM ou VRAM GPU).

3. **Exportation en ONNX**
   - À la fin de l'entraînement, le modèle est exporté sous `model.onnx` dans le dossier `./IEF2I-AmazonReviews-DeepSeach/`.

---

## Conversion du modèle en format OpenVINO IR

1. **Vérifier la forme d'entrée**
   - Après l'entraînement, notez le nombre d'articles (`num_items`) affiché dans `Shape of user_item_matrix: (num_users, num_items)` dans la sortie de `train.py`. Par exemple, si `num_items = 50000`, la forme d'entrée sera `[1, 50000]`.

2. **Convertir avec Model Optimizer (mo)**
   - Utilisez la commande suivante pour convertir `model.onnx` en `model.xml` et `model.bin` :
     ```bash
     mo --input_model model.onnx --output_dir . --input_shape "[1,50000]"
     ```
   - Remplacez `50000` par le `num_items` réel de votre matrice.

3. **Résultat**
   - Les fichiers `model.xml` et `model.bin` seront générés dans le dossier `./IEF2I-AmazonReviews-DeepSeach/`.

4. **Alternative avec OpenVINO Model Converter (OVC)**
   - Si `mo` échoue ou si vous préférez la nouvelle méthode :
     ```python
     from openvino import convert_model, save_model
     ov_model = convert_model("model.onnx", input_shape=[1, 50000])  # Ajustez la forme
     save_model(ov_model, "model.xml")
     ```
   - Exécutez ce code dans un script Python séparé.

---

## Configuration de l'application Flask

1. **Placer les fichiers modèle**
   - Assurez-vous que `model.xml` et `model.bin` sont dans le dossier `./IEF2I-AmazonReviews-DeepSeach/` aux côtés de `app.py`.

2. **Vérifier le dataset**
   - Confirmez que `amazon_reviews_2023.json` est dans `./IEF2I-AmazonReviews-DeepSeach/`.

3. **Lancer l'application**
   - Depuis le dossier `./IEF2I-AmazonReviews-DeepSeach/` :
     ```bash
     python app.py
     ```
   - L'application démarre sur `http://127.0.0.1:5000/`.
   - Le mode debug est activé, mais le reloader est désactivé (`use_reloader=False`) pour éviter un double démarrage.

4. **Utilisation**
   - Ouvrez un navigateur à `http://127.0.0.1:5000/`.
   - Sélectionnez un `user_id` dans la liste déroulante (limitée à 1000 utilisateurs pour des raisons de performance).
   - Soumettez pour voir les recommandations avec métadonnées (ASIN, titre, catégorie, prix, description).

---

## Résolution des problèmes courants

### 1. Erreur de mémoire
- Si la matrice dépasse 32 Go de RAM :
  - Réduisez `max_users`, `max_items`, ou `max_rows` dans `train.py` et `app.py`.
  - Exemple : `max_users=10000`, `max_items=10000`, `max_rows=500000`.

### 2. Incompatibilité de forme d'entrée
- Si `app.py` échoue avec un message comme `Shape mismatch` :
  - Vérifiez `num_items` dans la sortie de `train.py`.
  - Reconverti `model.onnx` avec la bonne forme (`--input_shape "[1,num_items]"`).

### 3. NPU non détecté
- Si OpenVINO ne trouve pas le NPU :
  - Listez les dispositifs disponibles :
    ```python
    core = Core()
    print(core.available_devices)
    ```
  - Remplacez `"NPU"` par `"CPU"` dans `compiled_model = core.compile_model(model, "NPU")` pour tester sur CPU.

### 4. Dataset manquant ou mal formaté
- Si `amazon_reviews_2023.json` n'a pas les colonnes attendues (`user_id`, `asin`, `rating`), ajustez `data_preparation.py` pour correspondre aux colonnes disponibles.

---

## Déploiement en production
- Ne pas utiliser `app.run()` en production (serveur de développement uniquement).
- Utilisez Gunicorn :
  ```bash
  pip install gunicorn
  gunicorn -w 4 app:app
  ```

---

## Crédits
- Dataset : Amazon Customer Reviews Dataset (UCSD).
- Technologies : Flask, PyTorch, OpenVINO, Pandas, NumPy.

---

Ce README fournit toutes les étapes nécessaires pour configurer, entraîner, et exécuter votre projet. Adaptez les valeurs comme `num_items` ou les chemins selon votre environnement spécifique.