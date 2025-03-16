import pandas as pd
import numpy as np

# 1. Fonction pour charger et préparer les données avec limitation de mémoire
def load_and_prepare_data(file_path, max_users=5000, max_items=5000, max_rows=1000000):
    """
    Charge et prépare les données en limitant le nombre d'utilisateurs et d'articles pour respecter les 32 Go de RAM.
    Args:
        file_path (str): Chemin vers le fichier JSON.
        max_users (int): Nombre maximum d'utilisateurs à charger.
        max_items (int): Nombre maximum d'articles à charger.
        max_rows (int): Nombre maximum de lignes à lire dans le JSON.
    Returns:
        user_item_matrix (pd.DataFrame): Matrice utilisateur-article.
        matrix_normalized (np.ndarray): Matrice normalisée.
    """
    # Charger un sous-ensemble du JSON
    print("Chargement des données...")
    data = pd.read_json(file_path, lines=True, nrows=max_rows)
    interactions = data[['user_id', 'asin', 'rating']]

    # Garder les métadonnées des articles (on suppose que le JSON contient ces colonnes)
    # Par exemple : title, category, price, description, etc.
    # On prend les colonnes pertinentes et on supprime les duplicatas pour chaque asin
    metadata_columns = ['asin', 'title', 'category', 'price', 'description']  # Ajustez selon votre dataset
    available_columns = [col for col in metadata_columns if col in data.columns]
    item_metadata = data[available_columns].drop_duplicates(subset='asin').set_index('asin')

    # Réduire la taille en sélectionnant les utilisateurs et articles les plus fréquents
    top_users = interactions['user_id'].value_counts().head(max_users).index
    top_items = interactions['asin'].value_counts().head(max_items).index
    interactions_subset = interactions[
        interactions['user_id'].isin(top_users) & 
        interactions['asin'].isin(top_items)
    ]

    # Gérer les duplicatas en prenant la moyenne des ratings
    interactions_subset = interactions_subset.groupby(['user_id', 'asin']).agg({'rating': 'mean'}).reset_index()

    # Créer la matrice utilisateur-article
    user_item_matrix = interactions_subset.pivot(index='user_id', columns='asin', values='rating').fillna(0)

    # Normaliser entre 0 et 1 (supposant des ratings de 1 à 5)
    matrix_normalized = user_item_matrix.values / 5.0

    # Estimation de la mémoire utilisée
    mem_usage = user_item_matrix.memory_usage(deep=True).sum() / 1024**3  # En Go
    print(f"Mémoire utilisée par la matrice utilisateur-article : {mem_usage:.2f} Go")
    if mem_usage > 32:
        raise MemoryError("La matrice dépasse 32 Go de RAM. Réduisez max_users ou max_items.")

    return user_item_matrix, matrix_normalized, item_metadata

# Example usage
if __name__ == "__main__":
    file_path = "./amazon_reviews_2023.json"
    user_item_matrix, matrix_normalized = load_and_prepare_data(file_path)
    print(f"Matrix shape: {user_item_matrix.shape}")
    print(f"Memory usage: {user_item_matrix.memory_usage().sum() / 1024**2:.2f} MB")