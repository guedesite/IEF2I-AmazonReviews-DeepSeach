import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from data_preparation import load_and_prepare_data

# 2. Définir le modèle d'autoencodeur
class Autoencoder(nn.Module):
    def __init__(self, num_items):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_items, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, num_items),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# 3. Entraîner le modèle avec CUDA et traitement par lots
def train_model(matrix_normalized, batch_size=64, num_epochs=20):
    """
    Entraîne le modèle d'autoencodeur avec CUDA et traitement par lots.
    Args:
        matrix_normalized (np.ndarray): Matrice normalisée.
        batch_size (int): Taille des lots.
        num_epochs (int): Nombre d'époques.
    Returns:
        model (Autoencoder): Modèle entraîné.
    """
    # Vérifier la disponibilité de CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilisation de : {device}")

    # Convertir la matrice en tenseur
    train_data = torch.FloatTensor(matrix_normalized)

    # Créer un DataLoader pour le traitement par lots
    dataset = TensorDataset(train_data, train_data)  # Entrées et cibles identiques pour l'autoencodeur
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialiser le modèle
    num_items = matrix_normalized.shape[1]
    model = Autoencoder(num_items).to(device)

    # Définir la perte et l'optimiseur
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Boucle d'entraînement
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_inputs, batch_targets in dataloader:
            batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)

            # Passe avant
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_targets)

            # Passe arrière et optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Avg Loss: {avg_loss:.4f}")

    return model

# 4. Exporter le modèle vers ONNX
def export_to_onnx(model, num_items, device, onnx_path="model.onnx"):
    model.eval()
    dummy_input = torch.randn(1, num_items, device=device)
    torch.onnx.export(model, dummy_input, onnx_path,
                      input_names=['input'], output_names=['output'])
    print(f"Modèle exporté sous : {onnx_path}")

# 5. Fonction principale
def main():
    # Paramètres pour limiter la mémoire
    max_users = 50000  # Réduire si nécessaire
    max_items = 50000  # Réduire si nécessaire
    max_rows =  1000000  # Réduire si le fichier JSON est trop gros

    # Charger et préparer les données
    user_item_matrix, matrix_normalized = load_and_prepare_data(
        "amazon_reviews_2023.json", max_users=max_users, max_items=max_items, max_rows=max_rows
    )
    print(f"Shape of user_item_matrix: {user_item_matrix.shape}")

    # Entraîner le modèle
    batch_size = 64
    num_epochs = 20
    model = train_model(matrix_normalized, batch_size=batch_size, num_epochs=num_epochs)

    # Exporter le modèle
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    export_to_onnx(model, matrix_normalized.shape[1], device)

if __name__ == "__main__":
    main()

# after model trained, convert it to json with: mo --input_model model.onnx --output_dir . --input_shape "[1,31417]"