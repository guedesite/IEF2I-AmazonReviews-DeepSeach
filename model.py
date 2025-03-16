import torch
import torch.nn as nn
import torch.onnx
import openvino as ov

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

def train_model(matrix_normalized, num_epochs=20):
    num_items = matrix_normalized.shape[1]
    model = Autoencoder(num_items)
    train_data = torch.FloatTensor(matrix_normalized)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        outputs = model(train_data)
        loss = criterion(outputs, train_data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    
    return model

def export_to_onnx(model, num_items, onnx_path="model.onnx"):
    dummy_input = torch.randn(1, num_items)
    torch.onnx.export(model, dummy_input, onnx_path)

def convert_to_openvino(onnx_path, openvino_path="model.xml"):
    # Note : Cette étape nécessite l'installation d'OpenVINO et une commande externe.
    # Exemple : mo --input_model model.onnx --output_dir .
    pass

# Exemple d'utilisation (à exécuter séparément) :
# from data_preparation import load_and_prepare_data
# user_item_matrix, matrix_normalized = load_and_prepare_data("path_to_amazon_reviews_2023.json")
# model = train_model(matrix_normalized)
# export_to_onnx(model, matrix_normalized.shape[1])
# convert_to_openvino("model.onnx")