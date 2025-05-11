from sklearn.datasets import load_digits
from sklearn.preprocessing import MinMaxScaler
import torch
from digits_autoencoder import DigitsAutoencoder

digits = load_digits()
X = MinMaxScaler().fit_transform(digits.data)

input_dim = X.shape[1]
hidden_dim = 16

model = DigitsAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.MSELoss()

X_tensor = torch.tensor(X, dtype=torch.float32)
for epoch in range(500):
    optimizer.zero_grad()
    reconstructed = model(X_tensor)
    loss = loss_fn(reconstructed, X_tensor)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss {loss.item():.4f}")

torch.save(model.state_dict(), "digits_encoder.pt")
print("✅ Encoder saved.")