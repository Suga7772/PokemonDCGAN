import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image

# Define your generator architecture
class Generator(torch.nn.Module):
    def __init__(self, latent_dim, img_channels, feature_maps):
        super(Generator, self).__init__()
        self.net = torch.nn.Sequential(
            # Input: latent_dim x 1 x 1
            torch.nn.ConvTranspose2d(latent_dim, feature_maps * 8, kernel_size=4, stride=1, padding=0),
            torch.nn.BatchNorm2d(feature_maps * 8),
            torch.nn.ReLU(True),
            # feature_maps*8 x 4 x 4
            torch.nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(feature_maps * 4),
            torch.nn.ReLU(True),
            # feature_maps*4 x 8 x 8
            torch.nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(feature_maps * 2),
            torch.nn.ReLU(True),
            # feature_maps*2 x 16 x 16
            torch.nn.ConvTranspose2d(feature_maps * 2, img_channels, kernel_size=4, stride=2, padding=1),
            torch.nn.Tanh()
            # img_channels x 32 x 32
        )

    def forward(self, x):
        return self.net(x)

# Load the generator model
def load_generator(checkpoint_path, device):
    generator = Generator(latent_dim=100, img_channels=3, feature_maps=64)  # Adjust parameters as needed
    generator.load_state_dict(torch.load(checkpoint_path, map_location=device))
    generator.to(device)
    generator.eval()
    return generator

# Generate an image
def generate_image(generator, latent_dim, device):
    fixed_noise = torch.randn(1, latent_dim, 1, 1, device=device)  # Generate random noise
    fake_image = generator(fixed_noise).detach().cpu()  # Generate image
    fake_image = (fake_image + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
    fake_image = fake_image.squeeze(0).permute(1, 2, 0).numpy()  # Convert to HWC and NumPy
    return fake_image

# Streamlit app
def main():
    st.title("Pokémon Image Generator")
    st.write("Click the button to generate a new Pokémon image.")

    # Load the generator
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = "generator_pokemonDCGAN.pth"  # Path to your generator model
    generator = load_generator(checkpoint_path, device)

    # Button to generate a new image
    if st.button("Generate New Pokémon"):
        fake_image = generate_image(generator, latent_dim=100, device=device)
        st.image(fake_image, caption="Generated Pokémon", use_column_width=True)

if __name__ == "__main__":
    main()