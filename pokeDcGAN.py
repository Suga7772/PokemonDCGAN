import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

# Define the Generator class

class Generator(nn.Module):
    def __init__(self, latent_dim, img_channels, feature_maps):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            # Input: latent_dim x 1 x 1
            nn.ConvTranspose2d(latent_dim, feature_maps * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(True),
            
            # feature_maps*8 x 4 x 4
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),
            
            # feature_maps*4 x 8 x 8
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            
            # feature_maps*2 x 16 x 16
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),
            
            # feature_maps x 32 x 32
            nn.ConvTranspose2d(feature_maps, img_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            # img_channels x 64 x 64
        )

    def forward(self, x):
        return self.net(x)

# Load the generator model
def load_generator(checkpoint_path, device):
    latent_dim=128

    generator = nn.Sequential(
    nn.ConvTranspose2d(latent_dim,512,kernel_size=4,stride=1,padding=0,bias = False),
    nn.BatchNorm2d(512),
    nn.ReLU(True),
    
    nn.ConvTranspose2d(512,256,kernel_size=4,stride=2,padding=1,bias = False),
    nn.BatchNorm2d(256),
    nn.ReLU(True),
    
    nn.ConvTranspose2d(256,128,kernel_size=4,stride=2,padding=1,bias = False),
    nn.BatchNorm2d(128),
    nn.ReLU(True),
    
    nn.ConvTranspose2d(128,64,kernel_size=4,stride=2,padding=1,bias = False),
    nn.BatchNorm2d(64),
    nn.ReLU(True),
    
    nn.ConvTranspose2d(64,3,kernel_size=4,stride=2,padding=1,bias = False),
    nn.Tanh()
)
    generate = generator(latent_dim=100, img_channels=3, feature_maps=64)  # Adjust parameters as needed
    generate.load_state_dict(torch.load(checkpoint_path, map_location=device))
    generate.to(device)
    generate.eval()
    return generate

# Generate an image
def generate_image(generator, latent_dim, device):
    # Generate a random latent vector
    fixed_noise = torch.randn(1, latent_dim, 1, 1, device=device)  # Generate random noise on the correct device
    
    # Generate an image
    fake_image = generator(fixed_noise).detach().cpu()  # Move to CPU and detach from computation graph
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
    if st.button("Generate New Fakey Pokémon"):
        fake_image = generate_image(generator, latent_dim=100, device=device)
        st.image(fake_image, caption="Generated Pokémon", use_column_width=True)

if __name__ == "__main__":
    main()