import streamlit as st
from PIL import Image
import os
import random
import torch
import torch.nn as nn

latent_dim = 128  # latent dimension
img_channels = 3  # number of image channels
feature_maps = 64  # number of feature maps
folder_path = "images"  # generating images path

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

def get_image_files(folder_path):
    # Get a list of all image files in the folder
    return [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

def display_generated_image(folder_path, image_files):
    if image_files:
        # Select a random image
        random_image = random.choice(image_files)
        image_path = os.path.join(folder_path, random_image)
        try:
            image = Image.open(image_path)
            # stling the position and overall appearance of the image
            st.markdown(
                f"""
                <div style="display: flex; justify-content: center; align-items: center; padding: 20px;">
                    <img src="data:image/png;base64,{image_to_base64(image)}" width="64" height="64">
                </div>
                """,
                unsafe_allow_html=True
            )
        except Exception as e:
            st.error(f"Error loading image {random_image}: {e}")
    else:
        st.write("No images found in the folder.")

def image_to_base64(image):
    # converting to base64 as guidec in the streamlit documentation for parsing 64x64 images
    from io import BytesIO
    import base64
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# app entry point
def main():
    st.title("Pokémon DCGAN Generator")

    # Get the list of image files
    image_files = display_generated_image(folder_path, get_image_files(folder_path))

    # Initializing session state to keep track of the current image
    if 'current_image' not in st.session_state:
        st.session_state.current_image = None

    # Button to generate pokemon image
    if st.button("Generate New Pokémon"):
        st.session_state.current_image = random.choice(image_files) if image_files else None

    # displaying the current image
    if st.session_state.current_image:
        image_path = os.path.join(folder_path, st.session_state.current_image)
        try:
            image = Image.open(image_path)
            # Display the image at its native resolution (64x64) without a caption
            st.image(image, width=64)  # Set width to 64 pixels for clarity
        except Exception as e:
            st.error(f"Error loading image {st.session_state.current_image}: {e}")
    else:
        st.write("Click the button to generate super one of a kind Cool")

if __name__ == "__main__":
    main()
