Pokémon DCGAN Model 🎨⚡
Welcome to the Pokémon DCGAN (Deep Convolutional Generative Adversarial Network) repository! This project showcases a generative model trained to create new, unique Pokémon-like images using a DCGAN architecture. The model was trained on a dataset of over 29,000 Pokémon images and deployed via Streamlit for real-time image generation. Dive into the world of AI-generated Pokémon and explore how deep learning can bring fictional creatures to life!

Generated Pokémon Example  : 

<p align="center">
  <img src="!https://github.com/user-attachments/assets/30536df2-a8ff-47f1-afa3-3dcc8ecc697e" alt="Generated Pokémon Example" />
</p>


![generatedPokemon-images-0049](https://github.com/user-attachments/assets/30536df2-a8ff-47f1-afa3-3dcc8ecc697e)

https://media3.giphy.com/media/v1.Y2lkPTc5MGI3NjExMjM0dnNydXppZWh5YjJjOXY2eHFwczRydHI0YmRyMzRhbnh1dWczNSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/6nWhy3ulBL7GSCvKw6/giphy.gif


📖 Table of Contents
Introduction

Features

Dataset

Methodology

Installation

Usage

Results

Deployment

Contributing

License

Acknowledgments

🌟 Introduction
This project explores the capabilities of Generative Adversarial Networks (GANs) to generate synthetic images of Pokémon faces that do not exist in the original dataset. The model leverages a DCGAN architecture and was trained for 450 epochs on a curated dataset of Pokémon images. The final model was deployed using Streamlit, allowing users to interactively generate and explore new Pokémon faces.

✨ Features
DCGAN Architecture: Utilizes a Deep Convolutional GAN to generate high-quality Pokémon images.

Large Dataset: Trained on over 29,000 Pokémon images from multiple sources.

Streamlit Deployment: Interactive web interface for real-time image generation.

Training Flexibility: Supports both DCGAN and progressive GAN (pGAN) architectures.

Efficient Preprocessing: Custom data loading and preprocessing pipeline.

📊 Dataset
The model was trained on a combination of three datasets:

Pokemon_V2 Dataset (Kaggle)

9,002 images from the Kaggle hub dataset.

Link: Pokemon_V2 Dataset

Pokemon Generation One (Kaggle)

20,100 images of Generation 1 Pokémon.

Link: Pokemon Generation One

Few-Shot Classification Dataset (GitHub)

Curated subset of Pokémon images.

Link: Few-Shot Classification Dataset

🛠 Methodology
GAN Architecture
The model uses a DCGAN architecture with the following components:

Generator
Input: 100-dimensional noise vector.

Layers:

Dense layer → Reshape → 4 Transposed Convolutional Blocks (Conv2DTranspose → BatchNorm → ReLU).

Final layer: Conv2DTranspose with 3 filters and tanh activation (output: 64x64x3 image).

Discriminator
Input: 64x64x3 RGB image.

Layers:

5 Convolutional Blocks (Conv2D → BatchNorm → LeakyReLU).

Final layer: Dense layer with sigmoid activation (real/fake classification).

Training
Epochs: 450

Hardware: Google Colab

Software: TensorFlow, PyTorch (for pGAN)

🚀 Installation
To get started, clone this repository and install the required dependencies.

bash
Copy
# Clone the repository
git clone https://github.com/Suga7772/PokemonDCGAN.git
cd PokemonDCGAN

# Install dependencies
pip install -r requirements.txt
🖥️ Usage
Training the Model
To train the DCGAN model on your own dataset or the provided Pokémon dataset, run:

bash
Copy
python train.py --dataset path/to/pokemon/dataset --epochs 450 --batch_size 32
Generating Pokémon Images
After training, you can generate new Pokémon images using the trained model:

bash
Copy
python generate.py --model path/to/trained/model --output_dir generated_images
📊 Results
Training Progress
Intermediate Results: Generated images after 50 epochs.

Final Results: High-quality Pokémon-like images generated after 450 epochs.

Epoch 50	Epoch 450
Epoch 50	Epoch 450
Note: Replace the placeholder images with actual generated images from your model.

🌐 Deployment
The model is deployed using Streamlit, providing a user-friendly interface for real-time Pokémon generation.

Live Website
Explore the live deployment here:
👉 Pokémon DCGAN Streamlit App

Features
Real-Time Generation: Instantly generate new Pokémon faces.

Minimalistic UI: Simple and intuitive interface.

Efficient Caching: Uses Streamlit's @st.cache for fast model loading.

🤝 Contributing
Contributions are welcome! If you'd like to improve this project, please follow these steps:

Fork the repository.

Create a new branch (git checkout -b feature/YourFeature).

Commit your changes (git commit -m 'Add some feature').

Push to the branch (git push origin feature/YourFeature).

Open a pull request.

📜 License
This project is licensed under the MIT License. See the LICENSE file for details.

🙏 Acknowledgments
The Pokémon Company for creating the amazing Pokémon universe.

Kaggle and GitHub for providing the datasets.

TensorFlow, PyTorch, and Streamlit for the tools and frameworks.

The open-source community for their invaluable contributions.

 ____________________________________________________________________________________________
🔗 Links
GitHub Repository

Live Streamlit App

Dataset Links

Happy generating! 🎉✨

Customization Tips:
Images: Replace placeholder images with actual screenshots or generated Pokémon images.

Training Logs: Add more details or visualizations of training progress if available.

Badges: Add badges for build status, license, etc., if applicable.

Let me know if you need further assistance! 🚀


<!-- Replace with an actual image of your generated Pokémon -->
