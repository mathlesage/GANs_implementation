import torch
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from model_cond import Generator
from utils import load_model


def show_images(real_images, generated_images, num_images=10):
    """
    Display real and generated images side by side.
    
    Args:
        real_images: Real MNIST images
        generated_images: Generated images from GAN
        num_images: Number of images to display
    """
    fig, axes = plt.subplots(2, num_images, figsize=(20, 4))
    
    for i in range(num_images):
        # Real images (top row)
        real_img = real_images[i].squeeze().cpu().numpy()
        axes[0, i].imshow(real_img, cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Real Images', fontsize=14, fontweight='bold')
        
        # Generated images (bottom row)
        gen_img = generated_images[i].squeeze().cpu().numpy()
        axes[1, i].imshow(gen_img, cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Generated Images', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('image_test/comparison.png', dpi=150, bbox_inches='tight')
    print("\n✓ Comparison saved to 'comparison.png'")
    plt.show()


def main():

    
    # Device setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using device: MPS")
    else:
        device = torch.device("cpu")
        print(f"Using device: CPU")
    
    print("\n" + "="*60)
    print("LOADING MODELS AND DATA")
    print("="*60)
    
    # Load real MNIST data
    print("\n1. Loading real MNIST images...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])
    
    data_path = os.getenv('DATA', 'data')
    dataset = datasets.MNIST(root=data_path, train=True, transform=transform, download=True)
    
    # Sample random real images
    indices = np.random.choice(len(dataset), 10, replace=False)
    real_images = []
    for idx in indices:
        img, label = dataset[idx]
        real_images.append(img)
    real_images = torch.stack(real_images)
    
    print(f"   ✓ Loaded {10} real images")
    
    # Load generator
    print("\n2. Loading trained generator...")
    mnist_dim = 784
    generator = Generator(g_output_dim=mnist_dim).to(device)
    generator = load_model(generator, 'checkpoints', device)
    generator.eval()
    print("   ✓ Generator loaded successfully")
    
    # Generate images
    print(f"\n3. Generating {10} images...")
    with torch.no_grad():
        z = torch.randn(10, 100, device=device)
        generated_images = generator(z)
        generated_images = generated_images.reshape(10, 1, 28, 28)
    
    print(f"   ✓ Generated {10} images")
    
    # Display
    print("\n" + "="*60)
    print("DISPLAYING COMPARISON")
    print("="*60)
    show_images(real_images, generated_images.cpu(), 10)
    
    # Also save individual grids
    print("\nSaving additional visualizations...")
    
    # Save grid of real images
    torchvision.utils.save_image(
        real_images,
        'image_test/real_grid.png',
        nrow=10,
        normalize=True,
        padding=2
    )
    print("   ✓ Saved 'real_grid.png'")
    
    # Save grid of generated images
    torchvision.utils.save_image(
        generated_images.cpu(),
        'image_test/generated_grid.png',
        nrow=10,
        normalize=True,
        padding=2
    )
    print("   ✓ Saved 'generated_grid.png'")
    
    print("\n" + "="*60)
    print("✓ DONE!")
    print("="*60)
    print("\nFiles created:")
    print("  - comparison.png (side-by-side comparison)")
    print("  - real_grid.png (real images only)")
    print("  - generated_grid.png (generated images only)")



if __name__ == "__main__":
    main()