import torch
import os
from tqdm import trange
import argparse
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from model import Generator, Discriminator
from utils import save_models 


if _name_ == '_main_':
    parser = argparse.ArgumentParser(description='Train GAN on MNIST.')
    parser.add_argument("--epochs", type=int, default=90, help="Number of epochs for training.")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=64, help="Size of mini-batches for SGD.")
    parser.add_argument("--gpus", type=int, default=-1, help="Number of GPUs to use (-1 for all available).")
    

    # Nouveaux arguments pour le Mélange de Gaussiennes (GM-GAN)
    parser.add_argument("--latent_dim", type=int, default=100, help="Dimension of the latent space (d).")
    parser.add_argument("--k_components", type=int, default=10, help="Number of Gaussian components (K).")
    parser.add_argument("--gm_c", type=float, default=0.1, help="Range [-c, c] for sampling Gaussian means.")
    parser.add_argument("--gm_sigma", type=float, default=0.15, help="Std dev (sigma) for Gaussian components.")

    args = parser.parse_args()

    to_download=False
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_type = "cuda"
        print(f"Using device: CUDA")
        if args.gpus == -1:
            args.gpus = torch.cuda.device_count()
            print(f"Using {args.gpus} GPUs.")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        device_type = "mps"
        print(f"Using device: MPS (Apple Metal)")
    else:
        device = torch.device("cpu")
        device_type = "cpu"
        print(f"Using device: CPU")
        
    # Create directories
    os.makedirs('checkpoints', exist_ok=True)
    data_path = os.getenv('DATA')
    if data_path is None:
        data_path = "data"
        to_download = True
        
    # Data Pipeline
    print('Dataset loading...')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])
    train_dataset = datasets.MNIST(root=data_path, train=True, transform=transform, download=to_download)
    test_dataset = datasets.MNIST(root=data_path, train=False, transform=transform, download=to_download)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    print('Dataset loaded.')

    # Model setup
    print('Model loading...')
    mnist_dim = 784
    G = Generator(g_output_dim=mnist_dim).to(device)
    D = Discriminator(mnist_dim).to(device)

    if args.gpus > 1:
        G = torch.nn.DataParallel(G)
        D = torch.nn.DataParallel(D)
    print('Model loaded.')

    G_optimizer = optim.Adam(G.parameters(), lr=args.lr)
    D_optimizer = optim.Adam(D.parameters(), lr=args.lr)
    
    print(f"Initialisation du Static GM-GAN avec K={args.k_components}...")
    K = args.k_components  
    d = args.latent_dim
    c = args.gm_c           
    sigma = args.gm_sigma     

    mus = (torch.rand(K, d) * 2 * c - c).to(device)
    

    std_devs = (torch.ones(K, d) * sigma).to(device)
    
 
    print('Start training:')
    n_epoch = args.epochs
    
    for epoch in range(1, n_epoch + 1):
        
        # Mettre les modèles en mode entraînement
        G.train()
        D.train()
        
        D_losses = []
        G_losses = []
        print(f'Epoch {epoch}/{n_epoch}')
        
        for batch_idx, (x, _) in enumerate(train_loader):

            x = x.view(-1, mnist_dim).to(device)
            batch_size = x.size(0)

            D_optimizer.zero_grad()
            
            D_output_real = D(x)
            D_loss_real = -torch.log(D_output_real + 1e-8)  # Ajout d'une petite constante pour la stabilité numérique
            
            
            k_indices_D = torch.randint(0, K, (batch_size,)).to(device)
            mu_batch_D = mus[k_indices_D]
            std_batch_D = std_devs[k_indices_D]

            epsilon_D = torch.randn_like(mu_batch_D).to(device) 
            z_D = mu_batch_D + epsilon_D * std_batch_D # Reparamétrisation
            
            x_fake_D = G(z_D).detach()

            # Calcul de la perte
            D_output_fake = D(x_fake_D)
            D_loss_fake = -torch.log(1 - D_output_fake + 1e-8)  # Stabilité numérique


            D_loss = torch.mean(0.5 * (D_loss_real + D_loss_fake))
            
            D_loss.backward()
            D_optimizer.step()

            G_optimizer.zero_grad()
            
            k_indices_G = torch.randint(0, K, (batch_size,)).to(device)
            mu_batch_G = mus[k_indices_G]
            std_batch_G = std_devs[k_indices_G]

            epsilon_G = torch.randn_like(mu_batch_G).to(device)
            z_G = mu_batch_G + epsilon_G * std_batch_G # Reparamétrisation
            
            x_fake_G = G(z_G)

            D_output_fake_G = D(x_fake_G)
            
            G_loss = -torch.log(D_output_fake_G + 1e-8)

            G_loss.backward()
            G_optimizer.step()
            
            

            D_losses.append(D_loss.item())
            G_losses.append(G_loss.item())
            
            
        avg_D_loss = sum(D_losses) / len(D_losses)
        avg_G_loss = sum(G_losses) / len(G_losses)
        
        
        print(f'\n>>> Epoch {epoch} Summary:')
        print(f'    Average D_loss: {avg_D_loss:.4f}')
        print(f'    Average G_loss: {avg_G_loss:.4f}')
        
        if epoch % 10 == 0:
            #Sauvegarde les poids de G et D 
            save_models(G, D, 'checkpoints_static_gm_gan')

            #Sauvegarder les paramètres du GMM
            gmm_params = {
                'mus': mus,
                'std_devs': std_devs,
                'K': K,
                'd': d
            }
            save_path = os.path.join('checkpoints_static_gm_gan', 'gmm_params.pth')
            torch.save(gmm_params, save_path)
            print(f'GMM parameters saved to {save_path}')

    print('Training done.')