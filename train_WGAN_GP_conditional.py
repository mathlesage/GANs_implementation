import torch
import os
from tqdm import trange
import argparse
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from model_cond import Generator, Discriminator, ConditionalLatentSampler
from utils import save_models, compute_fid, compute_fid_conditional, compute_gradient_penalty_cond, D_train_WGAN_GP_cond, G_train_WGAN_GP_cond
import torch.autograd as autograd
import matplotlib.pyplot as plt
import numpy as np




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Conditional WGAN-GP on MNIST.')
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs for training.")
    parser.add_argument("--lr_D", type=float, default=0.0001, help="Learning rate for D.")
    parser.add_argument("--lr_G", type=float, default=0.0001, help="Learning rate for G.")
    parser.add_argument("--lr_sampler", type=float, default=0.0001, help="Learning rate for latent sampler.")
    parser.add_argument("--batch_size", type=int, default=64, help="Size of mini-batches for SGD.")
    parser.add_argument("--gpus", type=int, default=-1, help="Number of GPUs to use (-1 for all available).")
    parser.add_argument("--latent_dim", type=int, default=100, help="Dimension of the latent space.")
    parser.add_argument("--n_critic", type=int, default=5, help="Number of D updates per G update.")
    parser.add_argument("--lambda_gp", type=float, default=10, help="Gradient penalty coefficient.")
    parser.add_argument("--beta1", type=float, default=0.5, help="Adam beta1 parameter.")
    parser.add_argument("--beta2", type=float, default=0.9, help="Adam beta2 parameter.")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of classes in MNIST.")

    args = parser.parse_args()

    to_download = False
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
    os.makedirs('checkpoints_wgan_gp_conditional', exist_ok=True)
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

    print('Model loading...')
    mnist_dim = 784
    
    G = Generator(g_output_dim=mnist_dim).to(device)
    
    # Discriminateur conditionnel
    D = Discriminator(mnist_dim, num_classes=args.num_classes).to(device)
    
    # NOUVEAU: Sampler de bruit latent avec paramètres apprenables a_i et b_i
    latent_sampler = ConditionalLatentSampler(
        latent_dim=args.latent_dim, 
        num_classes=args.num_classes
    ).to(device)

    if args.gpus > 1:
        G = torch.nn.DataParallel(G)
        D = torch.nn.DataParallel(D)
        latent_sampler = torch.nn.DataParallel(latent_sampler)
    print('Model loaded.')

    # Optimizers
    D_optimizer = optim.Adam(D.parameters(), lr=args.lr_D, betas=(args.beta1, args.beta2))
    
    # IMPORTANT: L'optimiseur du générateur inclut maintenant le latent_sampler
    # Cela permet d'optimiser à la fois G et les paramètres a_i, b_i
    G_optimizer = optim.Adam(
        list(G.parameters()) + list(latent_sampler.parameters()), 
        lr=args.lr_G, 
        betas=(args.beta1, args.beta2)
    )

    print('Start training Conditional WGAN-GP:')
    print(f'Configuration: n_critic={args.n_critic}, lambda_gp={args.lambda_gp}')
    print(f'Learnable parameters a_i and b_i for each of {args.num_classes} classes')
    
    n_epoch = args.epochs
    
    for epoch in range(1, n_epoch + 1):
        D_losses = []
        G_losses = []
        W_distances = []
        
        print(f'\nEpoch {epoch}/{n_epoch}')
        
        for batch_idx, (x, labels) in enumerate(train_loader):
            x = x.view(-1, mnist_dim).to(device)
            labels = labels.to(device)  # ← Labels maintenant utilisés
            batch_size = x.size(0)

            # --- Train Discriminator n_critic times ---
            for _ in range(args.n_critic):
                d_loss, w_dist = D_train_WGAN_GP_cond(
                    x, labels, latent_sampler, G, D, D_optimizer, device, args.lambda_gp
                )
                D_losses.append(d_loss)
                W_distances.append(w_dist)

            # --- Train Generator (+ latent_sampler) once ---
            g_loss = G_train_WGAN_GP_cond(labels, latent_sampler, G, D, G_optimizer, device)
            G_losses.append(g_loss)

        # Epoch summary
        avg_D_loss = sum(D_losses) / len(D_losses)
        avg_G_loss = sum(G_losses) / len(G_losses)
        avg_W_dist = sum(W_distances) / len(W_distances)
        
        print(f'\n>>> Epoch {epoch} Summary:')
        print(f'    Average D_loss: {avg_D_loss:.4f}')
        print(f'    Average G_loss: {avg_G_loss:.4f}')
        print(f'    Average Wasserstein Distance: {avg_W_dist:.4f}')

        # Afficher l'évolution des paramètres a_i et b_i
        if epoch % 10 == 0:
            params_summary = latent_sampler.module.get_params_summary() if args.gpus > 1 else latent_sampler.get_params_summary()
            print(f'\n    Learned parameters summary:')
            print(f'    Mean norms per class: {params_summary["mean_norms"]}')
            print(f'    Average std per class: {params_summary["std_means"]}')
        if epoch % 20 == 0:
            print('\n    Computing FID score...')
            fid = compute_fid_conditional(
                G, latent_sampler, device, 
                num_samples=10000,
                batch_size=128
            )
            print(f'    FID after epoch {epoch}: {fid:.4f}')
        # Save checkpoints and generate images
        if epoch % 10 == 0:
            G.eval()
            latent_sampler.eval()
            
            # Générer une image pour chaque chiffre (0-9)
            fig, axes = plt.subplots(2, 5, figsize=(15, 6))
            axes = axes.flatten()
            
            for digit in range(10):
                label = torch.tensor([digit], device=device)
                z = latent_sampler(label, device)
                image = G(z)
                image = image.reshape(1, 28, 28)
                
                axes[digit].imshow(image[0].cpu().detach().numpy(), cmap='gray')
                axes[digit].set_title(f'Digit {digit}')
                axes[digit].axis('off')
            
            plt.tight_layout()
            plt.savefig(f'checkpoints_wgan_gp_conditional/generated_epoch_{epoch}.png')
            plt.close()
            
            G.train()
            latent_sampler.train()
            

            
            # Save models
            checkpoint = {
                'generator': G.state_dict(),
                'discriminator': D.state_dict(),
                'latent_sampler': latent_sampler.state_dict(),
                'epoch': epoch
            }
            torch.save(checkpoint, f'checkpoints_wgan_gp_conditional/checkpoint_epoch_{epoch}.pt')
            print(f'    Models saved at epoch {epoch}')

    print('\nTraining done.')
    
    # Sauvegarder les paramètres finaux a_i et b_i
    final_params = latent_sampler.module.get_params_summary() if args.gpus > 1 else latent_sampler.get_params_summary()
    np.save('checkpoints_wgan_gp_conditional/final_means.npy', final_params['means'])
    np.save('checkpoints_wgan_gp_conditional/final_stds.npy', final_params['stds'])
    print('\nFinal learned parameters saved.')


# Commande pour lancer :
# python train_WGAN_GP_conditional.py --epochs 100 --lr_D 0.0001 --lr_G 0.0001 --batch_size 64 --n_critic 5 --lambda_gp 10