import torch
import os
from tqdm import trange
import argparse
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from model_cond import Generator, Discriminator
from utils import save_models, compute_fid, compute_gradient_penalty, D_train_WGAN_GP, G_train_WGAN_GP
import torch.autograd as autograd
import matplotlib.pyplot as plt




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train WGAN-GP on MNIST.')
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs for training.")
    parser.add_argument("--lr_D", type=float, default=0.0001, help="Learning rate for D.")
    parser.add_argument("--lr_G", type=float, default=0.0001, help="Learning rate for G.")
    parser.add_argument("--batch_size", type=int, default=64, help="Size of mini-batches for SGD.")
    parser.add_argument("--gpus", type=int, default=-1, help="Number of GPUs to use (-1 for all available).")
    parser.add_argument("--latent_dim", type=int, default=100, help="Dimension of the latent space.")
    parser.add_argument("--n_critic", type=int, default=5, help="Number of D updates per G update.")
    parser.add_argument("--lambda_gp", type=float, default=10, help="Gradient penalty coefficient.")
    parser.add_argument("--beta1", type=float, default=0.5, help="Adam beta1 parameter.")
    parser.add_argument("--beta2", type=float, default=0.9, help="Adam beta2 parameter.")

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
    os.makedirs('checkpoints_wgan_gp', exist_ok=True)
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

    # Optimizers (Adam avec betas spécifiques pour WGAN)
    D_optimizer = optim.Adam(D.parameters(), lr=args.lr_D, betas=(args.beta1, args.beta2))
    G_optimizer = optim.Adam(G.parameters(), lr=args.lr_G, betas=(args.beta1, args.beta2))

    print('Start training WGAN-GP:')
    print(f'Configuration: n_critic={args.n_critic}, lambda_gp={args.lambda_gp}')
    
    n_epoch = args.epochs
    
    for epoch in range(1, n_epoch + 1):
        D_losses = []
        G_losses = []
        W_distances = []
        
        print(f'Epoch {epoch}/{n_epoch}')
        
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.view(-1, mnist_dim).to(device)
            batch_size = x.size(0)

            # --- Train Discriminator n_critic times ---
            for _ in range(args.n_critic):
                z = torch.randn(batch_size, args.latent_dim, device=device)
                d_loss, w_dist = D_train_WGAN_GP(x, z, G, D, D_optimizer, device, args.lambda_gp)
                D_losses.append(d_loss)
                W_distances.append(w_dist)

            # --- Train Generator once ---
            z = torch.randn(batch_size, args.latent_dim, device=device)
            g_loss = G_train_WGAN_GP(z, G, D, G_optimizer, device)
            G_losses.append(g_loss)

        avg_D_loss = sum(D_losses) / len(D_losses)
        avg_G_loss = sum(G_losses) / len(G_losses)
        avg_W_dist = sum(W_distances) / len(W_distances)
        
        print(f'\n>>> Epoch {epoch} Summary:')
        print(f'    Average D_loss: {avg_D_loss:.4f}')
        print(f'    Average G_loss: {avg_G_loss:.4f}')
        print(f'    Average Wasserstein Distance: {avg_W_dist:.4f}')

        if epoch % 10 == 0:
            # FID calculation (adapter selon ton compute_fid)
            z1 = torch.randn(1, args.latent_dim, device=device)
            G.eval()
            image_1 = G(z1)
            image_1 = G(z1)
            image_1 = image_1.reshape(1, 1, 28, 28)

            # Afficher l'image
            plt.imshow(image_1[0, 0].cpu().detach().numpy(), cmap='gray')
            plt.title(f'Generated Image - Epoch {epoch}')
            plt.axis('off')
            plt.savefig(f'checkpoints_wgan_gp/image_epoch_{epoch}.png')
            plt.show()
            
            
            fid = compute_fid(G, device, num_samples=1000)
            print(f'    FID after epoch {epoch}: {fid:.4f}')
            save_models(G, D, 'checkpoints_wgan_gp')
            print(f'    Models saved at epoch {epoch}')
            open(f'checkpoints_wgan_gp/fid_epoch_{epoch}.txt', 'w').write(str(f"epoch:, {epoch} fid: {fid}"))

    print('Training done.')


