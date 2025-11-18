import torch
import os
from tqdm import trange
import argparse
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from model_cond import Generator, Discriminator
from utils import D_train, G_train, save_models, D_train_KL, G_train_KL, D_train_PR_KL, G_train_PR_KL, G_D_train_PR, G_train_PR_aux, D_train_PR_aux, G_train_PR_simple_sigmoid, D_train_PR_simple_sigmoid


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train GAN on MNIST.')
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs for training.")
    parser.add_argument("--lr", type=float, default=0.00005, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=64, help="Size of mini-batches for SGD.")
    parser.add_argument("--gpus", type=int, default=-1, help="Number of GPUs to use (-1 for all available).")
    parser.add_argument("--lambda_pr", type=float, default=1.0, help="Lambda for PR-divergence trade-off")
    args = parser.parse_args()

    to_download=False
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_type = "cuda"
        print(f"Using device: CUDA")
        # Use all available GPUs if args.gpus is -1
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
        num_workers=4,  # Use multiple workers for data loading
        pin_memory=True  # Faster data transfer to GPU
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

    # Wrap models in DataParallel if multiple GPUs are available
    if args.gpus > 1:
        G = torch.nn.DataParallel(G)
        D = torch.nn.DataParallel(D)
    print('Model loaded.')

    # Loss and optimizers 
    criterion = nn.BCELoss()
    G_optimizer = optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    D_optimizer = optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))
    



    print('Start training:')
    n_epoch = args.epochs
    
    for epoch in range(1, n_epoch + 1):
        
        D_losses = []
        G_losses = []
        print(f'Epoch {epoch}/{n_epoch}')
        for batch_idx, (x, _) in enumerate(train_loader):

            x = x.view(-1, mnist_dim).to(device)
            #D_loss = D_train(x, G, D, D_optimizer, criterion, device)
            #G_loss = G_train(x, G, D, G_optimizer, criterion, device)
            #D_loss = D_train_KL(x, G, D, D_optimizer, device)
            #G_loss = G_train_KL(x, G, D, G_optimizer, device)
            #D_loss = D_train_PR_KL(x, G, D, D_optimizer, args.lambda_pr, device)
            #G_loss = G_train_PR_KL(x, G, D, G_optimizer, args.lambda_pr, device)
            #G_loss , D_loss = G_D_train_PR(x, G, D, G_optimizer, D_optimizer, args.lambda_pr, device)
        
            
            D_loss = D_train_PR_simple_sigmoid(x, G, D, D_optimizer, args.lambda_pr, device)
            
            G_loss = G_train_PR_simple_sigmoid(x, G, D, G_optimizer, args.lambda_pr, device)

            D_losses.append(D_loss)
            G_losses.append(G_loss)
            
            
        avg_D_loss = sum(D_losses) / len(D_losses)
        avg_G_loss = sum(G_losses) / len(G_losses)
        
        print(f'\n>>> Epoch {epoch} Summary:')
        print(f'    Average D_loss: {avg_D_loss:.4f}')
        print(f'    Average G_loss: {avg_G_loss:.4f}')
        print(f'    Lambda PR: {args.lambda_pr}')
        if epoch % 10 == 0:
            save_models(G, D, 'checkpoints')

    print('Training done.')
