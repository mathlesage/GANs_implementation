import torch 
import torchvision
import os
import argparse
import numpy as np

from model_cond import Generator, ConditionalLatentSampler


def load_checkpoint(checkpoint_path, G, latent_sampler, device):
    """
    Charge un checkpoint sauvegardé
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    G.load_state_dict(checkpoint['generator'])
    latent_sampler.load_state_dict(checkpoint['latent_sampler'])
    
    epoch = checkpoint.get('epoch', 'unknown')
    fid = checkpoint.get('fid', 'N/A')
    
    print(f"Checkpoint loaded: epoch {epoch}, FID: {fid}")
    return G, latent_sampler


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate images from Conditional WGAN-GP.')
    parser.add_argument("--checkpoint", type=str, 
                        default="checkpoints_wgan_gp_conditional/checkpoint_epoch_600.pt",
                        help="Path to the checkpoint file.")
    parser.add_argument("--batch_size", type=int, default=2048,
                        help="The batch size to use for generation.")
    parser.add_argument("--n_samples", type=int, default=10000,
                        help="Total number of samples to generate.")
    parser.add_argument("--output_dir", type=str, default='samples',
                        help="Directory to save generated samples.")
    parser.add_argument("--latent_dim", type=int, default=100,
                        help="Dimension of the latent space.")
    parser.add_argument("--num_classes", type=int, default=10,
                        help="Number of classes (10 for MNIST).")
    parser.add_argument("--balanced", action='store_true',
                        help="Generate balanced samples across all classes.")
    
    args = parser.parse_args()

    # Device setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: CUDA")
    else:
        device = torch.device("cpu")
        print(f"Using device: CPU")

    
    # Model Pipeline
    mnist_dim = 784
    
    # Créer les modèles
    G = Generator(g_output_dim=mnist_dim).to(device)
    latent_sampler = ConditionalLatentSampler(
        latent_dim=args.latent_dim, 
        num_classes=args.num_classes
    ).to(device)
    
    # Charger le checkpoint
    G, latent_sampler = load_checkpoint(args.checkpoint, G, latent_sampler, device)
    
    # Multi-GPU si disponible
    if torch.cuda.device_count() > 1:
        G = torch.nn.DataParallel(G)
        latent_sampler = torch.nn.DataParallel(latent_sampler)
    
    G.eval()
    latent_sampler.eval()
    
    print('Model loaded.')
    
    # Afficher les paramètres appris
    print('\nLearned latent parameters:')
    if hasattr(latent_sampler, 'module'):
        params_summary = latent_sampler.module.get_params_summary()
    else:
        params_summary = latent_sampler.get_params_summary()
    
    print('Mean norms per class:', params_summary['mean_norms'])
    print('Average std per class:', params_summary['std_means'])

    print(f'\nStart Generating {args.n_samples} samples...')
    os.makedirs(args.output_dir, exist_ok=True)

    n_samples = 0
    
    with torch.no_grad():
        if not(args.balanced):
            # Génération équilibrée : même nombre d'images par classe
            print("Generating balanced samples across all classes...")
            samples_per_class = args.n_samples // args.num_classes
            
            for class_idx in range(args.num_classes):
                print(f"  Generating class {class_idx}...")
                class_samples = 0
                
                while class_samples < samples_per_class:
                    # Nombre d'images à générer dans ce batch
                    current_batch_size = min(args.batch_size, samples_per_class - class_samples)
                    
                    # Créer un batch de labels pour cette classe
                    labels = torch.full((current_batch_size,), class_idx, 
                                       dtype=torch.long, device=device)
                    
                    # Générer le bruit conditionné z ~ N(a_i, b_i)
                    z = latent_sampler(labels, device)
                    
                    # Générer les images
                    x = G(z)
                    x = x.reshape(current_batch_size, 28, 28)
                    
                    # Sauvegarder chaque image
                    for k in range(x.shape[0]):
                        if n_samples < args.n_samples:
                            torchvision.utils.save_image(
                                x[k:k+1], 
                                os.path.join(args.output_dir, f'{n_samples}.png')
                            )
                            n_samples += 1
                            class_samples += 1
        
        else:
            # Génération aléatoire : labels tirés uniformément
            print("Generating random samples...")
            
            while n_samples < args.n_samples:
                current_batch_size = min(args.batch_size, args.n_samples - n_samples)
                
                # Tirer des labels aléatoirement
                labels = torch.randint(0, args.num_classes, (current_batch_size,), 
                                      dtype=torch.long, device=device)
                
                # Générer le bruit conditionné z ~ N(a_i, b_i)
                z = latent_sampler(labels, device)
                
                # Générer les images
                x = G(z)
                x = x.reshape(current_batch_size, 28, 28)
                
                # Sauvegarder chaque image
                for k in range(x.shape[0]):
                    if n_samples < args.n_samples:
                        label = labels[k].item()
                        torchvision.utils.save_image(
                            x[k:k+1], 
                            os.path.join(args.output_dir, f'{n_samples}_class{label}.png')
                        )
                        n_samples += 1

    print(f'\n{n_samples} samples generated and saved to {args.output_dir}/')
    
    # Statistiques de génération
    if args.balanced:
        print(f'Generated {samples_per_class} samples per class')
    else:
        print('Samples generated with random class distribution')

