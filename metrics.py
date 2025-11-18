import sys
import shutil
import os
import torchvision
from torchvision import datasets, transforms 
from pathlib import Path
import argparse
import json
import pandas as pd

from model_cond import Generator

# Assurez-vous que les imports nécessaires sont là
try:
    from pytorch_fid.fid_score import calculate_fid_given_paths
    import torch 
except ImportError:
    print("ERREUR: Le module 'pytorch_fid' ou 'torch' n'est pas installé.")
    print("Veuillez exécuter : pip install pytorch-fid torch")
    sys.exit(1)

from precision_recall_kyn_utils import IPR


def load_generator(model_path: str, device):
    """
    Charge un générateur depuis un fichier .pth
    """
    print(f"Loading generator from {model_path}...")
    
    mnist_dim = 784
    model = Generator(g_output_dim=mnist_dim).to(device)
    
    # Charger les poids
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    
    model.eval()
    print('Generator loaded.')
    
    return model


def generate_n_images_standard(
    n: int, 
    batch_size: int,
    model_path: str,
    samples_dir: str,
    model_type: str = 'gan'
) -> str:
    """
    Génère 'n' images avec un GAN standard (Vanilla GAN, WGAN-GP, etc.)
    
    Args:
        n: Nombre d'images à générer
        batch_size: Taille des batches
        model_path: Chemin vers le fichier G.pth
        samples_dir: Dossier de sortie
        model_type: Type de modèle ('gan', 'wgan', 'gm_gan')
    """
    # Détection de l'appareil
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using device: MPS (Apple Metal)")
    else:
        device = torch.device("cpu")
        print(f"Using device: CPU")

    # Chargement du Modèle
    print('Model Loading...')
    model = load_generator(model_path, device)
    
    # Prise en charge du DataParallel si plusieurs GPUs
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # Génération d'Images
    print(f'Start Generating {n} images into folder: {samples_dir}...')
    output_dir = os.path.join(samples_dir, '0')
    os.makedirs(output_dir, exist_ok=True)

    n_samples_generated = 0
    with torch.no_grad():
        while n_samples_generated < n:
            current_batch_size = min(batch_size, n - n_samples_generated)
            
            # Bruit aléatoire (latente)
            z = torch.randn(current_batch_size, 100).to(device)
            
            x = model(z)                                # (B,784), valeurs ~[-1,1]
            x = x.view(current_batch_size, 1, 28, 28)   # (B,1,28,28)
            x = (x + 1) / 2.0                           # -> [0,1]
            x = x.clamp(0, 1)

            for k in range(x.size(0)):
                if n_samples_generated < n:
                    filepath = os.path.join(output_dir, f'{n_samples_generated}.png')
                    torchvision.utils.save_image(x[k], filepath)
                    n_samples_generated += 1

    print(f'Generation completed. {n_samples_generated} images saved in "{samples_dir}".')
    return samples_dir


def generate_n_images_gm_gan(
    n: int, 
    batch_size: int,
    model_path: str,
    gmm_params_path: str,
    samples_dir: str,
    K: int = 10,
    d: int = 100
) -> str:
    """
    Génère 'n' images avec un GM-GAN
    
    Args:
        n: Nombre d'images à générer
        batch_size: Taille des batches
        model_path: Chemin vers G.pth
        gmm_params_path: Chemin vers gmm_params.pth
        samples_dir: Dossier de sortie
        K: Nombre de Gaussians
        d: Dimension latente
    """
    # Détection de l'appareil
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using device: MPS (Apple Metal)")
    else:
        device = torch.device("cpu")
        print(f"Using device: CPU")

    # Chargement du Modèle
    print('Model Loading...')
    model = load_generator(model_path, device)
    
    # Charger les paramètres GMM
    print(f'Loading GMM parameters from {gmm_params_path}...')
    gmm_params = torch.load(gmm_params_path, map_location=device)
    mus = gmm_params['mus']
    std_devs = gmm_params['std_devs']
    print(f"GMM parameters loaded (K={K}, d={d}).")
    
    # Prise en charge du DataParallel si plusieurs GPUs
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # Génération d'Images
    print(f'Start Generating {n} images into folder: {samples_dir}...')
    output_dir = os.path.join(samples_dir, '0')
    os.makedirs(output_dir, exist_ok=True)

    n_samples_generated = 0
    with torch.no_grad():
        while n_samples_generated < n:
            current_batch_size = min(batch_size, n - n_samples_generated)
            
            # Échantillonner depuis le GMM
            k_indices = torch.randint(low=0, high=K, size=(current_batch_size,)).to(device)
            mu_batch = mus[k_indices]
            std_batch = std_devs[k_indices]
            epsilon = torch.randn(current_batch_size, d).to(device)
            z = mu_batch + epsilon * std_batch
            
            x = model(z)                                # (B,784), valeurs ~[-1,1]
            x = x.view(current_batch_size, 1, 28, 28)   # (B,1,28,28)
            x = (x + 1) / 2.0                           # -> [0,1]
            x = x.clamp(0, 1)

            for k in range(x.size(0)):
                if n_samples_generated < n:
                    filepath = os.path.join(output_dir, f'{n_samples_generated}.png')
                    torchvision.utils.save_image(x[k], filepath)
                    n_samples_generated += 1

    print(f'Generation completed. {n_samples_generated} images saved in "{samples_dir}".')
    return samples_dir


def save_n_mnist_images(n: int, path_to_save: str, train: bool = False) -> str:
    """
    Télécharge le jeu de données MNIST, sélectionne les 'n' premières images,
    les transforme et les sauvegarde dans un dossier spécifié.
    
    Args:
        n: Nombre d'images à sauvegarder
        path_to_save: Chemin de sauvegarde
        train: Si True, utilise train set, sinon test set
    """
    dataset_name = "train" if train else "test"
    print(f"\n--- Saving {n} MNIST {dataset_name} images ---")
    
    output_dir = os.path.join(path_to_save, '0')
    os.makedirs(output_dir, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    try:
        dataset = datasets.MNIST('./data', train=train, download=True, transform=transform)
    except Exception as e:
        print(f"ERROR downloading MNIST: {e}")
        sys.exit(1)
        
    print(f"Saving {n} first images in: {path_to_save}")
    for i in range(min(n, len(dataset))):
        img_tensor, label = dataset[i]
        filepath = os.path.join(output_dir, f'real_{i}.png')
        torchvision.utils.save_image(img_tensor, filepath)
        
    print(f"MNIST saving completed. {i + 1} images saved.")
    return path_to_save


def compute_metrics_standard(
    model_path: str,
    n_images: int = 10000,
    batch_size: int = 64,
    fid_dims: int = 2048,
    model_type: str = 'gan',
    gmm_params_path: str = None,
    use_train_set: bool = False
):
    """
    Calcule les métriques pour un modèle standard (GAN, WGAN, GM-GAN)
    
    Args:
        model_path: Chemin vers G.pth
        n_images: Nombre d'images à générer
        batch_size: Taille des batches pour FID
        fid_dims: Dimensions pour FID
        model_type: Type de modèle ('gan', 'wgan', 'gm_gan')
        gmm_params_path: Chemin vers gmm_params.pth (pour GM-GAN uniquement)
        use_train_set: Si True, compare avec train set, sinon test set
    """
    device_name = 'cuda:0' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    print(f"\n{'='*60}")
    print(f"Computing metrics for: {model_path}")
    print(f"Model type: {model_type.upper()}")
    print(f"{'='*60}")
    
    # Génération des images réelles
    real_path = save_n_mnist_images(n_images, 'real_mnist_images_temp', train=use_train_set)
    
    # Génération des images fake
    if model_type == 'gm_gan':
        if gmm_params_path is None:
            # Chercher automatiquement dans le même dossier que G.pth
            model_dir = os.path.dirname(model_path)
            gmm_params_path = os.path.join(model_dir, 'gmm_params.pth')
            if not os.path.exists(gmm_params_path):
                print(f"ERROR: gmm_params.pth not found at {gmm_params_path}")
                print("Please specify --gmm_params_path")
                sys.exit(1)
        
        fake_path = generate_n_images_gm_gan(
            n=n_images,
            batch_size=2048,
            model_path=model_path,
            gmm_params_path=gmm_params_path,
            samples_dir='fake_mnist_images_temp'
        )
    else:
        fake_path = generate_n_images_standard(
            n=n_images,
            batch_size=2048,
            model_path=model_path,
            samples_dir='fake_mnist_images_temp',
            model_type=model_type
        )
    
    # Chargement des datasets
    real_dataset = datasets.ImageFolder(
        root=real_path,
        transform=transforms.Compose([transforms.ToTensor()])
    )
    
    fake_dataset = datasets.ImageFolder(
        root=fake_path,
        transform=transforms.Compose([transforms.ToTensor()])
    )
    
    real_loader = torch.utils.data.DataLoader(real_dataset, batch_size=batch_size, shuffle=False)
    fake_loader = torch.utils.data.DataLoader(fake_dataset, batch_size=batch_size, shuffle=False)
    
    # Calcul des métriques PRDC
    print("\n--- Computing PRDC metrics ---")
    
    real_images = torch.cat([batch[0] for batch in real_loader], dim=0)
    fake_images = torch.cat([batch[0] for batch in fake_loader], dim=0)
    
    # Convertir en RGB si nécessaire
    if real_images.size(1) == 1:
        real_images = real_images.repeat(1, 3, 1, 1)
    if fake_images.size(1) == 1:
        fake_images = fake_images.repeat(1, 3, 1, 1)
    
    # Calcul Precision/Recall
    ipr_calculator = IPR(batch_size=batch_size, k=3, num_samples=n_images)
    
    print("Computing real images features...")
    ipr_calculator.compute_manifold_ref(real_images)
    
    print("Computing generated images features...")
    pr_results = ipr_calculator.precision_and_recall(fake_images)
    
    precision = pr_results.precision
    recall = pr_results.recall
    
    # Calcul FID
    print("\n--- Computing FID ---")
    fid_value = calculate_fid_given_paths(
        [os.path.join(real_path, '0'), os.path.join(fake_path, '0')],
        batch_size=batch_size,
        device=device_name,
        dims=fid_dims,
        num_workers=os.cpu_count()
    )
    
    print(f"\n{'='*60}")
    print(f"RESULTS: Precision={precision:.4f}, Recall={recall:.4f}, FID={fid_value:.4f}")
    print(f"{'='*60}\n")
    
    # Nettoyage
    if fake_path and os.path.isdir(fake_path):
        print(f"Cleaning temporary folder: {fake_path}")
        shutil.rmtree(fake_path)
    
    if real_path and os.path.isdir(real_path):
        print(f"Cleaning temporary folder: {real_path}")
        shutil.rmtree(real_path)
    
    return precision, recall, fid_value


if __name__ == "__main__":
    import torch.multiprocessing as mp, platform
    
    if platform.system() == "Windows":
        mp.set_start_method("spawn", force=True)
    
    # Parser d'arguments
    parser = argparse.ArgumentParser(description='Compute metrics for standard GAN models')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to G.pth file')
    parser.add_argument('--model_type', type=str, choices=['gan', 'wgan', 'gm_gan'], default='gan',
                        help='Type of model: gan (vanilla), wgan (WGAN-GP), or gm_gan (GM-GAN)')
    parser.add_argument('--gmm_params_path', type=str, default=None,
                        help='Path to gmm_params.pth (required for gm_gan, optional otherwise)')
    parser.add_argument('--n_images', type=int, default=10000,
                        help='Number of images to generate for evaluation')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for FID computation')
    parser.add_argument('--fid_dims', type=int, default=2048,
                        help='Dimensionality of Inception features')
    parser.add_argument('--use_train_set', action='store_true',
                        help='Use MNIST train set instead of test set for comparison')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file to save results (optional)')
    
    args = parser.parse_args()
    
    # Vérifications
    if not os.path.exists(args.model_path):
        print(f"ERROR: Model file not found: {args.model_path}")
        sys.exit(1)
    
    if args.model_type == 'gm_gan' and args.gmm_params_path is None:
        model_dir = os.path.dirname(args.model_path)
        args.gmm_params_path = os.path.join(model_dir, 'gmm_params.pth')
        print(f"GM-GAN detected, looking for gmm_params.pth at: {args.gmm_params_path}")
    
    print("\n" + "="*80)
    print("STANDARD GAN EVALUATION")
    print("="*80)
    print(f"Model path: {args.model_path}")
    print(f"Model type: {args.model_type.upper()}")
    print(f"Number of images: {args.n_images}")
    print(f"Using MNIST {'train' if args.use_train_set else 'test'} set")
    if args.model_type == 'gm_gan':
        print(f"GMM params: {args.gmm_params_path}")
    print("="*80 + "\n")
    
    # Calcul des métriques
    precision, recall, fid_value = compute_metrics_standard(
        model_path=args.model_path,
        n_images=args.n_images,
        batch_size=args.batch_size,
        fid_dims=args.fid_dims,
        model_type=args.model_type,
        gmm_params_path=args.gmm_params_path,
        use_train_set=args.use_train_set
    )
    
    # Sauvegarder les résultats si demandé
    if args.output:
        results = {
            'model_path': args.model_path,
            'model_type': args.model_type,
            'precision': float(precision),
            'recall': float(recall),
            'fid': float(fid_value),
            'n_images': args.n_images,
            'dataset': 'train' if args.use_train_set else 'test'
        }
        
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {args.output}")
    
    print(f"\nDone: Precision={precision:.4f}, Recall={recall:.4f}, FID={fid_value:.4f}")


