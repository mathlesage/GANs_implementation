import sys
import shutil
import os
import torchvision
from torchvision import datasets, transforms 
from pathlib import Path
import json
import pandas as pd

from model_cond import Generator, ConditionalLatentSampler

# Assurez-vous que les imports nécessaires sont là
try:
    from pytorch_fid.fid_score import calculate_fid_given_paths
    import torch 
except ImportError:
    print("ERREUR: Le module 'pytorch_fid' ou 'torch' n'est pas installé.")
    print("Veuillez exécuter : pip install pytorch-fid torch")
    sys.exit(1)

from precision_recall_kyn_utils import IPR


# Distribution réelle de MNIST train set
MNIST_TRAIN_DISTRIBUTION = {
    0: 5923,
    1: 6742,
    2: 5958,
    3: 6131,
    4: 5842,
    5: 5421,
    6: 5918,
    7: 6265,
    8: 5851,
    9: 5949
}
MNIST_TRAIN_TOTAL = sum(MNIST_TRAIN_DISTRIBUTION.values())


def load_conditional_checkpoint(checkpoint_path, device):
    """
    Charge un checkpoint du conditional WGAN-GP
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    mnist_dim = 784
    latent_dim = 100
    num_classes = 10
    
    # Créer les modèles
    G = Generator(g_output_dim=mnist_dim).to(device)
    latent_sampler = ConditionalLatentSampler(
        latent_dim=latent_dim,
        num_classes=num_classes
    ).to(device)
    
    # Charger les poids
    G.load_state_dict(checkpoint['generator'])
    latent_sampler.load_state_dict(checkpoint['latent_sampler'])
    
    epoch = checkpoint.get('epoch', 'unknown')
    fid = checkpoint.get('fid', 'N/A')
    
    print(f"Checkpoint loaded: epoch {epoch}, FID: {fid}")
    
    return G, latent_sampler, epoch


def generate_n_images_conditional(
    n: int,
    batch_size: int,
    checkpoint_path: str,
    samples_dir: str,
    use_mnist_distribution: bool = True
) -> str:
    """
    Génère 'n' images avec le conditional WGAN-GP
    
    Args:
        n: Nombre total d'images à générer
        batch_size: Taille des batches
        checkpoint_path: Chemin vers le checkpoint
        samples_dir: Dossier de sortie
        use_mnist_distribution: Si True, utilise la distribution MNIST réelle
                               Si False, génère uniformément (même nombre par classe)
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

    # Chargement du modèle
    print('Model Loading...')
    G, latent_sampler, epoch = load_conditional_checkpoint(checkpoint_path, device)
    
    # Prise en charge du DataParallel si plusieurs GPUs
    if torch.cuda.device_count() > 1:
        G = torch.nn.DataParallel(G)
        latent_sampler = torch.nn.DataParallel(latent_sampler)
    
    G.eval()
    latent_sampler.eval()
    print('Model loaded.')

    # Calcul du nombre d'images par classe
    if use_mnist_distribution:
        print("✓ Using MNIST REAL distribution")
        print("  (Distribution: 0:9.87%, 1:11.24%, 2:9.93%, 3:10.22%, 4:9.74%,")
        print("                5:9.04%, 6:9.86%, 7:10.44%, 8:9.75%, 9:9.92%)")
        samples_per_class = {}
        for class_idx in range(10):
            proportion = MNIST_TRAIN_DISTRIBUTION[class_idx] / MNIST_TRAIN_TOTAL
            samples_per_class[class_idx] = int(n * proportion)
        
        # Ajuster pour avoir exactement n
        total = sum(samples_per_class.values())
        if total < n:
            samples_per_class[1] += (n - total)
        
        print("\n  Samples per class:")
        for class_idx in range(10):
            proportion = (samples_per_class[class_idx] / n) * 100
            print(f"    Class {class_idx}: {samples_per_class[class_idx]:4d} ({proportion:.2f}%)")
    else:
        print("✓ Using UNIFORM distribution (equal samples per class)")
        base = n // 10
        remainder = n % 10
        samples_per_class = {i: base + (1 if i < remainder else 0) for i in range(10)}
        
        print("\n  Samples per class:")
        for class_idx in range(10):
            print(f"    Class {class_idx}: {samples_per_class[class_idx]:4d} ({(samples_per_class[class_idx]/n)*100:.2f}%)")

    # Génération d'images
    print(f'Start Generating {n} images into folder: {samples_dir}...')
    output_dir = os.path.join(samples_dir, '0')
    os.makedirs(output_dir, exist_ok=True)

    n_samples_generated = 0
    
    with torch.no_grad():
        for class_idx in range(10):
            n_class = samples_per_class[class_idx]
            if n_class == 0:
                continue
            
            print(f"  Generating {n_class} samples for class {class_idx}...")
            class_samples = 0
            
            while class_samples < n_class:
                current_batch_size = min(batch_size, n_class - class_samples)
                
                # Créer un batch de labels pour cette classe
                labels = torch.full((current_batch_size,), class_idx, 
                                   dtype=torch.long, device=device)
                
                # Générer le bruit conditionné z ~ N(a_i, b_i)
                z = latent_sampler(labels, device)
                
                # Générer les images
                x = G(z)
                x = x.view(current_batch_size, 1, 28, 28)
                x = (x + 1) / 2.0  # [-1, 1] -> [0, 1]
                x = x.clamp(0, 1)
                
                # Sauvegarder
                for k in range(x.size(0)):
                    if n_samples_generated < n:
                        filepath = os.path.join(output_dir, f'{n_samples_generated}.png')
                        torchvision.utils.save_image(x[k], filepath)
                        n_samples_generated += 1
                        class_samples += 1

    print(f'Generation completed. {n_samples_generated} images saved in "{samples_dir}".')
    return samples_dir


def save_n_mnist_images(n: int, path_to_save: str) -> str:
    """
    Télécharge le jeu de données MNIST, sélectionne les 'n' premières images,
    les transforme et les sauvegarde dans un dossier spécifié.
    """
    print(f"\n--- Saving {n} MNIST real images ---")
    
    output_dir = os.path.join(path_to_save, '0')
    os.makedirs(output_dir, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)

    
    print(f"Saving {n} first images in: {path_to_save}")
    for i in range(min(n, len(dataset))):
        img_tensor, label = dataset[i]
        filepath = os.path.join(output_dir, f'real_{i}.png')
        torchvision.utils.save_image(img_tensor, filepath)
    
    print(f"MNIST saving completed. {i + 1} images saved.")
    return path_to_save


def compute_metrics_conditional(
    checkpoint_path: str,
    n_images: int = 10000,
    batch_size: int = 64,
    fid_dims: int = 2048,
    use_mnist_distribution: bool = True
):
    """
    Calcule les métriques pour un checkpoint donné
    """
    device_name = 'cuda:0' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    print(f"\n{'='*60}")
    print(f"Computing metrics for: {checkpoint_path}")
    print(f"{'='*60}")
    
    # Génération des images réelles et fausses
    real_path = save_n_mnist_images(n_images, 'real_mnist_images_temp')
    fake_path = generate_n_images_conditional(
        n=n_images,
        batch_size=2048,
        checkpoint_path=checkpoint_path,
        samples_dir='fake_mnist_images_temp',
        use_mnist_distribution=use_mnist_distribution
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


def evaluate_multiple_epochs(
    checkpoint_dir: str,
    epochs: list,
    output_file: str = "metrics_results.json",
    n_images: int = 10000,
    use_mnist_distribution: bool = True
):
    """
    Évalue plusieurs epochs et sauvegarde les résultats
    """
    results = []
    
    for epoch in epochs:
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
        
        if not os.path.exists(checkpoint_path):
            print(f"WARNING: Checkpoint not found: {checkpoint_path}")
            continue
        
        try:
            precision, recall, fid = compute_metrics_conditional(
                checkpoint_path=checkpoint_path,
                n_images=n_images,
                use_mnist_distribution=use_mnist_distribution
            )
            
            results.append({
                'epoch': epoch,
                'precision': float(precision),
                'recall': float(recall),
                'fid': float(fid),
                'checkpoint': checkpoint_path,
                'distribution': 'mnist' if use_mnist_distribution else 'uniform',
                'n_images': n_images
            })
            
            # Sauvegarde après chaque epoch
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"Results saved to {output_file}")
            
        except Exception as e:
            print(f"ERROR processing epoch {epoch}: {e}")
            continue
    
    # Créer aussi un DataFrame pour faciliter l'analyse
    df = pd.DataFrame(results)
    csv_file = output_file.replace('.json', '.csv')
    df.to_csv(csv_file, index=False)
    print(f"\nResults also saved to {csv_file}")
    
    # Afficher un résumé
    print("\n" + "="*80)
    print("SUMMARY OF ALL RESULTS:")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
    
    return results


if __name__ == "__main__":
    import torch.multiprocessing as mp, platform
    import argparse
    
    if platform.system() == "Windows":
        mp.set_start_method("spawn", force=True)
    
    # Parser d'arguments
    parser = argparse.ArgumentParser(description='Evaluate Conditional WGAN-GP metrics')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_wgan_gp_conditional',
                        help='Directory containing checkpoints')
    parser.add_argument('--epochs', type=int, nargs='+', default=[100, 200, 300, 400, 500, 600],
                        help='List of epochs to evaluate (e.g., --epochs 100 200 300)')
    parser.add_argument('--n_images', type=int, default=10000,
                        help='Number of images to generate for evaluation')
    parser.add_argument('--output', type=str, default='conditional_wgan_gp_metrics.json',
                        help='Output JSON file name')
    parser.add_argument('--distribution', type=str, choices=['mnist', 'uniform'], default='mnist',
                        help='Distribution to use: "mnist" (real MNIST distribution) or "uniform" (equal per class)')
    
    args = parser.parse_args()
    
    # Configuration
    CHECKPOINT_DIR = args.checkpoint_dir
    EPOCHS_TO_EVALUATE = args.epochs
    N_IMAGES = args.n_images
    OUTPUT_FILE = args.output
    USE_MNIST_DISTRIBUTION = (args.distribution == 'mnist')
    
    print("\n" + "="*80)
    print("CONDITIONAL WGAN-GP EVALUATION")
    print("="*80)
    print(f"Checkpoint directory: {CHECKPOINT_DIR}")
    print(f"Epochs to evaluate: {EPOCHS_TO_EVALUATE}")
    print(f"Number of images: {N_IMAGES}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Distribution mode: {args.distribution.upper()}")
    if USE_MNIST_DISTRIBUTION:
        print("  → Using real MNIST distribution (9.04%-11.24% per class)")
    else:
        print("  → Using uniform distribution (10% per class)")
    print("="*80 + "\n")
    
    # Évaluer tous les epochs
    results = evaluate_multiple_epochs(
        checkpoint_dir=CHECKPOINT_DIR,
        epochs=EPOCHS_TO_EVALUATE,
        output_file=OUTPUT_FILE,
        n_images=N_IMAGES,
        use_mnist_distribution=USE_MNIST_DISTRIBUTION
    )
    
    print(f"Results saved to: {OUTPUT_FILE}")