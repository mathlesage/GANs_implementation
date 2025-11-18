import torch
import os
import torch.nn as nn
from torchvision.utils import save_image
from pytorch_fid import fid_score
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

#creation of a mixture Gaussian class
class GaussMixture(nn.Module):
    def __init__(self, K, d, c, sigma):
        super().__init__()
        self.K = K #number of Gaussians
        self.d = d #dimension of the variable
        self.mu = nn.Parameter(torch.empty(K, d).uniform_(-c,c))
        self.s = nn.Parameter((torch.ones(K, d) * sigma))

        # CORRECTION : Stocker log(sigma) au lieu de sigma pour garantir la positivité de sigma
        initial_log_s = torch.log(torch.tensor(sigma, dtype=torch.float32))
        self.log_s = nn.Parameter(torch.full((K, d), initial_log_s))

    def sample(self, n):
        device = self.mu.device
        k = torch.randint(0, self.K, (n,)).to(device)
        mu_k = self.mu[k]
        sigma_k = torch.exp(self.log_s[k])  # Utiliser exp(log_s) pour obtenir sigma positif
        epsilon = torch.randn_like(mu_k).to(device) #  N(0, I)
        z = mu_k + epsilon * sigma_k #  N(mu, std_dev^2*I)
        return z


def D_train(x, G, D, D_optimizer, criterion, device):
    #=======================Train the discriminator=======================#
    D.zero_grad()

    # train discriminator on real
    x_real = x.to(device)
    y_real = torch.ones(x.shape[0], 1, device=device)

    D_output = D(x_real)
    D_real_loss = criterion(D_output, y_real)
    D_real_score = D_output

    # train discriminator on fake
    z = torch.randn(x.shape[0], 100, device=device)
    x_fake = G(z)
    y_fake = torch.zeros(x.shape[0], 1, device=device)

    D_output = D(x_fake)
    
    D_fake_loss = criterion(D_output, y_fake)
    D_fake_score = D_output

    # gradient backprop & optimize ONLY D's parameters
    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()
        
    return  D_loss.data.item()


def G_train(x, G, D, G_optimizer, criterion, device):
    #=======================Train the generator=======================#
    G.zero_grad()

    z = torch.randn(x.shape[0], 100, device=device)
    y = torch.ones(x.shape[0], 1, device=device)
                 
    G_output = G(z)
    D_output = D(G_output)
    G_loss = criterion(D_output, y)

    # gradient backprop & optimize ONLY G's parameters
    G_loss.backward()
    G_optimizer.step()
        
    return G_loss.data.item()



def save_models(G, D, folder):
    torch.save(G.state_dict(), os.path.join(folder,'G.pth'))
    torch.save(D.state_dict(), os.path.join(folder,'D.pth'))


def load_model(G, folder, device):
    ckpt_path = os.path.join(folder,'G.pth')
    ckpt = torch.load(ckpt_path, map_location=device)
    G.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
    return G







def compute_gradient_penalty(D, real_samples, fake_samples, device):
    """Calcule le gradient penalty pour WGAN-GP"""
    batch_size = real_samples.size(0)
    alpha = torch.rand(batch_size, 1, device=device)
    alpha = alpha.expand_as(real_samples)
    
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = D(interpolates)
    
    fake = torch.ones(d_interpolates.size(), device=device, requires_grad=False)
    
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def D_train_WGAN_GP(x, z, G, D, D_optimizer, device, lambda_gp=10):
    """Entraîne le discriminateur/critique pour une itération"""
    D_optimizer.zero_grad()
    
    # Génère des fausses images
    fake_images = G(z).detach()
    
    # Scores du discriminateur
    real_validity = D(x)
    fake_validity = D(fake_images)
    
    # Wasserstein loss
    wasserstein_d = torch.mean(real_validity) - torch.mean(fake_validity)
    
    # Gradient penalty
    gradient_penalty = compute_gradient_penalty(D, x, fake_images, device)
    
    # Loss totale (on minimise -W + lambda*GP)
    d_loss = -wasserstein_d + lambda_gp * gradient_penalty
    
    d_loss.backward()
    D_optimizer.step()
    
    return d_loss.item(), wasserstein_d.item()


def G_train_WGAN_GP(z, G, D, G_optimizer, device):
    """Entraîne le générateur pour une itération"""
    G_optimizer.zero_grad()
    
    # Génère des images
    fake_images = G(z)
    
    # Le générateur veut maximiser D(G(z))
    fake_validity = D(fake_images)
    g_loss = -torch.mean(fake_validity)
    
    g_loss.backward()
    G_optimizer.step()
    
    return g_loss.item()

def D_train_PR_KL(x, G, D, D_optimizer, lambda_pr, device):
    """
    Train discriminator with PR-divergence (KL auxiliary).
    
    Args:
        lambda_pr: Trade-off parameter (small=recall, large=precision)
    """
    D.zero_grad()
    
    x_real = x.to(device)
    D_real = D(x_real)  
    
    z = torch.randn(x.shape[0], 100, device=device)
    x_fake = G(z)
    D_fake = D(x_fake.detach())
    


    real_term = torch.mean(D_real / lambda_pr)
    
    fake_term = torch.mean(torch.exp(D_fake - 1) * lambda_pr)
    
    D_loss = -real_term + fake_term
    
    if torch.isnan(D_loss) or torch.isinf(D_loss):
        print(f"Warning: D_loss is {D_loss}, skipping this batch")
        return 0.0
    
    D_loss.backward()
    torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=1.0)

    D_optimizer.step()
    
    return D_loss.item()


def G_train_PR_KL(x, G, D, G_optimizer, lambda_pr, device):
    """
    Train generator with PR-divergence (KL auxiliary).
    """
    G.zero_grad()
    
    z = torch.randn(x.shape[0], 100, device=device)
    x_fake = G(z)
    D_fake = D(x_fake)
    
    G_loss = -torch.mean(D_fake / lambda_pr)
    
    if torch.isnan(G_loss) or torch.isinf(G_loss):
        print(f"Warning: G_loss is {G_loss}, skipping this batch")
        return 0.0
    
    G_loss.backward()
    torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
    
    G_optimizer.step()
    
    return G_loss.item()










def G_D_train_GM(x, z,w, G, D, G_optimizer, D_optimizer, criterion, device):
    """
    Entraîne le Générateur et le Discriminateur pour une étape en utilisant 
    l'échantillonnage GM-GAN (z est fourni) et la perte BCELoss.
    """
    
    # Obtenir la taille du batch
    batch_size = x.size(0)
    
    # Créer les étiquettes (labels) que nous utiliserons
    # Étiquette "Vrai" = 1.0
    real_labels = torch.ones(batch_size, 1).to(device)
    # Étiquette "Faux" = 0.0
    fake_labels = torch.zeros(batch_size, 1).to(device)

    # ---------------------
    #  1. Entraîner le Discriminateur (D)
    # ---------------------
    D_optimizer.zero_grad()

    # --- Perte sur les vraies images ---
    # D prédit sur les vraies images 'x'
    real_output = D(x)
    # D doit prédire 1.0 (Vrai) pour ces images
    d_loss_real = criterion(real_output, real_labels)
    
    # Calculer les gradients pour les vraies images
    d_loss_real.backward()

    # --- Perte sur les fausses images ---
    # 'z' vient de l'extérieur (de notre GM-GAN)
    # Générer de fausses images
    fake_x = G(z)
    fake_x2 = G(w)
    
    # D prédit sur les fausses images.
    # .detach() est CRUCIAL: il empêche les gradients de remonter 
    # jusqu'au Générateur (G) pendant que nous entraînons D.
    fake_output = D(fake_x.detach())
    
    # D doit prédire 0.0 (Faux) pour ces images
    d_loss_fake = criterion(fake_output, fake_labels)
    
    # Calculer les gradients pour les fausses images
    d_loss_fake.backward()
    
    # Perte totale du Discriminateur
    d_loss = d_loss_real + d_loss_fake
    
    # Mettre à jour les poids de D
    D_optimizer.step()

    # ---------------------
    #  2. Entraîner le Générateur (G)
    # ---------------------
    G_optimizer.zero_grad()
    
    # Nous devons re-passer les fausses images dans D 
    # (sans .detach() cette fois) pour obtenir les gradients pour G
    fake_output = D(fake_x2)
    
    # Le Générateur G veut tromper D.
    # G gagne si D prédit 1.0 (Vrai) pour ses fausses images.
    g_loss = criterion(fake_output, real_labels)
    
    # Calculer les gradients pour G
    g_loss.backward()
    
    # Mettre à jour les poids de G
    G_optimizer.step()

    # Retourner les pertes pour l'affichage
    return g_loss.item(), d_loss.item()


def D_train_GM(x, z, G, D, D_optimizer, criterion, device):
    #=======================Train the discriminator=======================#
    D.zero_grad()

    # train discriminator on real
    x_real = x.to(device)
    y_real = torch.ones(x.shape[0], 1, device=device)

    D_output = D(x_real)
    D_real_loss = criterion(D_output, y_real)
    D_real_score = D_output

    # train discriminator on fake
    x_fake = G(z)
    y_fake = torch.zeros(x.shape[0], 1, device=device)

    D_output = D(x_fake.detach())
    
    D_fake_loss = criterion(D_output, y_fake)
    D_fake_score = D_output

    # gradient backprop & optimize ONLY D's parameters
    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()
        
    return  D_loss.data.item()


def G_train_GM(x, z, G, D, G_optimizer, criterion, device):
    #=======================Train the generator=======================#
    G.zero_grad()

    y = torch.ones(x.shape[0], 1, device=device)
                 
    G_output = G(z)
    D_output = D(G_output)
    G_loss = criterion(D_output, y)

    # gradient backprop & optimize ONLY G's parameters
    G_loss.backward()
    G_optimizer.step()
        
    return G_loss.data.item()









def compute_fid(G, device, num_samples=10000, batch_size=128):
    """
    Calcule le FID entre les images MNIST réelles et générées
    """
    # Créer les dossiers pour les images
    real_dir = 'fid_real_images'
    fake_dir = 'fid_fake_images'
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)
    
    # Sauvegarder des images réelles (à faire une seule fois)
    if len(os.listdir(real_dir)) == 0:
        print("Sauvegarde des images réelles...")
        data_path = os.getenv('DATA', 'data')
        real_dataset = datasets.MNIST(root=data_path, train=True, 
                                      transform=transforms.ToTensor(), 
                                      download=False)
        for i in range(min(num_samples, len(real_dataset))):
            img, _ = real_dataset[i]
            img = img.repeat(3, 1, 1)
            save_image(img, os.path.join(real_dir, f'real_{i}.png'))
    
    # Générer et sauvegarder des images fake
    print("Génération des images fake...")
    G.eval()
    with torch.no_grad():
        num_batches = num_samples // batch_size
        for i in range(num_batches):
            z = torch.randn(batch_size, 100, device=device)
            fake_images = G(z).view(-1, 1, 28, 28)
            # Normaliser de [-1, 1] à [0, 1]
            fake_images = (fake_images + 1) / 2
            # Convertir en RGB
            fake_images = fake_images.repeat(1, 3, 1, 1)
            
            for j in range(batch_size):
                img_idx = i * batch_size + j
                save_image(fake_images[j], 
                          os.path.join(fake_dir, f'fake_{img_idx}.png'))
    
    G.train()
    
    # Calculer le FID
    print("Calcul du FID...")
    fid_value = fid_score.calculate_fid_given_paths(
        [real_dir, fake_dir],
        batch_size=50,
        device=device,
        dims=2048
    )
    
    # Nettoyer les images fake (garder les real pour les prochaines évals)
    for f in os.listdir(fake_dir):
        os.remove(os.path.join(fake_dir, f))
    
    return fid_value


def compute_fid_conditional(G, latent_sampler, device, num_samples=10000, batch_size=128, num_classes=10):
    """
    Calcule le FID entre les images MNIST réelles et générées pour un conditional GAN
    
    Args:
        G: Générateur
        latent_sampler: ConditionalLatentSampler qui génère z ~ N(a_i, b_i)
        device: device torch
        num_samples: nombre total d'images à générer (réparties équitablement sur les classes)
        batch_size: taille des batchs pour la génération
        num_classes: nombre de classes (10 pour MNIST)
    
    Returns:
        fid_value: score FID global
    """
    # Créer les dossiers pour les images
    real_dir = 'fid_real_images'
    fake_dir = 'fid_fake_images'
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)
    
    # Sauvegarder des images réelles (à faire une seule fois)
    if len(os.listdir(real_dir)) == 0:
        print("Sauvegarde des images réelles...")
        data_path = os.getenv('DATA', 'data')
        real_dataset = datasets.MNIST(root=data_path, train=True, 
                                      transform=transforms.ToTensor(), 
                                      download=False)
        for i in range(min(num_samples, len(real_dataset))):
            img, _ = real_dataset[i]
            img = img.repeat(3, 1, 1)
            save_image(img, os.path.join(real_dir, f'real_{i}.png'))
    
    # Générer et sauvegarder des images fake
    print("Génération des images fake pour toutes les classes...")
    G.eval()
    latent_sampler.eval()
    
    with torch.no_grad():
        # Nombre d'images par classe (répartition équitable)
        samples_per_class = num_samples // num_classes
        img_idx = 0
        
        # Générer pour chaque classe
        for class_idx in range(num_classes):
            num_batches = samples_per_class // batch_size
            remaining = samples_per_class % batch_size
            
            # Générer par batches
            for batch_num in range(num_batches):
                # Créer un batch de labels pour cette classe
                labels = torch.full((batch_size,), class_idx, dtype=torch.long, device=device)
                
                # Générer le bruit conditionné z ~ N(a_i, b_i)
                z = latent_sampler(labels, device)
                
                # Générer les images
                fake_images = G(z).view(-1, 1, 28, 28)
                
                # Normaliser de [-1, 1] à [0, 1]
                fake_images = (fake_images + 1) / 2
                
                # Convertir en RGB
                fake_images = fake_images.repeat(1, 3, 1, 1)
                
                # Sauvegarder chaque image
                for j in range(batch_size):
                    save_image(fake_images[j], 
                              os.path.join(fake_dir, f'fake_{img_idx}.png'))
                    img_idx += 1
            
            # Générer les images restantes pour cette classe
            if remaining > 0:
                labels = torch.full((remaining,), class_idx, dtype=torch.long, device=device)
                z = latent_sampler(labels, device)
                fake_images = G(z).view(-1, 1, 28, 28)
                fake_images = (fake_images + 1) / 2
                fake_images = fake_images.repeat(1, 3, 1, 1)
                
                for j in range(remaining):
                    save_image(fake_images[j], 
                              os.path.join(fake_dir, f'fake_{img_idx}.png'))
                    img_idx += 1
        
        print(f"Généré {img_idx} images fake au total")
    
    G.train()
    latent_sampler.train()
    
    # Calculer le FID
    print("Calcul du FID global...")
    fid_value = fid_score.calculate_fid_given_paths(
        [real_dir, fake_dir],
        batch_size=50,
        device=device,
        dims=2048
    )
    
    # Nettoyer les images fake (garder les real pour les prochaines évals)
    for f in os.listdir(fake_dir):
        os.remove(os.path.join(fake_dir, f))
    
    return fid_value





def display_image(z, G, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    G.to(device)
    z = z.to(device)

    if z.dim() == 1:
        z = z.unsqueeze(0)  

    with torch.no_grad():
        x = G(z)                                
        x = x.view(1, 1, 28, 28)   
        x = (x + 1) / 2.0                           
        x = x.clamp(0, 1)

    img = x.squeeze().cpu().numpy()

    ig, ax = plt.subplots(1, 1, figsize=(3, 3))

    ax.imshow(img, cmap='gray')
    ax.axis('off')  
   
    plt.tight_layout()
    plt.show()
    
    return


def compute_gradient_penalty_cond(D, real_samples, fake_samples, labels, device):
    """Calcule le gradient penalty pour WGAN-GP conditionnel"""
    batch_size = real_samples.size(0)
    alpha = torch.rand(batch_size, 1, device=device)
    alpha = alpha.expand_as(real_samples)
    
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = D(interpolates, labels)  # ← Ajout des labels
    
    fake = torch.ones(d_interpolates.size(), device=device, requires_grad=False)
    
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def D_train_WGAN_GP_cond(x, labels, latent_sampler, G, D, D_optimizer, device, lambda_gp=10):
    """Entraîne le discriminateur/critique pour une itération"""
    D_optimizer.zero_grad()
    
    z = latent_sampler(labels, device)
    fake_images = G(z).detach()
    real_validity = D(x, labels)
    fake_validity = D(fake_images, labels)
    
    # Wasserstein loss
    wasserstein_d = torch.mean(real_validity) - torch.mean(fake_validity)
    
    # Gradient penalty
    gradient_penalty = compute_gradient_penalty(D, x, fake_images, labels, device)
    
    # Loss totale (on minimise -W + lambda*GP)
    d_loss = -wasserstein_d + lambda_gp * gradient_penalty
    
    d_loss.backward()
    D_optimizer.step()
    
    return d_loss.item(), wasserstein_d.item()


def G_train_WGAN_GP_cond(labels, latent_sampler, G, D, G_optimizer, device):
    """Entraîne le générateur pour une itération"""
    G_optimizer.zero_grad()    
    z = latent_sampler(labels, device)
    fake_images = G(z)
    
    # Le générateur veut maximiser D(G(z))
    fake_validity = D(fake_images, labels)
    g_loss = -torch.mean(fake_validity)
    
    g_loss.backward()
    G_optimizer.step()
    
    return g_loss.item()
