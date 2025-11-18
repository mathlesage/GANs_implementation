import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, g_output_dim):
        super(Generator, self).__init__()       
        self.fc1 = nn.Linear(100, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features*2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features*2)
        self.fc4 = nn.Linear(self.fc3.out_features, g_output_dim)

    # forward method
    def forward(self, x): 
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        return torch.tanh(self.fc4(x))


class ConditionalLatentSampler(nn.Module):
    """
    Module qui génère du bruit latent conditionné sur les labels.
    Les paramètres a_i (mean) et b_i (std) sont apprenables.
    """
    def __init__(self, latent_dim=100, num_classes=10):
        super(ConditionalLatentSampler, self).__init__()
        
        # Paramètres apprenables pour chaque classe
        # a_i : moyennes pour chaque classe
        self.means = nn.Parameter(torch.zeros(num_classes, latent_dim))
        
        # b_i : log(écart-type) pour garantir que std > 0
        self.log_stds = nn.Parameter(torch.zeros(num_classes, latent_dim))
        
        # Initialisation sensée
        nn.init.normal_(self.means, mean=0.0, std=0.1)
        nn.init.constant_(self.log_stds, 0.0)  # std = exp(0) = 1 au début
    
    def forward(self, labels, device):
        """
        Génère du bruit latent z ~ N(a_i, b_i) pour chaque label i
        
        Args:
            labels: tensor de shape (batch_size,) contenant les labels
            device: device pour le tenseur
            
        Returns:
            z: tensor de shape (batch_size, latent_dim) échantillonné selon N(a_i, b_i)
        """
        batch_size = labels.size(0)
        latent_dim = self.means.size(1)
        
        # Récupérer les paramètres conditionnels pour chaque label
        means = self.means[labels]  # (batch_size, latent_dim)
        stds = torch.exp(self.log_stds[labels])  # (batch_size, latent_dim)
        
        # Reparameterization trick: z = mean + std * epsilon
        # où epsilon ~ N(0, 1)
        epsilon = torch.randn(batch_size, latent_dim, device=device)
        z = means + stds * epsilon
        
        return z
    
    def get_params_summary(self):
        """Retourne un résumé des paramètres appris pour chaque classe"""
        with torch.no_grad():
            means = self.means.cpu().numpy()
            stds = torch.exp(self.log_stds).cpu().numpy()
            return {
                'means': means,
                'stds': stds,
                'mean_norms': torch.norm(self.means, dim=1).cpu().numpy(),
                'std_means': stds.mean(axis=1)
            }


class Discriminator(nn.Module):
    """
    Discriminateur conditionnel qui prend en entrée l'image ET le label
    """
    def __init__(self, d_input_dim, num_classes=10):
        super(Discriminator, self).__init__()
        
        # Embedding pour les labels
        self.label_embedding = nn.Embedding(num_classes, 50)
        
        # Le discriminateur prend maintenant image + embedding du label
        self.fc1 = nn.Linear(d_input_dim + 50, 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features//2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features//2)
        self.fc4 = nn.Linear(self.fc3.out_features, 1)

    def forward(self, x, labels):
        """
        Args:
            x: images de shape (batch_size, 784)
            labels: labels de shape (batch_size,)
        """
        # Obtenir l'embedding du label
        label_emb = self.label_embedding(labels)  # (batch_size, 50)
        
        # Concaténer image et label
        x = torch.cat([x, label_emb], dim=1)  # (batch_size, 784 + 50)

        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        return self.fc4(x)



