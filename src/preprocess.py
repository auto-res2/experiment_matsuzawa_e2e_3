import torch
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader

class DiffusionSampler:
    """Main diffusion sampler that can operate in baseline or ADPO+ adaptive mode"""
    
    def __init__(self, diffusion_model, state_extractor=None, policy=None, max_steps=50, adaptive=False):
        self.model = diffusion_model
        self.max_steps = max_steps
        self.adaptive = adaptive
        self.state_extractor = state_extractor
        self.policy = policy
        
        if self.adaptive and (self.state_extractor is None or self.policy is None):
            raise ValueError("State extractor and policy are required for adaptive mode")

    def sample(self, init_latent):
        """Sample from the diffusion model using either baseline or adaptive approach"""
        current_latent = init_latent.clone()
        iterations = 0
        latent_quality_list = []
        rl_actions = []
        
        while iterations < self.max_steps:
            noise_scale = torch.tensor([current_latent.std().item()])
            pred_reliability = torch.tensor([1.0 - current_latent.var().item()])
            latent_quality = 1.0 / (current_latent.var().item() + 1e-5)
            latent_quality_list.append(latent_quality)
            
            if self.adaptive:
                state_rep = self.state_extractor(current_latent.unsqueeze(0), noise_scale, pred_reliability)
                action = self.policy.predict(state_rep.detach().cpu().numpy())
                rl_actions.append(action)
                noise_mod, step_mod, skip_flag = action
                
                if skip_flag > 0.5:
                    iterations += 1
                    continue
                
                current_noise = noise_mod * noise_scale * torch.ones_like(current_latent)
                step_factor = step_mod
            else:
                current_noise = 1.0 * noise_scale * torch.ones_like(current_latent)
                step_factor = 1.0
            
            step = self.model(current_latent, current_noise) * step_factor
            current_latent = current_latent + step
            iterations += 1
            
            if len(latent_quality_list) > 5:
                recent_qualities = latent_quality_list[-5:]
                if np.std(recent_qualities) < 0.01:  # Quality has converged
                    break
        
        return current_latent, iterations, latent_quality_list, rl_actions

class NoisePerturbationSampler(DiffusionSampler):
    """Extended sampler that introduces noise perturbations for robustness testing"""
    
    def sample(self, init_latent, perturb_factor=1.5):
        """Sample with dynamic noise perturbations"""
        current_latent = init_latent.clone()
        iterations = 0
        latent_quality_list = []
        rl_actions = []
        perturbation_log = []
        
        while iterations < self.max_steps:
            base_noise_scale = torch.tensor([current_latent.std().item()])
            
            is_perturbed = torch.rand(1).item() > 0.7
            if is_perturbed:
                noise_scale = base_noise_scale * perturb_factor
                perturbation_log.append((iterations, perturb_factor))
            else:
                noise_scale = base_noise_scale
                perturbation_log.append((iterations, 1.0))
            
            pred_reliability = torch.tensor([1.0 - current_latent.var().item()])
            latent_quality = 1.0 / (current_latent.var().item() + 1e-5)
            latent_quality_list.append(latent_quality)
            
            state_rep = self.state_extractor(current_latent.unsqueeze(0), noise_scale, pred_reliability)
            action = self.policy.predict(state_rep.detach().cpu().numpy())
            rl_actions.append(action)
            noise_mod, step_mod, skip_flag = action
            
            if skip_flag > 0.5:
                iterations += 1
                continue
            
            current_noise = noise_mod * noise_scale * torch.ones_like(current_latent)
            step = self.model(current_latent, current_noise) * step_mod
            current_latent = current_latent + step
            iterations += 1
        
        return current_latent, iterations, latent_quality_list, rl_actions

def preprocess_data(data_path=None, data_type="synthetic"):
    """Preprocess data for diffusion model training"""
    if data_type == "synthetic":
        print("Generating synthetic data for testing...")
        data = torch.randn(1000, 16)  # 1000 samples of 16-dimensional data
        labels = torch.randint(0, 10, (1000,))  # Random labels
        return data, labels
    
    elif data_type == "cifar10":
        print("CIFAR-10 preprocessing not implemented in this demo")
        return preprocess_data(data_type="synthetic")
    
    elif data_type == "celeba":
        print("CelebA preprocessing not implemented in this demo")
        return preprocess_data(data_type="synthetic")
    
    else:
        raise ValueError(f"Unknown data type: {data_type}")

def create_data_loader(data, labels, batch_size=32, shuffle=True):
    """Create PyTorch DataLoader from preprocessed data"""
    
    class SimpleDataset(Dataset):
        def __init__(self, data, labels):
            self.data = data
            self.labels = labels
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx], self.labels[idx]
    
    dataset = SimpleDataset(data, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

def normalize_data(data, method="standard"):
    """Normalize data using specified method"""
    if method == "standard":
        mean = data.mean(dim=0, keepdim=True)
        std = data.std(dim=0, keepdim=True)
        normalized = (data - mean) / (std + 1e-8)
    
    elif method == "minmax":
        min_val = data.min(dim=0, keepdim=True)[0]
        max_val = data.max(dim=0, keepdim=True)[0]
        normalized = (data - min_val) / (max_val - min_val + 1e-8)
    
    elif method == "none":
        normalized = data
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized

def add_noise_schedule(data, noise_levels=None):
    """Add noise to data according to a schedule for diffusion training"""
    if noise_levels is None:
        noise_levels = torch.linspace(0.1, 1.0, 10)
    
    noisy_data = []
    for noise_level in noise_levels:
        noise = torch.randn_like(data) * noise_level
        noisy_sample = data + noise
        noisy_data.append(noisy_sample)
    
    return torch.stack(noisy_data)

def prepare_experiment_data(latent_dim=16, num_samples=100):
    """Prepare data specifically for ADPO+ experiments"""
    print(f"Preparing experiment data: {num_samples} samples of {latent_dim}D")
    
    test_samples = []
    
    normal_samples = torch.randn(num_samples // 3, latent_dim)
    test_samples.append(normal_samples)
    
    uniform_samples = torch.rand(num_samples // 3, latent_dim) * 2 - 1  # Range [-1, 1]
    test_samples.append(uniform_samples)
    
    mixed_samples = torch.randn(num_samples - 2 * (num_samples // 3), latent_dim)
    mixed_samples += torch.rand_like(mixed_samples) * 0.5  # Add uniform noise
    test_samples.append(mixed_samples)
    
    all_samples = torch.cat(test_samples, dim=0)
    
    indices = torch.randperm(len(all_samples))
    all_samples = all_samples[indices]
    
    print(f"Generated {len(all_samples)} test samples")
    print(f"Sample statistics - Mean: {all_samples.mean():.4f}, Std: {all_samples.std():.4f}")
    
    return all_samples

def validate_data_quality(data):
    """Validate the quality of preprocessed data"""
    validation_results = {}
    
    validation_results['has_nan'] = torch.isnan(data).any().item()
    validation_results['has_inf'] = torch.isinf(data).any().item()
    
    validation_results['mean'] = data.mean().item()
    validation_results['std'] = data.std().item()
    validation_results['min'] = data.min().item()
    validation_results['max'] = data.max().item()
    
    validation_results['shape'] = list(data.shape)
    validation_results['total_elements'] = data.numel()
    
    quality_score = 1.0
    if validation_results['has_nan'] or validation_results['has_inf']:
        quality_score -= 0.5
    if validation_results['std'] < 0.01:  # Too little variance
        quality_score -= 0.2
    if validation_results['std'] > 10.0:  # Too much variance
        quality_score -= 0.2
    
    validation_results['quality_score'] = max(0.0, quality_score)
    
    return validation_results
