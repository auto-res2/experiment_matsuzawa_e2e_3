import torch
import numpy as np
import matplotlib.pyplot as plt
import time

class DummyPolicy:
    """Dummy policy to simulate RL PPO behavior for ADPO+"""
    
    def __init__(self, action_dim=3):
        self.action_dim = action_dim
        self.action_history = []

    def predict(self, state):
        """Predict action given state representation"""
        noise_mod = np.random.uniform(0.8, 1.2)
        step_mod = np.random.uniform(0.8, 1.2)
        skip_flag = np.random.uniform(0.0, 1.0)
        
        action = (noise_mod, step_mod, skip_flag)
        self.action_history.append(action)
        return action

    def get_action_statistics(self):
        """Get statistics about actions taken"""
        if not self.action_history:
            return {}
        
        actions = np.array(self.action_history)
        return {
            'noise_mod_mean': np.mean(actions[:, 0]),
            'noise_mod_std': np.std(actions[:, 0]),
            'step_mod_mean': np.mean(actions[:, 1]),
            'step_mod_std': np.std(actions[:, 1]),
            'skip_flag_mean': np.mean(actions[:, 2]),
            'skip_flag_std': np.std(actions[:, 2]),
            'total_actions': len(self.action_history)
        }

def evaluate_model_quality(final_latent, quality_history):
    """Evaluate the quality of generated samples"""
    metrics = {}
    
    metrics['final_variance'] = final_latent.var().item()
    metrics['final_mean'] = final_latent.mean().item()
    metrics['final_std'] = final_latent.std().item()
    
    if quality_history:
        metrics['initial_quality'] = quality_history[0]
        metrics['final_quality'] = quality_history[-1]
        metrics['quality_improvement'] = (quality_history[-1] - quality_history[0]) / quality_history[0] * 100
        metrics['max_quality'] = max(quality_history)
        metrics['min_quality'] = min(quality_history)
        metrics['quality_stability'] = np.std(quality_history)
    
    return metrics

def calculate_fid_score(latent_quality_avg):
    """Simulate FID score calculation (lower is better)"""
    fid = np.abs(latent_quality_avg - 10) * 0.8
    return fid

def calculate_ssim_score(latent_quality_avg):
    """Simulate SSIM score calculation (higher is better)"""
    ssim = np.tanh(latent_quality_avg / 10) * 1.1
    return np.clip(ssim, 0, 1)

def calculate_clip_similarity():
    """Simulate CLIP similarity score for conditional generation"""
    return np.clip(np.random.normal(0.7, 0.1), 0.5, 0.9)

def evaluate_sampling_efficiency(baseline_time, baseline_iterations, adpo_time, adpo_iterations):
    """Evaluate sampling efficiency improvements"""
    efficiency_metrics = {}
    
    efficiency_metrics['time_speedup'] = baseline_time / adpo_time if adpo_time > 0 else 1.0
    efficiency_metrics['time_reduction_percent'] = (baseline_time - adpo_time) / baseline_time * 100
    
    efficiency_metrics['iteration_reduction'] = baseline_iterations - adpo_iterations
    efficiency_metrics['iteration_reduction_percent'] = (baseline_iterations - adpo_iterations) / baseline_iterations * 100
    
    efficiency_metrics['efficiency_score'] = (efficiency_metrics['time_speedup'] + 
                                            efficiency_metrics['iteration_reduction_percent'] / 100) / 2
    
    return efficiency_metrics

def evaluate_noise_robustness(rl_actions, quality_history):
    """Evaluate robustness to noise perturbations"""
    robustness_metrics = {}
    
    if rl_actions:
        actions = np.array(rl_actions)
        
        robustness_metrics['noise_mod_diversity'] = np.std(actions[:, 0])
        robustness_metrics['step_mod_diversity'] = np.std(actions[:, 1])
        robustness_metrics['skip_rate'] = (actions[:, 2] > 0.5).mean()
        
        robustness_metrics['action_variance'] = np.var(actions, axis=0).mean()
        
    if quality_history:
        robustness_metrics['quality_variance'] = np.var(quality_history)
        robustness_metrics['quality_trend'] = np.polyfit(range(len(quality_history)), quality_history, 1)[0]
    
    return robustness_metrics

def generate_evaluation_report(baseline_results, adpo_results, experiment_type="general"):
    """Generate comprehensive evaluation report"""
    print(f"\n{'='*60}")
    print(f"EVALUATION REPORT - {experiment_type.upper()}")
    print(f"{'='*60}")
    
    print("\nQUALITY METRICS:")
    print(f"Baseline FID: {baseline_results.get('fid', 'N/A'):.4f}")
    print(f"ADPO+ FID: {adpo_results.get('fid', 'N/A'):.4f}")
    print(f"Baseline SSIM: {baseline_results.get('ssim', 'N/A'):.4f}")
    print(f"ADPO+ SSIM: {adpo_results.get('ssim', 'N/A'):.4f}")
    
    print("\nEFFICIENCY METRICS:")
    print(f"Baseline Iterations: {baseline_results.get('iterations', 'N/A')}")
    print(f"ADPO+ Iterations: {adpo_results.get('iterations', 'N/A')}")
    print(f"Baseline Time: {baseline_results.get('time', 'N/A'):.4f}s")
    print(f"ADPO+ Time: {adpo_results.get('time', 'N/A'):.4f}s")
    
    if 'fid' in baseline_results and 'fid' in adpo_results:
        fid_improvement = (baseline_results['fid'] - adpo_results['fid']) / baseline_results['fid'] * 100
        print(f"FID Improvement: {fid_improvement:.1f}%")
    
    if 'iterations' in baseline_results and 'iterations' in adpo_results:
        iter_reduction = (baseline_results['iterations'] - adpo_results['iterations']) / baseline_results['iterations'] * 100
        print(f"Iteration Reduction: {iter_reduction:.1f}%")
    
    print(f"{'='*60}")

def run_evaluation_suite(sampler, test_data, num_samples=10):
    """Run comprehensive evaluation suite"""
    print("Running evaluation suite...")
    
    results = {
        'samples': [],
        'times': [],
        'iterations': [],
        'qualities': []
    }
    
    for i in range(num_samples):
        print(f"Evaluating sample {i+1}/{num_samples}")
        
        start_time = time.time()
        final_latent, iterations, quality_history, _ = sampler.sample(test_data)
        end_time = time.time()
        
        results['samples'].append(final_latent)
        results['times'].append(end_time - start_time)
        results['iterations'].append(iterations)
        results['qualities'].append(quality_history)
    
    aggregate_results = {
        'avg_time': np.mean(results['times']),
        'std_time': np.std(results['times']),
        'avg_iterations': np.mean(results['iterations']),
        'std_iterations': np.std(results['iterations']),
        'avg_final_quality': np.mean([q[-1] for q in results['qualities']]),
        'std_final_quality': np.std([q[-1] for q in results['qualities']])
    }
    
    print("Evaluation suite completed!")
    return results, aggregate_results
