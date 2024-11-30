import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import networkx as nx
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple
import numpy as np
from tqdm import tqdm
import json
import logging
from pathlib import Path

class SocialNetworkDataset(Dataset):
    def __init__(self, data_path: str, min_interactions: int = 5):

        self.data = pd.read_csv(data_path)
        
        interaction_counts = self.data.groupby('user_id').size()
        valid_users = interaction_counts[interaction_counts >= min_interactions].index
        self.data = self.data[self.data['user_id'].isin(valid_users)]
        
        # Create network graph
        self.graph = self._create_social_network()
        
        # Process attributes and sequences
        self.user_attributes = self._process_attributes()
        self.sequences = self._create_sequences()
        
        # Compute network statistics
        self.avg_clustering_coeff = nx.average_clustering(self.graph)
        self.diameter = nx.diameter(self.graph)
        
    def _create_social_network(self) -> nx.Graph:
        G = nx.Graph()
        edges = self.data[['source_user', 'target_user']].values
        G.add_edges_from(edges)
        return G
    
    def _process_attributes(self) -> np.ndarray:
        categorical_cols = ['gender', 'age_group']  
        numerical_cols = ['activity_score'] 
        
        # Process categorical features
        categorical_data = pd.get_dummies(self.data[categorical_cols])
        
        # Combine with numerical features
        attribute_data = pd.concat([
            categorical_data,
            self.data[numerical_cols]
        ], axis=1)
        
        return attribute_data.values
    
    def _create_sequences(self) -> List[Dict]:
        sequences = []
        for user_id, group in self.data.groupby('source_user'):
            interactions = group.sort_values('timestamp').values
            cascade_ids = group['cascade_id'].unique()
            
            sequences.append({
                'user_id': user_id,
                'interactions': interactions,
                'cascade_ids': cascade_ids
            })
        return sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq = self.sequences[idx]
        return {
            'user_id': torch.LongTensor([seq['user_id']]),
            'attributes': torch.FloatTensor(self.user_attributes[idx]),
            'interactions': torch.LongTensor(seq['interactions']),
            'positions': torch.arange(len(seq['interactions'])),
            'cascade_ids': torch.LongTensor(seq['cascade_ids'])
        }

class Evaluator:
    def __init__(self, k_values: List[int] = [5, 10, 20, 40]):
        self.k_values = k_values
    
    def compute_rmse(self, predictions: torch.Tensor, ground_truth: torch.Tensor) -> float:
        return torch.sqrt(F.mse_loss(predictions, ground_truth)).item()
    
    def compute_precision_recall_at_k(self, predictions: torch.Tensor, 
                                    ground_truth: torch.Tensor,
                                    k: int) -> Tuple[float, float]:
        _, top_k_indices = torch.topk(predictions, k, dim=1)
        
        precision_scores = []
        recall_scores = []
        
        for i, indices in enumerate(top_k_indices):
            true_pos = torch.sum(ground_truth[i][indices]).item()
            precision = true_pos / k
            recall = true_pos / torch.sum(ground_truth[i]).item()
            precision_scores.append(precision)
            recall_scores.append(recall)
            
        return np.mean(precision_scores), np.mean(recall_scores)
    
    def compute_gini_coefficient(self, predictions: torch.Tensor) -> float:
        sorted_preds = torch.sort(predictions.view(-1))[0]
        n = len(sorted_preds)
        index = torch.arange(1, n + 1, dtype=torch.float)
        return ((2 * torch.sum(index * sorted_preds)) / 
                (n * torch.sum(sorted_preds)) - (n + 1) / n).item()
    
    def compute_diversity_metrics(self, model_outputs: Dict[str, torch.Tensor]) -> Dict[str, float]:
        metrics = {}
        
        # ILD Score
        metrics['ild_score'] = model_outputs['ild_score'].item()
        
        # Category Coverage
        metrics['category_coverage'] = model_outputs['cc_score'].item()
        
        # Simpson's Diversity Index
        embeddings = model_outputs['unbiased_embeds']
        similarities = torch.matmul(embeddings, embeddings.transpose(0, 1))
        n = len(embeddings)
        metrics['sdi'] = (1 - (torch.sum(similarities) - n) / 
                         (n * (n - 1))).item()
        
        return metrics

class CEDATrainer:
    def __init__(self, model: CEDA, config: ModelConfig):
        self.model = model
        self.config = config
        self.evaluator = Evaluator()
        
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger('CEDA_Training')
        logger.setLevel(logging.INFO)
        return logger
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        epoch_metrics = defaultdict(list)
        
        for batch in tqdm(train_loader, desc="Training"):
            self.optimizer.zero_grad()
            
            outputs = self.model(
                batch['attributes'],
                batch['positions'],
                batch['interactions'],
                batch['graph']
            )
            
            # Compute losses (Eq. 20)
            residual_loss = F.mse_loss(
                outputs['comprehensive_embeds'],
                outputs['unbiased_embeds']
            )
            
            diffusion_loss = F.binary_cross_entropy(
                outputs['diffusion_probs'],
                batch['ground_truth']
            )
            
            diversity_loss = (1 - outputs['ild_score']) + (1 - outputs['cc_score'])
            
            # Combined loss
            total_loss = (self.config.lambda1 * residual_loss +
                         self.config.lambda2 * diffusion_loss +
                         self.config.lambda3 * diversity_loss)
            
            # Backward pass
            total_loss.backward()
            self.optimizer.step()
            
            # Record metrics
            epoch_metrics['total_loss'].append(total_loss.item())
            epoch_metrics['residual_loss'].append(residual_loss.item())
            epoch_metrics['diffusion_loss'].append(diffusion_loss.item())
            epoch_metrics['diversity_loss'].append(diversity_loss.item())
            
        return {k: np.mean(v) for k, v in epoch_metrics.items()}
    
    @torch.no_grad()
    def evaluate(self, eval_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        all_outputs = []
        all_ground_truth = []
        
        for batch in tqdm(eval_loader, desc="Evaluating"):
            outputs = self.model(
                batch['attributes'],
                batch['positions'],
                batch['interactions'],
                batch['graph']
            )
            all_outputs.append(outputs)
            all_ground_truth.append(batch['ground_truth'])
        
        # Combine batch outputs
        combined_outputs = {
            k: torch.cat([out[k] for out in all_outputs]) 
            for k in all_outputs[0].keys()
        }
        ground_truth = torch.cat(all_ground_truth)
        
        metrics = {}
        
        # RMSE
        metrics['rmse'] = self.evaluator.compute_rmse(
            combined_outputs['diffusion_probs'], 
            ground_truth
        )
        
        # Precision and Recall @K
        for k in self.evaluator.k_values:
            precision, recall = self.evaluator.compute_precision_recall_at_k(
                combined_outputs['diffusion_probs'],
                ground_truth,
                k
            )
            metrics[f'precision@{k}'] = precision
            metrics[f'recall@{k}'] = recall
        
        # Diversity metrics
        diversity_metrics = self.evaluator.compute_diversity_metrics(combined_outputs)
        metrics.update(diversity_metrics)
        
        return metrics

def main():
    config = ModelConfig()
    
    datasets = {
        'twitter': 'data/twitter.csv',
        'google_plus': 'data/google_plus.csv',
        'facebook': 'data/facebook.csv'
    }
    
    results = {}
    for dataset_name, data_path in datasets.items():
        print(f"\nTraining on {dataset_name} dataset")
        
        # Load and prepare dataset
        dataset = SocialNetworkDataset(data_path)
        
        train_size = int(0.7 * len(dataset))
        val_size = int(0.1 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config.batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size
        )
        
        # Initialize model
        model = CEDA(
            config=config,
            num_users=len(dataset),
            attr_dim=dataset.user_attributes.shape[1],
            max_seq_len=max(len(seq['interactions']) for seq in dataset.sequences),
            num_categories=dataset.num_categories
        )
        
        # Initialize trainer
        trainer = CEDATrainer(model, config)
        
        # Training loop
        best_val_metrics = None
        for epoch in range(config.num_epochs):
            # Train
            train_metrics = trainer.train_epoch(train_loader)
            
            # Validate
            val_metrics = trainer.evaluate(val_loader)
            
            # Save best model
            if (best_val_metrics is None or 
                val_metrics['rmse'] < best_val_metrics['rmse']):
                best_val_metrics = val_metrics
                torch.save(
                    model.state_dict(),
                    f'models/best_model_{dataset_name}.pt'
                )
            
            # Log progress
            print(f"Epoch {epoch + 1}/{config.num_epochs}")
            print(f"Train Loss: {train_metrics['total_loss']:.4f}")
            print(f"Validation RMSE: {val_metrics['rmse']:.4f}")
        
        # Test final model
        model.load_state_dict(
            torch.save(f'models/best_model_{dataset_name}.pt')
        )
        test_metrics = trainer.evaluate(test_loader)
        
        # Store results
        results[dataset_name] = {
            'test_metrics': test_metrics,
            'network_stats': {
                'avg_clustering_coeff': dataset.avg_clustering_coeff,
                'diameter': dataset.diameter
            }
        }
    
    # Save final results
    with open('results/final_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    # Print final results
    print("\nFinal Results:")
    for dataset_name, result in results.items():
        print(f"\n{dataset_name} Dataset Results:")
        for metric, value in result['test_metrics'].items():
            print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()