import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Set
import math
from dataclasses import dataclass

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


@dataclass
class ModelConfig:
    """Configuration for CEDA model hyperparameters"""
    hidden_dim: int = 128
    num_heads: int = 16
    dropout: float = 0.1
    batch_size: int = 256
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    num_epochs: int = 100
    
    lambda1: float = 1.0  # Residual loss weight
    lambda2: float = 1.0  # Diffusion prediction loss weight
    lambda3: float = 1.0  # ILD loss weight
    lambda4: float = 1.0  # CC loss weight
    
    # Intervention thresholds
    theta_m: float = 0.5  # ILD score threshold
    theta_n: int = 3      # Minimum neighbor threshold
    theta_c: float = 0.3  # Content diversity threshold

class UserDualModelling(nn.Module):
    def __init__(self, num_users: int, attr_dim: int, hidden_dim: int, max_seq_len: int):
        super().__init__()
        self.attr_dim = attr_dim
        self.hidden_dim = hidden_dim
        
        # Attribute embedding layer
        self.attr_embedding = nn.Linear(attr_dim, hidden_dim)
        
        self.register_buffer(
            'pos_encoding',
            self._create_position_encoding(max_seq_len, hidden_dim)
        )
        
        self.WE = nn.Linear(2 * hidden_dim, hidden_dim)
        
    def _create_position_encoding(self, max_seq_len: int, dim: int) -> torch.Tensor:
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        
        pe = torch.zeros(max_seq_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
    
    def forward(self, user_attributes: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        attr_embeds = self.attr_embedding(user_attributes)
        
        pos_embeds = self.pos_encoding[positions]
        
        combined = torch.cat([attr_embeds, pos_embeds], dim=-1)
        return self.WE(combined)

class CausalTransformer(nn.Module):

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.observation_mapper = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.residual_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.ff_network = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(4 * hidden_dim, hidden_dim)
        )
        
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def compute_residual(self, embeddings: torch.Tensor, 
                        observations: torch.Tensor) -> torch.Tensor:
        predicted = self.predictor(embeddings)
        observed = self.observation_mapper(observations)
        residual = self.residual_mlp(predicted - observed)
        return residual
    
    def causal_attention(self, query: torch.Tensor, key: torch.Tensor, 
                        value: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        Q = self.query_proj(query).view(-1, self.num_heads, self.hidden_dim // self.num_heads)
        K = self.key_proj(key).view(-1, self.num_heads, self.hidden_dim // self.num_heads)
        V = self.value_proj(value).view(-1, self.num_heads, self.hidden_dim // self.num_heads)
        R = residual.view(-1, self.num_heads, self.hidden_dim // self.num_heads)
        
        # Compute attention scores with causal adjustment
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.hidden_dim // self.num_heads)
        scores = scores - torch.matmul(Q, R.transpose(-2, -1))
        
        # Apply softmax and dropout
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # Compute weighted sum and project
        output = torch.matmul(attention, V)
        output = output.reshape(-1, self.hidden_dim)
        return self.output_proj(output)
    
    def forward(self, embeddings: torch.Tensor, observations: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual-adjusted attention"""
        # Compute residual embeddings
        residual = self.compute_residual(embeddings, observations)
        
        # Multi-head attention with residual adjustment
        attended = self.causal_attention(embeddings, embeddings, embeddings, residual)
        attended = self.layer_norm1(embeddings + self.dropout(attended))
        
        # Feed forward network
        output = self.ff_network(attended)
        output = self.layer_norm2(attended + self.dropout(output))
        
        return output

class SocialDiffusionPredictor(nn.Module):
    """
    Component 3: Social Diffusion Predictor
    """
    def __init__(self, hidden_dim: int, num_categories: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_categories = num_categories
        
        self.diffusion_predictor = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.category_predictor = nn.Linear(hidden_dim, num_categories)
        
    def predict_diffusion(self, source_embeds: torch.Tensor, 
                         target_embeds: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([source_embeds, target_embeds], dim=-1)
        return self.diffusion_predictor(combined)
    
    def predict_categories(self, embeds: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.category_predictor(embeds))
    
    def compute_diversity_metrics(self, embeds: torch.Tensor, 
                                cascade_indices: List[List[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        ild_scores = []
        for cascade in cascade_indices:
            if len(cascade) < 2:
                continue
            cascade_embeds = embeds[cascade]
            similarities = F.cosine_similarity(
                cascade_embeds.unsqueeze(1),
                cascade_embeds.unsqueeze(0),
                dim=-1
            )
            ild = 1 - (similarities.sum() - len(cascade)) / (len(cascade) * (len(cascade) - 1))
            ild_scores.append(ild)
        ild_score = torch.stack(ild_scores).mean() if ild_scores else torch.tensor(0.0)
        
        categories = self.predict_categories(embeds)
        category_coverage = (categories > 0.5).float().sum(dim=0)
        cc_score = (category_coverage > 0).float().sum() / self.num_categories
        
        return ild_score, cc_score

class TargetedInterventions:
    """Component 4: Targeted Interventions"""
    def __init__(self, config: ModelConfig):
        self.config = config
        
    def identify_low_diversity_clusters(self, embeddings: torch.Tensor, 
                                      graph: nx.Graph,
                                      diffusion_predictor: SocialDiffusionPredictor) -> List[int]:
        clusters = list(nx.community.greedy_modularity_communities(graph))
        low_diversity = []
        
        for i, cluster in enumerate(clusters):
            cluster_embeds = embeddings[list(cluster)]
            ild_score = diffusion_predictor.compute_ild(cluster_embeds, [range(len(cluster))])
            
            if ild_score < self.config.theta_m:
                low_diversity.append(i)
                
        return low_diversity
    
    def identify_bottleneck_users(self, embeddings: torch.Tensor,
                                graph: nx.Graph,
                                diffusion_predictor: SocialDiffusionPredictor) -> Set[int]:
        bottleneck_users = set()
        communities = list(nx.community.greedy_modularity_communities(graph))
        
        for user in graph.nodes():
            # Check minimum neighbor threshold
            user_communities = set()
            for i, comm in enumerate(communities):
                if user in comm:
                    user_communities.add(i)
                    
            if len(user_communities) >= self.config.theta_n:
                # Check content diversity
                user_embed = embeddings[user].unsqueeze(0)
                cc_score = diffusion_predictor.compute_category_coverage(user_embed)
                
                if cc_score < self.config.theta_c:
                    bottleneck_users.add(user)
                    
        return bottleneck_users
    
    def apply_diversity_injection(self, graph: nx.Graph, 
                                low_diversity_clusters: List[int],
                                embeddings: torch.Tensor) -> nx.Graph:
        """Apply diversity-aware content injection strategy"""
        modified_graph = graph.copy()
        communities = list(nx.community.greedy_modularity_communities(graph))
        
        for cluster_idx in low_diversity_clusters:
            cluster = communities[cluster_idx]
            
            # Find highest degree node
            highest_degree_node = max(cluster, key=lambda n: graph.degree(n))
            
            # Remove one input edge
            in_edges = list(graph.in_edges(highest_degree_node))
            if in_edges:
                edge_to_remove = max(in_edges, 
                                   key=lambda e: graph.degree(e[0]))
                modified_graph.remove_edge(*edge_to_remove)
            
            # Add diverse edge from outside cluster
            other_clusters = set(range(len(communities))) - {cluster_idx}
            if other_clusters:
                # Select most diverse source cluster
                source_cluster_idx = max(other_clusters,
                    key=lambda i: F.cosine_similarity(
                        embeddings[list(communities[i])].mean(0),
                        embeddings[list(cluster)].mean(0),
                        dim=0
                    )
                )
                source_node = max(communities[source_cluster_idx],
                                key=lambda n: graph.degree(n))
                modified_graph.add_edge(source_node, highest_degree_node)
        
        return modified_graph
    
    def apply_cross_category_bridging(self, graph: nx.Graph,
                                    bottleneck_users: Set[int],
                                    embeddings: torch.Tensor,
                                    diffusion_predictor: SocialDiffusionPredictor) -> nx.Graph:
        """Apply cross-category bridging strategy"""
        modified_graph = graph.copy()
        
        for user in bottleneck_users:
            user_embed = embeddings[user].unsqueeze(0)
            user_categories = diffusion_predictor.predict_categories(user_embed)
            
            # Find users with complementary categories
            potential_targets = []
            for other in graph.nodes():
                if other != user and other not in graph.neighbors(user):
                    other_embed = embeddings[other].unsqueeze(0)
                    other_categories = diffusion_predictor.predict_categories(other_embed)
                    
                    category_complement = torch.sum(
                        user_categories * (1 - other_categories)
                    ).item()
                    
                    if category_complement > 0:
                        potential_targets.append((other, category_complement))
            
            # Add edges to most complementary users
            potential_targets.sort(key=lambda x: x[1], reverse=True)
            for target, _ in potential_targets[:self.config.cross_category_links]:
                modified_graph.add_edge(user, target)
        
        return modified_graph
    
    def apply_interventions(self, graph: nx.Graph,
                          embeddings: torch.Tensor,
                          diffusion_predictor: SocialDiffusionPredictor) -> nx.Graph:
        """Apply both intervention strategies sequentially"""
        # Identify intervention points
        low_diversity_clusters = self.identify_low_diversity_clusters(
            embeddings, graph, diffusion_predictor
        )
        
        # Apply diversity injection
        modified_graph = self.apply_diversity_injection(
            graph, low_diversity_clusters, embeddings
        )
        
        # Identify bottleneck users in modified graph
        bottleneck_users = self.identify_bottleneck_users(
            embeddings, modified_graph, diffusion_predictor
        )
        
        # Apply cross-category bridging
        final_graph = self.apply_cross_category_bridging(
            modified_graph, bottleneck_users, embeddings, diffusion_predictor
        )
        
        return final_graph

class CEDA(nn.Module):
    """Complete CEDA model integrating all components"""
    def __init__(self, config: ModelConfig, num_users: int, attr_dim: int, 
                 max_seq_len: int, num_categories: int):
        super().__init__()
        self.config = config
        
        # Initialize components
        self.user_modelling = UserDualModelling(
            num_users=num_users,
            attr_dim=attr_dim,
            hidden_dim=config.hidden_dim,
            max_seq_len=max_seq_len
        )
        
        self.causal_transformer = CausalTransformer(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout
        )
        
        self.diffusion_predictor = SocialDiffusionPredictor(
            hidden_dim=config.hidden_dim,
            num_categories=num_categories
        )
        
        self.interventions = TargetedInterventions(config)
        
    def forward(self, user_attributes: torch.Tensor,
                positions: torch.Tensor,
                observations: torch.Tensor,
                graph: nx.Graph = None) -> Dict[str, torch.Tensor]:
        """Complete forward pass through all components"""
        # User Dual Modelling
        comprehensive_embeds = self.user_modelling(user_attributes, positions)
        
        # Causal Transformer
        unbiased_embeds = self.causal_transformer(comprehensive_embeds, observations)
        
        # Social Diffusion Prediction
        diffusion_probs = self.diffusion_predictor.predict_diffusion(
            unbiased_embeds.unsqueeze(1),
            unbiased_embeds.unsqueeze(0)
        )
        
        ild_score, cc_score = self.diffusion_predictor.compute_diversity_metrics(
            unbiased_embeds,
            [range(len(user_attributes))]
        )
        
        # Apply interventions if graph is provided
        modified_graph = None
        if graph is not None:
            modified_graph = self.interventions.apply_interventions(
                graph, unbiased_embeds, self.diffusion_predictor
            )
        
        return {
            'comprehensive_embeds': comprehensive_embeds,
            'unbiased_embeds': unbiased_embeds,
            'diffusion_probs': diffusion_probs,
            'ild_score': ild_score,
            'cc_score': cc_score,
            'modified_graph': modified_graph
        }