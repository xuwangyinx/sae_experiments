import itertools

import numpy as np
import torch
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression

from .utils import *


def get_labeled_features(split1, split2, encoder, split, aggregation="max", featurize_batch_size=16):
    # Splits are lists of strings
    split1_indices, split1_acts = encoder.featurize_text(split1, batch_size=featurize_batch_size)[split]
    split2_indices, split2_acts = encoder.featurize_text(split2, batch_size=featurize_batch_size)[split]
    # Convert top k indices, acts to expanded tensor of last dimension n_features
    expanded_split1 = expand_latents(split1_indices, split1_acts, encoder.n_features)[:, 2:, :] # Index out BOS token
    expanded_split2 = expand_latents(split2_indices, split2_acts, encoder.n_features)[:, 2:, :] # Index out BOS token
    features, labels = get_labeled(expanded_split1, expanded_split2, aggregation)
    return features, labels


def get_feature_ranking(features, labels, method="logistic"):
    """
    Find the top features that help distinguish between the distributions
    """
    assert method in ["logistic", "mean_diff", "mutual_info"], "Invalid method"
    if method == "logistic":
        # Convert to numpy for sklearn
        features_np = features.cpu().numpy()
        labels_np = labels.cpu().numpy()
        # Use sklearn's LogisticRegression
        lr = LogisticRegression(fit_intercept=False, max_iter=1000)
        lr.fit(features_np, labels_np)        
        # Get feature importance
        importance = np.abs(lr.coef_[0])
        ranks = np.argsort(importance)[::-1].tolist()  # Sort in descending order
    elif method == "mean_diff":
        # Mean difference method using PyTorch
        pos_mask = labels == 1
        neg_mask = ~pos_mask
        pos_mean = features[pos_mask].mean(dim=0)
        neg_mean = features[neg_mask].mean(dim=0)
        mean_diff = torch.abs(pos_mean - neg_mean)
        ranks = torch.argsort(mean_diff, descending=True).tolist()
    elif method == "mutual_info":
         # Convert to numpy for scikit-learn
        X = features.numpy()
        y = labels.numpy()
        # Mutual information method
        mi = mutual_info_classif(X, y)
        ranks = np.argsort(mi)[::-1]  # Sort in descending order    
    return ranks