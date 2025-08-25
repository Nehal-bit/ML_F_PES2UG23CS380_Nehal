# lab.py
import torch

def get_entropy_of_dataset(tensor: torch.Tensor):
    # Extract target column (last column)
  target_col = tensor[:, -1]
    
    # Get unique classes and counts
  unique_classes, counts = torch.unique(target_col, return_counts=True)
    
    # Compute probabilities
  probs = counts.float() / counts.sum()
    
    # Calculate entropy: -Σ(p * log2(p))
  entropy = -(probs * torch.log2(probs)).sum()
    
  return entropy.item()



def get_avg_info_of_attribute(tensor: torch.Tensor, attribute: int):
  total_len = len(tensor)
  attr_values = tensor[:, attribute]
    
  unique_vals = torch.unique(attr_values)
  avg_info = 0.0
    
  for val in unique_vals:
    subset = tensor[attr_values == val]
    subset_entropy = get_entropy_of_dataset(subset)
    weight = len(subset) / total_len
    avg_info += weight * subset_entropy
    
  return avg_info


def get_information_gain(tensor: torch.Tensor, attribute: int):
  dataset_entropy = get_entropy_of_dataset(tensor)
  avg_info = get_avg_info_of_attribute(tensor, attribute)
  info_gain = dataset_entropy - avg_info
  return round(info_gain, 4)



def get_selected_attribute(tensor: torch.Tensor):
  num_attributes = tensor.shape[1] - 1  # exclude target
  info_gains = {}
    
  for attr in range(num_attributes):
    info_gains[attr] = get_information_gain(tensor, attr)
    
    # Select attribute with max info gain
  best_attr = max(info_gains, key=info_gains.get)
  return info_gains, best_attr
