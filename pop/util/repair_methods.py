import numpy as np
from typing import List, Dict, Any

# --- CORE REPAIR METHODS ---
def normalize_weights(weights: np.ndarray) -> np.ndarray:
    """
    Normalize weights to sum=1 with no short-selling.
    
    Args:
        weights: Raw portfolio weights
        
    Returns:
        Normalized weights where all weights ≥ 0 and sum=1
    """
    weights = np.maximum(weights, 0)
    total = weights.sum()
    return weights / total if total > 0 else np.ones_like(weights) / len(weights)

def clip_weights(weights: np.ndarray, max_pos: float = 0.1) -> np.ndarray:
    """
    Clip weights to [0, max_pos] and normalize.
    
    Args:
        weights: Raw portfolio weights
        max_pos: Maximum allowed position size
        
    Returns:
        Clipped & normalized weights
    """
    clipped = np.clip(weights, 0, max_pos)
    return normalize_weights(clipped)

def restart_weights(weights: np.ndarray, 
                   random: np.random.RandomState) -> np.ndarray:
    """
    Generate new random weights if portfolio is invalid.
    
    Args:
        weights: Raw portfolio weights
        random: Controlled random number generator
        
    Returns:
        Valid random weights if input is all zeros, else normalized weights
    """
    if np.all(weights == 0):
        new_weights = random.uniform(0, 1, len(weights))
        return normalize_weights(new_weights)
    return normalize_weights(weights)

def shrink_weights(weights: np.ndarray, 
                  max_pos: float = 0.1) -> np.ndarray:
    """
    Enforce position limits through iterative adjustment.
    
    1. Remove negative weights
    2. If total > 1, scale down
    3. Apply position limits
    4. Re-normalize
    
    Args:
        weights: Raw portfolio weights
        max_pos: Maximum allowed position size
        
    Returns:
        Valid weights respecting all constraints
    """
    # Step 1: No short-selling
    positive_weights = np.maximum(weights, 0)
    
    # Step 2: Cap total at 1
    total = positive_weights.sum()
    if total > 1:
        positive_weights /= total
        
    # Step 3: Apply position limit
    limited_weights = np.clip(positive_weights, 0, max_pos)
    
    # Step 4: Final normalization
    return normalize_weights(limited_weights)

# --- ALGORITHM-SPECIFIC WRAPPERS ---
def pso_repair(candidate: List[float], 
              args: Dict[str, Any]) -> List[float]:
    """
    PSO repair interface.
    
    Args:
        candidate: Individual particle's position
        args: Algorithm parameters with repair method key
        
    Returns:
        Repaired candidate weights
    """
    method = args.get('repair_method', 'normalize')
    weights = np.array(candidate)
    
    if method == 'clip':
        repaired = clip_weights(weights, max_pos=0.1)
    elif method == 'restart':
        random = args.get('random', np.random)
        repaired = restart_weights(weights, random)
    elif method == 'shrink':
        repaired = shrink_weights(weights, max_pos=0.1)
    else:  # Default to normalize
        repaired = normalize_weights(weights)
        
    return repaired.tolist()

def ga_repair(random: np.random.RandomState, 
             candidates: List[List[float]], 
             args: Dict[str, Any]) -> List[List[float]]:
    """
    GA repair interface.
    
    Args:
        random: Controlled random number generator
        candidates: Population of candidate solutions
        args: Algorithm parameters with repair method key
        
    Returns:
        Repaired population
    """
    method = args.get('repair_method', 'normalize')
    repaired = []
    
    for candidate in candidates:
        weights = np.array(candidate)
        
        if method == 'clip':
            new_weights = clip_weights(weights, 0.1)
        elif method == 'restart':
            new_weights = restart_weights(weights, random)
        elif method == 'shrink':
            new_weights = shrink_weights(weights, 0.1)
        else:
            new_weights = normalize_weights(weights)
            
        repaired.append(new_weights.tolist())
        
    return repaired

# --- METHOD REGISTRY ---
REPAIR_METHODS_PSO = {
    "normalize": pso_repair,
    "clip": lambda c, a: pso_repair(c, {**a, 'repair_method': 'clip'}),
    "restart": lambda c, a: pso_repair(c, {**a, 'repair_method': 'restart'}),
    "shrink": lambda c, a: pso_repair(c, {**a, 'repair_method': 'shrink'})
}

REPAIR_METHODS_GA = {
    "normalize": ga_repair,
    "clip": lambda r, c, a: ga_repair(r, c, {**a, 'repair_method': 'clip'}),
    "restart": lambda r, c, a: ga_repair(r, c, {**a, 'repair_method': 'restart'}),
    "shrink": lambda r, c, a: ga_repair(r, c, {**a, 'repair_method': 'shrink'})
}