import numpy as np

# --- BASE METHODS (accept single candidate + args)
def _normalize_single(candidate, args):
    weights = np.maximum(candidate, 0)
    total = np.sum(weights)
    return (weights / total).tolist() if total != 0 else (np.ones_like(weights) / len(weights)).tolist()

def _clip_single(candidate, args):
    weights = np.clip(candidate, 0, 0.1)
    total = np.sum(weights)
    return (weights / total).tolist() if total != 0 else (np.ones_like(weights) / len(weights)).tolist()

def _restart_single(candidate, args):
    if np.sum(candidate) == 0:
        weights = np.random.rand(len(candidate))
        return (weights / np.sum(weights)).tolist()
    return _normalize_single(candidate, args)

def _shrink_single(candidate, args):
    weights = np.clip(candidate, 0, None)  # no short-selling
    total = np.sum(weights)
    if total > 1.0:
        weights = weights / total  # shrink to sum=1
    weights = np.clip(weights, 0, 0.1)  # position limit
    final_total = np.sum(weights)
    return (weights / final_total).tolist() if final_total != 0 else (np.ones_like(weights) / len(weights)).tolist()


# --- WRAPPERS FOR PSO (expect candidate, args)
def repair_normalize(candidate, args): return _normalize_single(candidate, args)
def repair_clipped_normalize(candidate, args): return _clip_single(candidate, args)
def repair_random_restart(candidate, args): return _restart_single(candidate, args)
def repair_shrink(candidate, args): return _shrink_single(candidate, args)

# --- WRAPPERS FOR GA (expect list of candidates, random, args)
def repair_normalize_ga(random, candidates, args): return [repair_normalize(c, args) for c in candidates]
def repair_clipped_normalize_ga(random, candidates, args): return [repair_clipped_normalize(c, args) for c in candidates]
def repair_random_restart_ga(random, candidates, args): return [repair_random_restart(c, args) for c in candidates]
def repair_shrink_ga(random, candidates, args): return [repair_shrink(c, args) for c in candidates]

# Dicts for lookup
REPAIR_METHODS_PSO = {
    "normalize": repair_normalize,
    "clip": repair_clipped_normalize,
    "restart": repair_random_restart,
    "shrink": repair_shrink
}

REPAIR_METHODS_GA = {
    "normalize": repair_normalize_ga,
    "clip": repair_clipped_normalize_ga,
    "restart": repair_random_restart_ga,
    "shrink": repair_shrink_ga
}