import numpy as np


def add_noise(sequence, noise_std=3.0):
    """Add small Gaussian noise to angle values (in degrees)."""
    noise = np.random.normal(0, noise_std, sequence.shape)
    return np.clip(sequence + noise, 0, 180)


def time_warp(sequence, warp_factor_range=(0.85, 1.15)):
    """Stretch or compress the sequence in time, then resample back to original length."""
    seq_len = sequence.shape[0]
    warp_factor = np.random.uniform(*warp_factor_range)
    new_len = max(5, int(seq_len * warp_factor))

    old_idx = np.linspace(0, seq_len - 1, seq_len)
    new_idx = np.linspace(0, seq_len - 1, new_len)

    warped = np.zeros((new_len, sequence.shape[1]))
    for col in range(sequence.shape[1]):
        warped[:, col] = np.interp(new_idx, old_idx, sequence[:, col])

    # resample back to original seq_len so shape stays consistent
    final_idx = np.linspace(0, new_len - 1, seq_len).astype(int)
    return warped[final_idx]


def mirror_lr(sequence):
    """
    Swap left/right angle columns.
    Assumes column order: [l_elbow, r_elbow, l_shoulder, r_shoulder, l_hip, r_hip, l_knee, r_knee]
    """
    mirrored = sequence.copy()
    swap_pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
    for l_idx, r_idx in swap_pairs:
        mirrored[:, [l_idx, r_idx]] = mirrored[:, [r_idx, l_idx]]
    return mirrored


def scale_angles(sequence, scale_range=(0.95, 1.05)):
    """Slightly scale all angle values to simulate body-proportion variation."""
    scale = np.random.uniform(*scale_range)
    return np.clip(sequence * scale, 0, 180)


def random_frame_dropout(sequence, drop_prob=0.1):
    """Randomly drop a few frames, then resample back to original length."""
    seq_len = sequence.shape[0]
    keep_mask = np.random.rand(seq_len) > drop_prob
    if keep_mask.sum() < 5:  # safety: never drop almost everything
        keep_mask[:5] = True

    kept = sequence[keep_mask]
    idxs = np.linspace(0, len(kept) - 1, seq_len).astype(int)
    return kept[idxs]


def augment_sequence(sequence):
    """Apply a random combination of augmentations to one sequence."""
    aug = sequence.copy()

    if np.random.rand() < 0.7:
        aug = add_noise(aug)
    if np.random.rand() < 0.5:
        aug = time_warp(aug)
    if np.random.rand() < 0.3:
        aug = mirror_lr(aug)
    if np.random.rand() < 0.5:
        aug = scale_angles(aug)
    if np.random.rand() < 0.3:
        aug = random_frame_dropout(aug)

    return aug