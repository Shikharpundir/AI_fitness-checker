import numpy as np
from pathlib import Path
from augment_utils import augment_sequence

DATA_DIR = Path("data/processed_sequences")

X = np.load(DATA_DIR / "X.npy")
y = np.load(DATA_DIR / "y.npy")

print("Original class distribution:")
classes, counts = np.unique(y, return_counts=True)
for c, n in zip(classes, counts):
    print(f"  {c}: {n}")

target_count = counts.max()  # balance everyone up to the largest class
print(f"\nBalancing all classes to {target_count} samples each...\n")

X_balanced = [X]
y_balanced = [y]

for cls, count in zip(classes, counts):
    deficit = target_count - count
    if deficit <= 0:
        continue

    class_indices = np.where(y == cls)[0]
    class_sequences = X[class_indices]

    augmented_samples = []
    for _ in range(deficit):
        base_seq = class_sequences[np.random.randint(len(class_sequences))]
        augmented_samples.append(augment_sequence(base_seq))

    X_balanced.append(np.array(augmented_samples))
    y_balanced.append(np.array([cls] * deficit))

    print(f"  {cls}: added {deficit} augmented samples")

X_final = np.concatenate(X_balanced, axis=0)
y_final = np.concatenate(y_balanced, axis=0)

# shuffle so augmented samples aren't all clustered together
shuffle_idx = np.random.permutation(len(X_final))
X_final = X_final[shuffle_idx]
y_final = y_final[shuffle_idx]

np.save(DATA_DIR / "X_balanced.npy", X_final)
np.save(DATA_DIR / "y_balanced.npy", y_final)

print(f"\nFinal balanced dataset: {X_final.shape[0]} samples")
final_classes, final_counts = np.unique(y_final, return_counts=True)
for c, n in zip(final_classes, final_counts):
    print(f"  {c}: {n}")