import numpy as np
from scipy.sparse import load_npz
from sklearn.decomposition import TruncatedSVD

# Parameters
SPARSE_MATRIX_FILE = "Rname_sparse.npz"
N_COMPONENTS = 128  # Change this if you want more/fewer dimensions

# Load matrix
print("Loading sparse matrix...")
Rname = load_npz(SPARSE_MATRIX_FILE)

# Run Truncated SVD
print(f"Running Truncated SVD with {N_COMPONENTS} components...")
svd = TruncatedSVD(n_components=N_COMPONENTS, random_state=42)
U = svd.fit_transform(Rname)    # (num_entities, k)
Vt = svd.components_            # (k, num_features)

# Save matrices
np.save("Rname_U.npy", U)
np.save("Rname_Vt.npy", Vt)

print("SVD complete. Matrices saved:")
print(f"  - U shape: {U.shape} -> saved as Rname_U.npy")
print(f"  - Vt shape: {Vt.shape} -> saved as Rname_Vt.npy")
