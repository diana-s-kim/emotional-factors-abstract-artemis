import numpy as np
from scipy.stats import ortho_group
dim_embedding=128
x=ortho_group.rvs(dim_embedding)
np.savez("ortho_128.npz",orthogonal_set_128=x)

