
from ase.io import read
import numpy as np
from ase.visualize import view
configs = read('init_Fe110_4H2N_n100.extxyz', index=':')
print(f'Total configs: {len(configs)}')
view(configs)
slab_z_max = 6.986
heights = []
min_dists = []

for atoms in configs:
    syms = atoms.get_chemical_symbols()
    pos = atoms.get_positions()
    ads_idx = [i for i, s in enumerate(syms) if s in ('H', 'N')]
    ads_pos = pos[ads_idx]
    
    # 吸附原子到表面的高度
    heights.extend(ads_pos[:, 2] - slab_z_max)
    
    # 吸附原子之间的最小距离
    if len(ads_pos) >= 2:
        from scipy.spatial.distance import pdist
        min_dists.append(pdist(ads_pos).min())

heights = np.array(heights)
min_dists = np.array(min_dists)

print(f'Adsorbate height above surface (Å):')
print(f'  min={heights.min():.2f}, max={heights.max():.2f}, mean={heights.mean():.2f}')
print(f'Min adsorbate-adsorbate distance per config (Å):')
print(f'  min={min_dists.min():.2f}, max={min_dists.max():.2f}, mean={min_dists.mean():.2f}')