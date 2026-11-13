from ase.io import read
import numpy as np

pool = read("ablation_results_Hookean/coupled_seed42/candidate_pool.extxyz", index=":")
minima = [a for a in pool if a.info.get("source") == "minimum"]

print(f"Total minima: {len(minima)}")
print(f"\nFirst basin (initial):")
m = minima[0]
slab_top = m.get_positions()[:32, 2].max()
ads = m.get_positions()[32:]
print(f"  slab_top z = {slab_top:.2f}")
print(f"  ads z heights above slab: {[f'{p[2]-slab_top:.2f}' for p in ads]}")
print(f"  ads species: {[m.symbols[i+32] for i in range(len(ads))]}")

print(f"\nLast basin (E should be ~-341):")
m = minima[-1]
slab_top = m.get_positions()[:32, 2].max()
ads = m.get_positions()[32:]
E = m.info.get('energy', 'N/A')
print(f"  E = {E}")
print(f"  ads z heights above slab: {[f'{p[2]-slab_top:.2f}' for p in ads]}")
print(f"  ads species: {[m.symbols[i+32] for i in range(len(ads))]}")