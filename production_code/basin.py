from ase.io import read, write
from ase.visualize import view

pool = read("ablation_results_Hookean/coupled_seed42/candidate_pool.extxyz", index=":")
minima = [a for a in pool if a.info.get("source") == "minimum"]

for tag, m in [("first", minima[0]),
               ("mid", minima[len(minima)//2]),
               ("last", minima[-1])]:
    slab_top = m.get_positions()[:32, 2].max()
    ads = m.get_positions()[32:]
    z_above = ads[:, 2] - slab_top
    view(m)
    print(f"\n{tag} basin:")
    print(f"  ads atoms z above slab top: {z_above}")
    print(f"  ads atoms xy:               {ads[:, :2]}")
    write(f"{tag}_basin.extxyz", m)