'''
This script contains the base densities (or "unique shapes") chosen for the MDS map.

Using the unique shapes, the unique pairs needed to determine all pairwise distances of
    the (transformed) base densities are determined and written to `unique_pairs.json`.
'''


import json
import os
import random
from itertools import combinations_with_replacement

from distance_calculation_campaign.copula_families import (load_target, MixtureCopula, MirroredCopula,
                             KhoudrajiCopula, GluedCopula)


sigma = 0.26

# Shapes sorted by symmetry group (0=Group A, 1=Group B, ...):
unique_shapes_category_0 = {
    "(khoudraji_gumbel6_0.4)": KhoudrajiCopula(load_target("GumbelCopula", 6, sigma), None, shape=(0.4, 0)),
    "(khoudraji_gaussian0.9_0.4)": KhoudrajiCopula(load_target("GaussianCopula", 0.9, sigma), None, shape=(0.4, 0)),
    "(khoudraji_frank25_0.4)": KhoudrajiCopula(load_target("FrankCopula", 25, sigma), None, shape=(0.4, 0)),
}

unique_shapes_category_1 = {
    "(mix_gaussian0.6_clayton10_75)": MixtureCopula([("GaussianCopula", 0.6), ("ClaytonCopula", 10)], [0.75, 0.25], sigma),
    "(mix_gaussian0.6_clayton10_50)": MixtureCopula([("GaussianCopula", 0.6), ("ClaytonCopula", 10)], [0.5, 0.5], sigma),

    "(clayton1)": load_target("ClaytonCopula", 1, sigma),
    "(clayton4)": load_target("ClaytonCopula", 4, sigma),
    "(clayton10)": load_target("ClaytonCopula", 10, sigma),

    "(gumbel2)": load_target("GumbelCopula", 2, sigma),
    "(gumbel4)": load_target("GumbelCopula", 4, sigma),
    "(gumbel6)": load_target("GumbelCopula", 6, sigma),

    "(joe6)": load_target("JoeCopula", 6, sigma),
}

unique_shapes_category_2 = {
    "(mix_clayton4_clayton4@u0u1)": MixtureCopula([     load_target("ClaytonCopula", 4, sigma),
                        MirroredCopula(load_target("ClaytonCopula", 4, sigma), (0, 1))],
                        [0.5, 0.5], sigma),
    "(mix_clayton10_clayton10@u0u1)": MixtureCopula([     load_target("ClaytonCopula", 10, sigma),
                        MirroredCopula(load_target("ClaytonCopula", 10, sigma), (0, 1))],
                        [0.5, 0.5], sigma),

    "(gaussian0.99)": load_target("GaussianCopula", 0.99, sigma),
    "(gaussian0.9)": load_target("GaussianCopula", 0.9, sigma),
    "(gaussian0.6)": load_target("GaussianCopula", 0.6, sigma),
    "(gaussian0.3)": load_target("GaussianCopula", 0.3, sigma),

    "(frank4)": load_target("FrankCopula", 4, sigma),
    "(frank10)": load_target("FrankCopula", 10, sigma),
    "(frank25)": load_target("FrankCopula", 25, sigma),

    "(plackett5)": load_target("PlackettCopula", 5, sigma),
    "(plackett20)": load_target("PlackettCopula", 20, sigma),
    "(plackett50)": load_target("PlackettCopula", 50, sigma),

    "(studentt0.5_2)": load_target("StudentTCopula", (0.5, 2), sigma),
    "(studentt0.5_10)": load_target("StudentTCopula", (0.5, 10), sigma),
    "(studentt0.8_2)": load_target("StudentTCopula", (0.8, 2), sigma),
    "(studentt0.8_10)": load_target("StudentTCopula", (0.8, 10), sigma),
}

unique_shapes_category_3 = {
    "(glued_frank8@u0_frank8)": GluedCopula(
                        MirroredCopula(load_target("FrankCopula", 8, sigma), 0),
                        load_target("FrankCopula", 8, sigma),
                        0.5, 0),

    "(glued_khoudraji_clayton8@u1_u0u1)": GluedCopula(
                        MirroredCopula(KhoudrajiCopula(load_target("ClaytonCopula", 8, sigma), None, shape=(0.5, 0.0)), 1),
                        MirroredCopula(KhoudrajiCopula(load_target("ClaytonCopula", 8, sigma), None, shape=(0.5, 0.0)), (0, 1)),
                        0.5, 0),
}

unique_shapes_category_4 = {
    "(independence)": load_target("IndependenceCopula", None, sigma),
}

unique_shapes = {   **unique_shapes_category_0,
                    **unique_shapes_category_1, 
                    **unique_shapes_category_2,
                    **unique_shapes_category_3,
                    **unique_shapes_category_4}


# Using the tricks in Section 3.2 of the thesis, the needed pairwise distances are determined.
# First, for two base distributions within a symmetry group:
pairs_to_be_calculated = []

for name_a, name_b in combinations_with_replacement(unique_shapes_category_0, 2):
    if name_a != name_b:
        pairs_to_be_calculated.append((name_a, name_b))
        pairs_to_be_calculated.append((name_a, name_b + "!T!M0"))
    pairs_to_be_calculated.append((name_a, name_b + "!M0"))
    pairs_to_be_calculated.append((name_a, name_b + "!T"))
    pairs_to_be_calculated.append((name_a, name_b + "!T!M01"))
    pairs_to_be_calculated.append((name_a, name_b + "!T!M1"))
    pairs_to_be_calculated.append((name_a, name_b + "!M1"))
    pairs_to_be_calculated.append((name_a, name_b + "!M01"))

for name_a, name_b in combinations_with_replacement(unique_shapes_category_1, 2):
    if name_a != name_b:
        pairs_to_be_calculated.append((name_a, name_b))
    pairs_to_be_calculated.append((name_a, name_b + "!M0"))
    pairs_to_be_calculated.append((name_a, name_b + "!M01"))

for name_a, name_b in combinations_with_replacement(unique_shapes_category_2, 2):
    if name_a != name_b:
        pairs_to_be_calculated.append((name_a, name_b))
    pairs_to_be_calculated.append((name_a, name_b + "!M0"))

for name_a, name_b in combinations_with_replacement(unique_shapes_category_3, 2):
    if name_a != name_b:
        pairs_to_be_calculated.append((name_a, name_b))
    pairs_to_be_calculated.append((name_a, name_b + "!T"))
    pairs_to_be_calculated.append((name_a, name_b + "!M1"))

for name_a, name_b in combinations_with_replacement(unique_shapes_category_4, 2):
    if name_a != name_b:
        pairs_to_be_calculated.append((name_a, name_b))


# Second, for base distributions in different symmetry groups, the necessary pairs are determined:
for name_a in unique_shapes_category_0:

    for name_b in unique_shapes_category_1:
        pairs_to_be_calculated.append((name_a, name_b))
        pairs_to_be_calculated.append((name_a, name_b + "!M0"))
        pairs_to_be_calculated.append((name_a, name_b + "!M1"))
        pairs_to_be_calculated.append((name_a, name_b + "!M01"))

    for name_b in unique_shapes_category_2:
        pairs_to_be_calculated.append((name_a, name_b))
        pairs_to_be_calculated.append((name_a, name_b + "!M0"))

    for name_b in unique_shapes_category_3:
        pairs_to_be_calculated.append((name_a, name_b))
        pairs_to_be_calculated.append((name_a, name_b + "!M1"))
        pairs_to_be_calculated.append((name_a, name_b + "!T"))
        pairs_to_be_calculated.append((name_a, name_b + "!T!M0"))

    for name_b in unique_shapes_category_4:
        pairs_to_be_calculated.append((name_a, name_b))

for name_a in unique_shapes_category_1:

    for name_b in unique_shapes_category_2:
        pairs_to_be_calculated.append((name_a, name_b))
        pairs_to_be_calculated.append((name_a, name_b + "!M0"))

    for name_b in unique_shapes_category_3:
        pairs_to_be_calculated.append((name_a, name_b))
        pairs_to_be_calculated.append((name_a, name_b + "!M1"))

    for name_b in unique_shapes_category_4:
        pairs_to_be_calculated.append((name_a, name_b))

for name_a in unique_shapes_category_2:

    for name_b in unique_shapes_category_3:
        pairs_to_be_calculated.append((name_a, name_b))

    for name_b in unique_shapes_category_4:
        pairs_to_be_calculated.append((name_a, name_b))

for name_a in unique_shapes_category_3:

    for name_b in unique_shapes_category_4:
        pairs_to_be_calculated.append((name_a, name_b))


# The pairs are used to create a json file where they are all listed. 
# One by one, the corrsponding distances will later be filled in.
random.Random(0).shuffle(pairs_to_be_calculated)

pairs_to_be_calculated_text = [a + "/" + b for (a,b) in pairs_to_be_calculated]

unique_pairs = [{"pair": pair, "status": "queued"}
                for pair in dict.fromkeys(pairs_to_be_calculated_text)]

if __name__ == "__main__":
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "unique_pairs.json"), "w") as f:
        json.dump(unique_pairs, f, indent=2)
        f.write("\n")


# (Helper function)
def get_cat(copula):
    copula = copula.split("!")[0]

    if copula in unique_shapes_category_0:
        return 0
    if copula in unique_shapes_category_1:
        return 1
    if copula in unique_shapes_category_2:
        return 2
    if copula in unique_shapes_category_3:
        return 3
    if copula in unique_shapes_category_4:
        return 4
    else:
        return None    