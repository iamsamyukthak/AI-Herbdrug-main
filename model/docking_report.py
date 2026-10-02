import re

file = "docking/results/quercetin_CYP3A4_out.pdbqt"

text = open(file).read()

affinities = re.findall(
    r"REMARK VINA RESULT:\s+(-?\d+\.\d+)",
    text
)

print("\n========== MOLECULAR DOCKING ==========")
print("Ligand  : Quercetin")
print("Protein : CYP3A4")

if affinities:
    print("Best affinity:", affinities[0], "kcal/mol")
    print("Number of poses:", len(affinities))
else:
    print("Docking result not found.")

print("\nDocking is supporting biological evidence.")
print("It does not by itself establish clinical interaction.")