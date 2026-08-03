# 1FAP test system — FKBP12·rapamycin·FRB

A small, real ternary complex used as gluebind's integration / end-to-end test
system. Derived from [PDB 1FAP](https://www.rcsb.org/structure/1FAP) (FKBP12
bound to the FRB domain of mTOR, bridged by the natural-product glue rapamycin).

It is chosen because it is *small* (≈200 residues total, no metal sites) yet a
genuine molecular-glue ternary complex, and because the crystal has each protein
on its own chain with a `TER` between them — exercising the input→complex atom
mapping that guards against BioSimSpace's `TER`-split index remapping.

This is a **machinery**-validation system (does the pipeline run end to end and
return a sane ΔG), not a benchmark against published affinities.

## Role mapping

| file | molecule | PDB source | role |
|------|----------|-----------|------|
| `receptor.prm7` / `receptor.rst7` | FKBP12 | chain A, residues 1–107 | receptor (glue-presenting protein) |
| `target.prm7` / `target.rst7` | FRB domain of mTOR | chain B, residues 2018–2112 | target |
| `glue.sdf` | rapamycin | ligand `RAP`, bound pose | glue (`assign_to: receptor`) |
| `waters.prm7` / `waters.rst7` | crystal waters | `HOH`, 23 waters | optional `waters` input |

Rapamycin binds FKBP12 tightly and the FKBP12·rapamycin surface then recruits
FRB, so the glue is assigned to the receptor.

The crystal waters are a **separate** input (`inputs.waters`), not embedded in the
protein topologies: gluebind appends them as the last complex component so they
never perturb the protein/glue atom blocks the restraint map anchors on. They
exercise that path and demonstrate `inputs.waters`.

Counts: receptor 1663 atoms / 107 CA; target 1583 atoms / 95 CA; glue
C₅₁H₇₉NO₁₃, 65 heavy atoms (144 with H), net-neutral; waters 23 TIP3P molecules
(69 atoms), kept at their crystallographic positions.

## Provenance / regeneration

Proteins parameterised with `ff19SB` (dry — gluebind assembles and solvates the
complex itself). tleap renumbers FRB to 1–95. The glue SDF keeps the crystal
**bound pose**; bond orders were assigned from the RCSB ideal-ligand template
(`RAP_ideal.sdf`) and hydrogens added — its perceived stereochemistry is the
crystal's, not the idealised CCD structure. The crystal waters are parameterised
as `TIP3P` at their crystallographic positions (1FAP already models the water
hydrogens). Rebuild with `build_inputs.sh` (needs AmberTools + RDKit + network
access to RCSB).
