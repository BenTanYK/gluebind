#!/usr/bin/env bash
# Regenerate the 1FAP test fixture (FKBP12 receptor, FRB target, rapamycin glue)
# from the RCSB deposition. Requires AmberTools (pdb4amber, tleap) and RDKit on
# PATH, plus network access to files.rcsb.org.
#
#   ./build_inputs.sh [output_dir]   (default: current directory)
set -euo pipefail

OUT="${1:-.}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
cd "$WORK"

curl -sS --fail -o 1FAP.pdb https://files.rcsb.org/download/1FAP.pdb
curl -sS --fail -o RAP_ideal.sdf https://files.rcsb.org/ligands/download/RAP_ideal.sdf

# --- split components (fixed-column parse: chain @22, resname @18) -----------
awk 'substr($0,1,4)=="ATOM"   && substr($0,22,1)=="A"'   1FAP.pdb > fkbp12_raw.pdb
awk 'substr($0,1,4)=="ATOM"   && substr($0,22,1)=="B"'   1FAP.pdb > frb_raw.pdb
awk 'substr($0,1,6)=="HETATM" && substr($0,18,3)=="RAP"' 1FAP.pdb > rap_raw.pdb
awk 'substr($0,1,6)=="HETATM" && substr($0,18,3)=="HOH"' 1FAP.pdb > waters_raw.pdb

# --- proteins: dry ff19SB prm7/rst7 (gluebind solvates the complex itself) ---
for name in fkbp12:receptor frb:target; do
    src="${name%%:*}"; role="${name##*:}"
    pdb4amber -i "${src}_raw.pdb" -o "${src}.pdb" --nohyd --dry >/dev/null 2>&1
    tleap -f - <<EOF >/dev/null 2>&1
source leaprc.protein.ff19SB
m = loadpdb ${src}.pdb
saveamberparm m ${role}.prm7 ${role}.rst7
quit
EOF
done

# --- glue: rapamycin in the crystal bound pose, bond orders from CCD ideal ---
python - <<'PY'
from rdkit import Chem
from rdkit.Chem import AllChem
crystal = Chem.MolFromPDBFile("rap_raw.pdb", removeHs=True, sanitize=False)
template = Chem.RemoveHs(Chem.MolFromMolFile("RAP_ideal.sdf"))
mol = AllChem.AssignBondOrdersFromTemplate(template, crystal)
mol = Chem.AddHs(mol, addCoords=True)
mol.SetProp("_Name", "MOL")
Chem.MolToMolFile(mol, "glue.sdf")
PY

# --- crystal waters: TIP3P at crystallographic positions (separate input) ----
pdb4amber -i waters_raw.pdb -o waters.pdb >/dev/null 2>&1
tleap -f - <<'EOF' >/dev/null 2>&1
source leaprc.water.tip3p
m = loadpdb waters.pdb
saveamberparm m waters.prm7 waters.rst7
quit
EOF

mkdir -p "$OUT"
cp receptor.prm7 receptor.rst7 target.prm7 target.rst7 glue.sdf \
   waters.prm7 waters.rst7 "$OUT"/
echo "wrote fixture to $OUT"
