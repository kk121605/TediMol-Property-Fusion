"""Independent RDKit evaluation; not a differentiable training loss."""
import math
import selfies as sf
from rdkit import Chem
from rdkit.Chem import Crippen, rdMolDescriptors
from rdkit.Contrib.SA_Score import sascorer

FUNCTIONS = {"logP": Crippen.MolLogP, "SA": sascorer.calculateScore,
             "TPSA": rdMolDescriptors.CalcTPSA}

def compute_generated_properties(value, names):
    try:
        smiles = sf.decoder(value)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumAtoms() == 0:
            return smiles, False, {}
        props = {name: float(FUNCTIONS[name](mol)) for name in names}
        if not all(math.isfinite(v) for v in props.values()):
            return smiles, False, {}
        return Chem.MolToSmiles(mol, isomericSmiles=True), True, props
    except (ValueError, TypeError, RuntimeError, OverflowError, ZeroDivisionError):
        return "", False, {}
