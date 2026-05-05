from dataclasses import dataclass
from typing import List, Optional

import torch
from torch_geometric.data import Data
from rdkit import Chem

@dataclass
class GraphBuilder:
    """Convert SMILES strings into PyTorch Geometric Data objects."""

    def smiles_to_graph(self, smiles: str) -> Optional[Data]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Node features: atomic number, degree, formal charge, aromatic, in ring
        x_list: List[List[float]] = []
        for atom in mol.GetAtoms():
            x_list.append([
                atom.GetAtomicNum(),
                atom.GetDegree(),
                atom.GetFormalCharge(),
                float(atom.GetIsAromatic()),
                float(atom.IsInRing()),
            ])

        # Edge index and edge features: bond type, conjugation, in ring
        edge_index_list: List[List[int]] = []
        edge_attr_list: List[List[float]] = []

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            bt = bond.GetBondType()
            if bt == Chem.rdchem.BondType.SINGLE:
                bond_type = 1.0
            elif bt == Chem.rdchem.BondType.DOUBLE:
                bond_type = 2.0
            elif bt == Chem.rdchem.BondType.TRIPLE:
                bond_type = 3.0
            elif bt == Chem.rdchem.BondType.AROMATIC:
                bond_type = 1.5
            else:
                bond_type = 0.0

            feat = [
                bond_type,
                float(bond.GetIsConjugated()),
                float(bond.IsInRing()),
            ]

            # undirected graph: add both directions
            edge_index_list.append([i, j])
            edge_index_list.append([j, i])
            edge_attr_list.append(feat)
            edge_attr_list.append(feat)

        x = torch.tensor(x_list, dtype=torch.float)
        if edge_index_list:
            edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)
        else:
            # Handle molecules with no explicit bonds (shouldn't happen here, but be safe)
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 3), dtype=torch.float)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

if __name__ == "__main__":
    gb = GraphBuilder()
    g = gb.smiles_to_graph("c1ccccc1")
    print(g)