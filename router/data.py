import json
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from data.format import VOCAB
from utils import const

try:
    from torch_geometric.data import Data, Batch
except Exception:  # pragma: no cover - optional dependency
    Data = None
    Batch = None

try:
    from Bio.PDB import PDBParser
    from Bio.PDB.Polypeptide import is_aa
except Exception:  # pragma: no cover - optional dependency
    PDBParser = None
    is_aa = None

from data.mmap_dataset import create_mmap, MMAPDataset


AA3 = {abrv for _, abrv in const.aas}


def _warn(msg: str) -> None:
    print(f"[router][warn] {msg}")


def _parse_jsonl(jsonl_path: str) -> List[Dict[str, Any]]:
    records = []
    with open(jsonl_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            records.append(json.loads(line))
    return records


def _residue_id_to_str(chain_id: str, residue) -> str:
    het, resseq, icode = residue.get_id()
    icode = icode.strip() if isinstance(icode, str) else ""
    return f"{chain_id}:{resseq}{icode}"


def _load_chain_atoms(
    pdb_path: str,
    chain_id: str,
    backbone_only: bool,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str]]]:
    if PDBParser is None or is_aa is None:
        raise RuntimeError("Biopython is required to parse PDB files.")
    if not os.path.exists(pdb_path):
        _warn(f"Missing PDB: {pdb_path}")
        return None

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("pocket", pdb_path)
    model = next(structure.get_models())
    if chain_id not in model:
        _warn(f"Chain {chain_id} not found in {pdb_path}")
        return None

    n_channel = 4 if backbone_only else VOCAB.MAX_ATOM_NUMBER
    chain = model[chain_id]
    residue_ids: List[str] = []
    X_list: List[List[List[float]]] = []
    S_list: List[int] = []
    atom_mask_list: List[List[int]] = []

    for residue in chain:
        if not is_aa(residue, standard=True):
            continue
        resname = residue.get_resname().upper()
        if resname not in AA3:
            continue
        symbol = VOCAB.abrv_to_symbol(resname)
        if symbol is None:
            continue
        atom2coord = {atom.get_name(): atom.get_coord().tolist() for atom in residue.get_atoms()}
        if not atom2coord:
            continue
        bb_pos = torch.tensor(list(atom2coord.values()), dtype=torch.float).mean(dim=0).tolist()

        atom_names = VOCAB.backbone_atoms
        if not backbone_only:
            atom_names = atom_names + const.sidechain_atoms[symbol]
        coords = []
        mask = []
        for atom_name in atom_names:
            if atom_name in atom2coord:
                coords.append(atom2coord[atom_name])
                mask.append(1)
            else:
                coords.append(bb_pos)
                mask.append(0)
        n_pad = n_channel - len(coords)
        for _ in range(n_pad):
            coords.append(bb_pos)
            mask.append(0)

        X_list.append(coords)
        S_list.append(VOCAB.symbol_to_idx(symbol))
        atom_mask_list.append(mask)
        residue_ids.append(_residue_id_to_str(chain_id, residue))

    if not S_list:
        _warn(f"No valid residues for chain {chain_id} in {pdb_path}")
        return None

    X = torch.tensor(X_list, dtype=torch.float)
    S = torch.tensor(S_list, dtype=torch.long)
    atom_mask = torch.tensor(atom_mask_list, dtype=torch.bool)
    position_ids = torch.arange(1, len(S_list) + 1, dtype=torch.long)
    return X, S, position_ids, atom_mask, residue_ids


class RouterDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str,
        pdb_dir: str,
        type_list: List[str],
        chain_id: str = "R",
        backbone_only: bool = False,
        use_mmap: bool = False,
        mmap_dir: Optional[str] = None,
        max_warn: int = 20,
    ) -> None:
        super().__init__()
        self.type_list = type_list
        self.type_to_idx = {t: i for i, t in enumerate(type_list)}
        self.pdb_dir = pdb_dir
        self.chain_id = chain_id
        self.backbone_only = backbone_only
        self.max_warn = max_warn
        self._warn_count = 0

        self._use_mmap = use_mmap
        self._mmap = None

        if use_mmap:
            if mmap_dir is None:
                raise ValueError("mmap_dir is required when use_mmap is True.")
            self._mmap = MMAPDataset(mmap_dir)
            self._records = []
        else:
            self._records = _parse_jsonl(jsonl_path)

    def __len__(self) -> int:
        if self._use_mmap:
            return len(self._mmap)
        return len(self._records)

    def get_sample_id(self, idx: int) -> str:
        if self._use_mmap:
            return self._mmap._indexes[idx][0]
        return self._records[idx].get("sample_id", "")

    def build_id_index(self) -> Dict[str, int]:
        mapping: Dict[str, int] = {}
        for idx in range(len(self)):
            sample_id = self.get_sample_id(idx)
            if sample_id and sample_id not in mapping:
                mapping[sample_id] = idx
        return mapping

    def get_c_star(self, idx: int) -> int:
        if self._use_mmap:
            try:
                props = self._mmap._properties[idx]
                if len(props) >= 2:
                    return int(props[1])
            except Exception:
                pass
            record = self._mmap[idx]
            return int(record.get("c_star", -1))
        record = self._records[idx]
        c_star = record.get("c_star")
        if isinstance(c_star, str):
            return int(self.type_to_idx.get(c_star, -1))
        if c_star is None:
            return -1
        return int(c_star)

    def _maybe_warn(self, msg: str) -> None:
        if self._warn_count >= self.max_warn:
            return
        self._warn_count += 1
        _warn(msg)

    def _build_graph(self, pdb_path: str) -> Optional[Data]:
        loaded = _load_chain_atoms(pdb_path, self.chain_id, self.backbone_only)
        if loaded is None:
            self._maybe_warn(f"Skip PDB: {pdb_path}")
            return None
        X, S, position_ids, atom_mask, residue_ids = loaded

        if Data is None:
            raise RuntimeError("torch_geometric is required for graph Data.")
        data = Data(X=X, S=S, position_ids=position_ids, atom_mask=atom_mask)
        data.num_nodes = X.size(0)
        data.residue_ids = residue_ids
        return data

    def _build_item_from_record(self, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        sample_id = record["sample_id"]
        pdb_path = os.path.join(self.pdb_dir, f"{sample_id}.pdb")
        graph = self._build_graph(pdb_path)
        if graph is None:
            return None

        c_star = record.get("c_star")
        if c_star not in self.type_to_idx:
            self._maybe_warn(f"Unknown type {c_star} for {sample_id}")
            return None

        item = {
            "graph": graph,
            "peptide_length": int(record["peptide_length"]),
            "c_star": self.type_to_idx[c_star],
            "type_soft": torch.tensor(record.get("type_soft", []), dtype=torch.float),
            "gt_pair": torch.tensor(record.get("gt_pair", [-1, -1]), dtype=torch.long),
            "pair_soft": torch.tensor(record.get("pair_soft", []), dtype=torch.float),
            "sample_id": sample_id,
        }
        return item

    def _build_item_from_mmap(self, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        graph = record["graph"]
        X = torch.tensor(graph["X"], dtype=torch.float)
        S = torch.tensor(graph["S"], dtype=torch.long)
        position_ids = torch.tensor(graph["position_ids"], dtype=torch.long)
        atom_mask = torch.tensor(graph["atom_mask"], dtype=torch.bool)
        if Data is None:
            raise RuntimeError("torch_geometric is required for graph Data.")
        data = Data(X=X, S=S, position_ids=position_ids, atom_mask=atom_mask)
        data.num_nodes = X.size(0)
        data.residue_ids = graph.get("residue_ids", [])

        item = {
            "graph": data,
            "peptide_length": int(record["peptide_length"]),
            "c_star": int(record["c_star"]),
            "type_soft": torch.tensor(record.get("type_soft", []), dtype=torch.float),
            "gt_pair": torch.tensor(record.get("gt_pair", [-1, -1]), dtype=torch.long),
            "pair_soft": torch.tensor(record.get("pair_soft", []), dtype=torch.float),
            "sample_id": record.get("sample_id", ""),
        }
        return item

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        if self._use_mmap:
            record = self._mmap[idx]
            return self._build_item_from_mmap(record)
        record = self._records[idx]
        return self._build_item_from_record(record)


def router_collate(items: List[Optional[Dict[str, Any]]]):
    items = [item for item in items if item is not None]
    if not items:
        return None
    if Batch is None:
        raise RuntimeError("torch_geometric is required for batching.")
    graphs = [item["graph"] for item in items]
    batch_graph = Batch.from_data_list(graphs)
    batch = {
        "peptide_length": [item["peptide_length"] for item in items],
        "c_star": torch.tensor([item["c_star"] for item in items], dtype=torch.long),
        "type_soft": [item["type_soft"] for item in items],
        "gt_pair": [item["gt_pair"] for item in items],
        "pair_soft": [item["pair_soft"] for item in items],
        "sample_id": [item["sample_id"] for item in items],
    }
    return batch_graph, batch


def _mmap_iterator(
    records: Iterable[Dict[str, Any]],
    pdb_dir: str,
    type_list: List[str],
    chain_id: str,
    backbone_only: bool,
    progress: Optional[Any] = None,
) -> Iterable[Tuple[str, Dict[str, Any], List[Any], int]]:
    type_to_idx = {t: i for i, t in enumerate(type_list)}
    for idx, record in enumerate(records):
        if progress is not None:
            progress.update(1)
        sample_id = record["sample_id"]
        pdb_path = os.path.join(pdb_dir, f"{sample_id}.pdb")
        loaded = _load_chain_atoms(pdb_path, chain_id, backbone_only)
        if loaded is None:
            continue
        X, S, position_ids, atom_mask, residue_ids = loaded
        graph = {
            "X": X.tolist(),
            "S": S.tolist(),
            "position_ids": position_ids.tolist(),
            "atom_mask": atom_mask.tolist(),
            "residue_ids": residue_ids,
        }
        c_star = record.get("c_star")
        if c_star not in type_to_idx:
            continue
        item = {
            "graph": graph,
            "peptide_length": int(record["peptide_length"]),
            "c_star": type_to_idx[c_star],
            "type_soft": record.get("type_soft", []),
            "gt_pair": record.get("gt_pair", [-1, -1]),
            "pair_soft": record.get("pair_soft", []),
            "sample_id": sample_id,
        }
        properties = [item["peptide_length"], item["c_star"]]
        yield sample_id, item, properties, idx


def build_router_mmap(
    jsonl_path: str,
    pdb_dir: str,
    out_dir: str,
    type_list: List[str],
    chain_id: str = "R",
    backbone_only: bool = False,
) -> None:
    records = _parse_jsonl(jsonl_path)
    progress = None
    try:
        from tqdm import tqdm  # type: ignore

        progress = tqdm(total=len(records), desc="build_router_mmap", unit="sample")
    except Exception:
        progress = None
    try:
        iterator = _mmap_iterator(
            records,
            pdb_dir,
            type_list,
            chain_id,
            backbone_only,
            progress=progress,
        )
        create_mmap(iterator, out_dir, total_len=len(records))
    finally:
        if progress is not None:
            progress.close()
