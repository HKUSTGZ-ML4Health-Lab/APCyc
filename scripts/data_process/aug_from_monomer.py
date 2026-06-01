import os
import re
import gzip
import time
import shutil
import argparse
from copy import deepcopy
from tempfile import NamedTemporaryFile
import multiprocessing as mp

import numpy as np
from Bio.PDB import PDBParser,Chain,Model,Structure, PDBIO
from Bio.PDB.DSSP import dssp_dict_from_pdb_file
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from rdkit.Chem.rdMolDescriptors import CalcTPSA
from freesasa import calcBioPDB
from rdkit.Chem import MolFromSmiles

from globals import CACHE_DIR, CONTACT_DIST
from utils.logger import print_log
from utils.file_utils import cnt_num_files, get_filename
from data.mmap_dataset import create_mmap
from data.converter.pdb_to_list_blocks import pdb_to_list_blocks
from data.converter.blocks_interface import blocks_cb_interface, blocks_interface

from .pepbind import clustering


def parse():
    parser = argparse.ArgumentParser(description='Filter peptide-like loop from monomers')
    parser.add_argument('--database_dir', type=str, required=True,
                        help='Directory of pdb database processed in monomers')
    parser.add_argument('--pdb_dir', type=str, required=True, help='Directory to PDB database')
    parser.add_argument('--out_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--pocket_th', type=float, default=10.0, help='Threshold for determining pocket')
    parser.add_argument('--n_cpu', type=int, default=4, help='Number of CPU to use')

    return parser.parse_args()


# Constants
AA3TO1 = {
    'ALA':'A', 'VAL':'V', 'PHE':'F', 'PRO':'P', 'MET':'M',
    'ILE':'I', 'LEU':'L', 'ASP':'D', 'GLU':'E', 'LYS':'K',
    'ARG':'R', 'SER':'S', 'THR':'T', 'TYR':'Y', 'HIS':'H',
    'CYS':'C', 'ASN':'N', 'GLN':'Q', 'TRP':'W', 'GLY':'G',}
    
hydrophobic_residues=['V','I','L','M','F','W','C']
charged_residues=['H','R','K','D','E']

def add_cb(input_array):
    #from protein mpnn
    #The virtual Cβ coordinates were calculated using ideal angle and bond length definitions: b = Cα - N, c = C - Cα, a = cross(b, c), Cβ = -0.58273431*a + 0.56802827*b - 0.54067466*c + Cα.
    N,CA,C,O = input_array
    b = CA - N
    c = C - CA
    a = np.cross(b,c)
    CB = np.around(-0.58273431*a + 0.56802827*b - 0.54067466*c + CA,3)
    return CB #np.array([N,CA,C,CB,O])

aaSMILES = {'G':  'NCC(=O)O',
            'A':  'N[C@@]([H])(C)C(=O)O',
            'R':  'N[C@@]([H])(CCCNC(=N)N)C(=O)O',
            'N':  'N[C@@]([H])(CC(=O)N)C(=O)O',
            'D':  'N[C@@]([H])(CC(=O)O)C(=O)O',
            'C':  'N[C@@]([H])(CS)C(=O)O',
            'E':  'N[C@@]([H])(CCC(=O)O)C(=O)O',
            'Q':  'N[C@@]([H])(CCC(=O)N)C(=O)O',
            'H':  'N[C@@]([H])(CC1=CN=C-N1)C(=O)O',
            'I':  'N[C@@]([H])(C(CC)C)C(=O)O',
            'L':  'N[C@@]([H])(CC(C)C)C(=O)O',
            'K':  'N[C@@]([H])(CCCCN)C(=O)O',
            'M':  'N[C@@]([H])(CCSC)C(=O)O',
            'F':  'N[C@@]([H])(Cc1ccccc1)C(=O)O',
            'P':  'N1[C@@]([H])(CCC1)C(=O)O',
            'S':  'N[C@@]([H])(CO)C(=O)O',
            'T':  'N[C@@]([H])(C(O)C)C(=O)O',
            'W':  'N[C@@]([H])(CC(=CN2)C1=C2C=CC=C1)C(=O)O',
            'Y':  'N[C@@]([H])(Cc1ccc(O)cc1)C(=O)O',
            'V':  'N[C@@]([H])(C(C)C)C(=O)O'}


class Filter:
    def __init__(
            self,
            min_loop_len = 4,
            max_loop_len = 25,
            min_BSA = 400,
            min_relBSA = 0.2,
            max_relncBSA = 0.3,
            saved_maxlen = 25,
            saved_BSA = 400,
            saved_relBSA = 0.2,
            saved_helix_ratio = 1.0,
            saved_strand_ratio = 1.0,
            cyclic=False
        ) -> None:

        self.re_filter = re.compile(r'D[GPS]|[P]{2,}|C')  #https://www.thermofisher.cn/cn/zh/home/life-science/protein-biology/protein-biology-learning-center/protein-biology-resource-library/pierce-protein-methods/peptide-design.html
        self.cache_dir = CACHE_DIR

        self.min_loop_len = min_loop_len
        self.max_loop_len = max_loop_len
        self.min_BSA = min_BSA
        self.min_relBSA = min_relBSA
        self.max_relncBSA = max_relncBSA
        self.saved_maxlen = saved_maxlen
        self.saved_BSA = saved_BSA
        self.saved_relBSA = saved_relBSA
        self.saved_helix_ratio = saved_helix_ratio
        self.saved_strand_ratio = saved_strand_ratio
        self.cyclic = cyclic

    @classmethod
    def get_ss_info(cls, pdb_path: str):
        dssp, keys = dssp_dict_from_pdb_file(pdb_path, DSSP='mkdssp')
        ss_info = {}
        for key in keys:
            chain_id, value = key[0], dssp[key]
            if chain_id not in ss_info:
                ss_info[chain_id] = []
            ss_type = value[1]
            if ss_type in ['H', 'G', 'I']:
                ss_info[chain_id].append('a')
            elif ss_type in ['B', 'E', 'T', 'S']:
                ss_info[chain_id].append('b')
            elif ss_type == '-':
                ss_info[chain_id].append('c')
            else:
                raise ValueError(f'SS type {ss_type} cannot be recognized!')
        return ss_info
    
    @classmethod
    def get_bsa(self, receptor_chain: Chain.Chain, ligand_chain: Chain.Chain):
        lig_chain_id = ligand_chain.get_id()
        tmp_structure = Structure.Structure('tmp')
        tmp_model = Model.Model(0)
        tmp_structure.add(tmp_model) 
        tmp_model.add(ligand_chain)
        unbounded_SASA = calcBioPDB(tmp_structure)[0].residueAreas()[lig_chain_id]
        unbounded_SASA = [k.total for k in unbounded_SASA.values()]

        tmp_model.add(receptor_chain)
        bounded_SASA = calcBioPDB(tmp_structure)[0].residueAreas()[lig_chain_id]
        bounded_SASA = [k.total for k in bounded_SASA.values()]

        abs_bsa = sum(unbounded_SASA[1:-1]) - sum(bounded_SASA[1:-1])
        rel_bsa = abs_bsa / sum(unbounded_SASA[1:-1])
        rel_nc_bsa = (unbounded_SASA[0] + unbounded_SASA[-1] - bounded_SASA[0] - bounded_SASA[-1]) / (unbounded_SASA[0] + unbounded_SASA[-1])

        return abs_bsa, rel_bsa, rel_nc_bsa, tmp_structure
    
    def filter_pdb(self, pdb_path, selected_chains=None):
        parser = PDBParser(QUIET=True)
        ss_info = self.get_ss_info(pdb_path)
        structure = parser.get_structure('anonym', pdb_path)
        
        for model in structure.get_models():  # use model 1 only
            structure = model
            break

        results = []
        for chain in structure.get_chains():
            if selected_chains is not None and chain.get_id() not in selected_chains:
                continue
            chain_ss_info = None if ss_info is None else ss_info[chain.get_id()]
            results.extend(self.filter_chain(chain, chain_ss_info))
        
        return results


    def filter_chain(self, chain, ss_info=None):

        non_standard = False
        for res in chain:
            if res.get_resname() not in AA3TO1:
                non_standard = True
                break
        if non_standard:
            return []
        
        if len(ss_info) != len(chain):
            return []

        cb_coord = []
        seq = ''
        for res in chain:
            seq += AA3TO1[res.get_resname()]
            try:
                cb_coord.append(res['CB'].get_coord())
            except:
                tmp_coord = np.array([
                    res['N'].get_coord(),
                    res['CA'].get_coord(),
                    res['C'].get_coord(),
                    res['O'].get_coord()
                ])
                cb_coord.append(add_cb(tmp_coord))
        cb_coord = np.array(cb_coord)
        cb_contact = np.linalg.norm(cb_coord[None,:,:,] - cb_coord[:,None,:],axis=-1)
        if self.cyclic:
            possible_ss = (cb_contact >= 3.5) & (cb_contact <= 5) #https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4987930/
        else:
            possible_ss = np.ones(cb_contact.shape, dtype=bool)
        possible_ss = np.triu(np.tril(possible_ss, self.max_loop_len - 1), self.min_loop_len - 1)
        ss_pair = np.where(possible_ss)
        accepted, saved_spans = [], []
        for i, j in zip(ss_pair[0],ss_pair[1]):
            redundant = False
            for exist_i, exist_j in saved_spans:
                overlap = min(j, exist_j) - max(i, exist_i) + 1
                if overlap / (j - i + 1) > 0.4 or overlap / (exist_j - exist_i + 1) > 0.4:
                    redundant = True
                    break
            if redundant:
                continue
            #20A neighbor
            min_dist = np.min(cb_contact[i : j + 1], axis=0)
            min_dist[max(i - 5, 0):min(j + 6, len(seq))] = 21
            neighbors_20A = np.where(min_dist < 20)[0]
            if len(neighbors_20A) < 16:
                continue

            #sequence filter
            pep_seq = seq[i:j+1]
            #cystine 2P and DGDA filter
            if self.re_filter.search(pep_seq) is not None:
                continue
            prot_param=ProteinAnalysis(pep_seq)
            aa_percent = prot_param.get_amino_acids_percent()
            max_ratio = max(aa_percent.values())
            #Discard if any amino acid represents more than 25% of the total sequence
            if max_ratio > 0.25:
                continue
            hydrophobic_ratio = sum([aa_percent[k] for k in hydrophobic_residues])
            #hydrophobic amino acids exceeds 45%
            if hydrophobic_ratio > 0.45:
                continue
            #charged  amino acids exceeds 45% or less than 25%
            charged_ratio = sum([aa_percent[k] for k in charged_residues])
            if charged_ratio > 0.45 or charged_ratio < 0.25:
                continue
            #instablility index>40
            if prot_param.instability_index() >= 40:
                continue

            # #TPSA filter (for cell penetration)
            # mol_weight = prot_param.molecular_weight()
            # pepsmile='O'
            # for k in pep_seq:
            #     pepsmile=pepsmile[:-1] + aaSMILES[k]         
            # pepsmile = MolFromSmiles(pepsmile)
            # tpsa = CalcTPSA(pepsmile)
            # if tpsa <= mol_weight * 0.2:
            #     continue

            #build structure and get BSA
            receptor_chain = Chain.Chain('R') 
            ligand_chain = Chain.Chain('L') 
            for k,res in enumerate(chain):
                if k >= i and k <= j:
                    ligand_chain.add(res.copy())
                elif k in neighbors_20A:
                    receptor_chain.add(res.copy())
            
            abs_bsa, rel_bsa, rel_nc_bsa, tmp_structure = self.get_bsa(receptor_chain, ligand_chain)
            if abs_bsa <= self.min_BSA or rel_bsa <= self.min_relBSA or (self.cyclic and rel_nc_bsa >= self.max_relncBSA):
                continue
            
            #prepare for output
            length = j - i + 1
            if ss_info is None:
                helix_ratio = -1
                strand_ratio = -1
                coil_ratio = -1
            else:
                ssa = ss_info[i:j+1]
                helix_ratio = ssa.count('a') / length
                strand_ratio = ssa.count('b') / length
                coil_ratio = ssa.count('c') / length
                # helix_ratio = (ssa.count("G") + ssa.count("H") + ssa.count("I") + ssa.count("T")) / length
                # strand_ratio = (ssa.count("E") + ssa.count("B")) / length
                # coil_ratio = (ssa.count("S")+ssa.count("C")) / length
            if length <= self.saved_maxlen and abs_bsa >= self.saved_BSA and rel_bsa >= self.saved_relBSA and helix_ratio <= self.saved_helix_ratio and strand_ratio <= self.saved_strand_ratio:
                output_structure = deepcopy(tmp_structure)
            else:
                output_structure = None
            accepted.append((
                i , j, length, abs_bsa, rel_bsa, helix_ratio, strand_ratio, coil_ratio, output_structure
            ))
            saved_spans.append((i, j))

        return accepted
    

def get_non_redundant(mmap_dir):
    np.random.seed(12)
    index_path = os.path.join(mmap_dir, 'index.txt')
    parent_dir = mmap_dir

    # load index file
    items = {}
    with open(index_path, 'r') as fin:
        lines = fin.readlines()
        for line in lines:
            values = line.strip().split('\t')
            _id, seq = values[0], values[-1]
            chain, pdb_file = _id.split('_')
            items[_id] = (seq, chain, pdb_file)

    # make temporary directory
    tmp_dir = os.path.join(parent_dir, 'tmp')
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    else:
        raise ValueError(f'Working directory {tmp_dir} exists!')
    
    # 1. get non-redundant dimer by 90% seq-id
    fasta = os.path.join(tmp_dir, 'seq.fasta')
    with open(fasta, 'w') as fout:
        for _id in items:
            fout.write(f'>{_id}\n{items[_id][0]}\n')
    id2clu, clu2id = clustering(fasta, tmp_dir, 0.9)
    non_redundant = []
    for clu in clu2id:
        ids = clu2id[clu]
        non_redundant.append(np.random.choice(ids))
    print_log(f'Non-redundant entries: {len(non_redundant)}')
    shutil.rmtree(tmp_dir)

    # 2. construct non_redundant items
    indexes = {}
    for _id in non_redundant:
        _, chain, pdb_file = items[_id]
        if pdb_file not in indexes:
            indexes[pdb_file] = []
        indexes[pdb_file].append(chain)

    return indexes


def mp_worker(data_dir, tmp_dir, pdb_file, selected_chains, pep_filter, pdb_out_dir, queue):
    category = pdb_file[4:6]
    category_dir = os.path.join(data_dir, category)
    path = os.path.join(category_dir, pdb_file)
    tmp_file = os.path.join(tmp_dir, f'{pdb_file}.decompressed')
    pdb_id = get_filename(pdb_file.split('.')[0])

    # uncompress the file to the tmp file
    with gzip.open(path, 'rb') as fin:
        with open(tmp_file, 'wb') as fout:
            shutil.copyfileobj(fin, fout)

    files = []
    try:
        # biotitie_pdb_file = biotite_pdb.PDBFile.read(tmp_file)
        # biotite_struct = biotitie_pdb_file.get_structure(model=1)
        # ss_info = { chain: annotate_sse(biotite_struct, chain_id=chain) for chain in selected_chains } 
        results = pep_filter.filter_pdb(tmp_file, selected_chains=selected_chains)
        for item in results:
            i, j, struct = item[0], item[1], item[-1]
            if struct is None:
                continue
            io = PDBIO()
            io.set_structure(struct)
            _id = pdb_id + f'_{i}_{j}'
            save_path = os.path.join(pdb_out_dir, _id + '.pdb')
            io.save(save_path)
            files.append(save_path)
    except Exception:  # pdbs with missing backbone coordinates or DSSP failed
        pass
    queue.put((pdb_file, files))
    os.remove(tmp_file)
    

def process_iterator(indexes, data_dir, tmp_dir, out_dir, pocket_th, n_cpu):
    pdb_out_dir = os.path.join(out_dir, 'pdbs')
    if not os.path.exists(pdb_out_dir):
        os.makedirs(pdb_out_dir)
    pep_filter = Filter()

    file_cnt, pointer, filenames = 0, 0, list(indexes.keys())
    id2task = {}
    queue = mp.Queue()
    # initialize tasks
    for _ in range(n_cpu):
        task_id = filenames[pointer]
        id2task[task_id] = mp.Process(
            target=mp_worker,
            args=(data_dir, tmp_dir, task_id, indexes[task_id], pep_filter, pdb_out_dir, queue)
        )
        id2task[task_id].start()
        pointer += 1

    while True:
        if len(id2task) == 0:
            break

        if not queue.qsize:  # no finished ones
            time.sleep(1)
            continue

        pdb_file, paths = queue.get()
        file_cnt += 1
        id2task[pdb_file].join()
        del id2task[pdb_file]

        # add the next task
        if pointer < len(filenames):
            task_id = filenames[pointer]
            id2task[task_id] = mp.Process(
                target=mp_worker,
                args=(data_dir, tmp_dir, task_id, indexes[task_id], pep_filter, pdb_out_dir, queue)
            )
            id2task[task_id].start()
            pointer += 1

        # handle processed data
        for save_path in paths:
            _id = get_filename(save_path)

            list_blocks, chains = pdb_to_list_blocks(save_path, return_chain_ids=True)
            if chains[0] == 'L':
                list_blocks, chains = (list_blocks[1], list_blocks[0]), (chains[1], chains[0])

            rec_blocks, lig_blocks = list_blocks
            rec_chain, lig_chain = chains
            try:
                _, (pocket_idx, _) = blocks_cb_interface(rec_blocks, lig_blocks, pocket_th)
            except KeyError:
                continue
            rec_num_units = sum([len(block) for block in rec_blocks])
            lig_num_units = sum([len(block) for block in lig_blocks])
            rec_data = [block.to_tuple() for block in rec_blocks]
            lig_data = [block.to_tuple() for block in lig_blocks]
            rec_seq = ''.join([AA3TO1[block.abrv] for block in rec_blocks])
            lig_seq = ''.join([AA3TO1[block.abrv] for block in lig_blocks])

            yield _id, (rec_data, lig_data), [
                len(rec_blocks), len(lig_blocks), rec_num_units, lig_num_units,
                rec_chain, lig_chain, rec_seq, lig_seq,
                ','.join([str(idx) for idx in pocket_idx]),
                ], file_cnt


def main(args):
    indexes = get_non_redundant(args.database_dir)
    cnt = len(indexes)
    tmp_dir = './tmp'
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    print_log(f'Processing data from directory: {args.pdb_dir}.')
    print_log(f'Number of entries: {cnt}')
    create_mmap(
        process_iterator(indexes, args.pdb_dir, tmp_dir, args.out_dir, args.pocket_th, args.n_cpu),
        args.out_dir, cnt)
    
    print_log('Finished!')

    shutil.rmtree(tmp_dir)


if __name__ == '__main__':
    main(parse())
