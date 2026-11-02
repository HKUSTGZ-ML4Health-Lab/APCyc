"""External CPSea/OpenMM adapter: preserve identities and use explicit predicted sites."""
import argparse
import io
import json
from pathlib import Path
import sys


def run(args):
    sys.path.insert(0, str(args.cpsea_relax_dir.resolve()))
    from base_model import ForceFieldMinimizer
    import pdbfixer
    from openmm import app

    request = json.loads(args.request.read_text())
    spec, chain_id = request['closure'], request['chain']

    class PredictedTopologyMinimizer(ForceFieldMinimizer):
        def _fix_cyclic(self, fixer, cyclic_chains, cyclic_opts):
            modeller = app.Modeller(fixer.topology, fixer.positions)
            chain = next(c for c in modeller.topology.chains() if c.id == chain_id)
            residues = list(chain.residues())
            i, j = spec['indices']
            removals = []
            if spec['type'] == 'HT':
                delete = {i: {'OXT'}, j: {'H2', 'H3'}}
            elif spec['type'] == 'SS':
                if any(residues[k].name != 'CYS' for k in [i, j]):
                    raise ValueError('Disulfide adapter refuses residue mutation')
                delete = {i: {'HG'}, j: {'HG'}}
            else:
                if residues[i].name != 'LYS' or residues[j].name not in {'ASP', 'ASN'}:
                    raise ValueError('Isopeptide adapter requires Lys-Asp/Asn')
                leaving = {'OD2'} if residues[j].name == 'ASP' else {'ND2', 'HD21', 'HD22'}
                delete = {i: {'HZ2', 'HZ3'}, j: leaving}
            for index, names in delete.items():
                removals.extend(a for a in residues[index].atoms() if a.name in names)
            modeller.delete(removals)
            text = io.StringIO()
            app.PDBFile.writeFile(modeller.topology, modeller.positions, text, keepIds=True)
            fixed = pdbfixer.PDBFixer(pdbfile=io.StringIO(text.getvalue()))
            residues = list(next(c for c in fixed.topology.chains() if c.id == chain_id).residues())
            atoms = [next(a for a in residues[k].atoms() if a.name == name)
                     for k, name in zip(spec['indices'], spec['atoms'])]
            connections = [f'CONECT{int(atoms[0].id):5d}{int(atoms[1].id):5d}',
                           f'CONECT{int(atoms[1].id):5d}{int(atoms[0].id):5d}']
            return self._add_connects(text.getvalue(), connections), connections

    output = Path(request['output'])
    runner = PredictedTopologyMinimizer(platform='CPU', ligand_chain=chain_id)
    runner(request['source'], str(output), cyclic_chains=[chain_id])
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError('CPSea relaxation failed to produce a structure; inspect the per-sample log.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--request', required=True, type=Path)
    p.add_argument('--cpsea-relax-dir', required=True, type=Path)
    run(p.parse_args())
