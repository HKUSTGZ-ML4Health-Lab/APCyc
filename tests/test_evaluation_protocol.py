"""Evaluation conventions, Rosetta provenance and energy-guided sampling."""
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import sys
import os
import unittest
from unittest.mock import patch

import numpy as np
import torch
from Bio.Align import PairwiseAligner, substitution_matrices
from apcyc.guidance import sigma2_from_transition
from pipelines.adapters.rosetta import protocol_metadata
from pipelines.evaluate_paper import evaluate, cramers_v
from pipelines.measure_paper import sequence_distance, ca_rmsd, measure
import test_paper_fixes as fixtures


class EvaluationProtocol(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def pdb(self, path, shift=0, peptide_shift=0, length=4):
        serial, lines = 1, []
        for chain, n in [('R', 3), ('L', length)]:
            for i in range(n):
                for atom, offset in [('N', -.5), ('CA', 0), ('C', .5), ('O', .7)]:
                    x = i * 2.0 + offset + shift + (peptide_shift if chain == 'L' else 0)
                    y = 5.0 if chain == 'L' else 0.0
                    lines.append(f'ATOM  {serial:5d} {atom:^4s} ALA {chain}{i+1:4d}    {x:8.3f}{y:8.3f}{0.:8.3f}{1.:6.2f}{20.:6.2f}          {atom[0]:>2s}\n')
                    serial += 1
            lines.append('TER\n')
        path.write_text(''.join(lines) + 'END\n')

    def fixture(self):
        rows = []
        for sid, shift in [('a', 0), ('b', 5)]:
            self.pdb(self.root / f'{sid}.pdb', peptide_shift=shift)
            rows.append(dict(id=sid, receptor_id='r', pdb=f'{sid}.pdb', sequence='AAAA', peptide_chain='L', pair_hat=[0, 3], pair_index_base=0))
        manifest = self.root / 'predictions.jsonl'
        manifest.write_text(''.join(json.dumps(x) + '\n' for x in rows))
        props = self.root / 'scores.csv'
        props.write_text('id,rosetta_dG,rosetta_total_score\na,-1,-500\nb,-2,-600\n')
        return manifest, props

    def test_blosum_distance_is_symmetric_and_matches_explicit_alignment(self):
        a, b = 'PKGYAAPSA', 'KPAVYKFTL'
        aligner = PairwiseAligner(mode='global')
        aligner.substitution_matrix = substitution_matrices.load('BLOSUM62')
        aligner.open_gap_score = aligner.extend_gap_score = 0
        expected = 1 - aligner.score(a, b) / np.sqrt(aligner.score(a, a) * aligner.score(b, b))
        self.assertAlmostEqual(sequence_distance(a, b), expected)
        self.assertEqual(sequence_distance(a, b), sequence_distance(b, a))
        self.assertEqual(sequence_distance(a, a), 0)
        self.assertGreater(sequence_distance('AAAA', 'WWWW'), .4)
        with self.assertRaises(ValueError):
            sequence_distance('AX', 'AA')

    def test_receptor_frame_rmsd_retains_translation_and_rejects_unequal_lengths(self):
        x = np.array([[0, 0, 0], [1, 2, 0], [2, 1, 3]])
        self.assertEqual(ca_rmsd(x, x + np.array([3, 4, 0])), 5.)
        with self.assertRaisesRegex(ValueError, 'equal'):
            ca_rmsd(x, x[:2])

    def test_manifest_measurement_id_join_distance_and_provenance(self):
        manifest, props = self.fixture()
        xml = Path(__file__).resolve().parents[1] / 'configs/evaluation/rosetta.xml'
        binary = self.root / 'binary'; binary.write_bytes(b'fixture')
        provenance = protocol_metadata(xml, [self.root / 'a.pdb', self.root / 'b.pdb'], binary)
        provenance['scores'] = {'a': {'rosetta_dG': -1, 'rosetta_total_score': -500},
                                'b': {'rosetta_dG': -2, 'rosetta_total_score': -600}}
        metadata = self.root / 'scores.protocol.json'; metadata.write_text(json.dumps(provenance))
        rows, conventions, pairwise, hashes = measure(manifest, props, metadata)
        self.assertEqual([r['cyclization_distance'] for r in rows], [6., 6.])
        self.assertEqual(pairwise['receptors']['r']['ca_rmsd'][0][1], 5.)
        result = evaluate(rows, conventions, pairwise, metrics=['rosetta_total_score'])
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['aggregate']['rosetta_total_score'], -550.)
        self.assertEqual(result['aggregate']['cramers_v'], 0.)
        self.assertEqual(hashes['a'], hashlib.sha256((self.root / 'a.pdb').read_bytes()).hexdigest())
        provenance['structures_sha256']['a'] = 'invalid'
        metadata.write_text(json.dumps(provenance))
        with self.assertRaisesRegex(ValueError, 'hashes'):
            measure(manifest, props, metadata)

    def test_measurement_rejects_wrong_frame_length_sequence_or_missing_candidates(self):
        manifest, props = self.fixture()
        self.pdb(self.root / 'b.pdb', shift=1)
        with self.assertRaisesRegex(ValueError, 'receptor coordinates'):
            measure(manifest, props)
        self.pdb(self.root / 'b.pdb', length=3)
        with self.assertRaisesRegex(ValueError, 'sequence'):
            measure(manifest, props)
        text = manifest.read_text().replace('"sequence": "AAAA"', '"sequence": null').replace('"pair_hat": [0, 3]', '"pair_hat": [0, 2]')
        manifest.write_text(text)
        with self.assertRaisesRegex(ValueError, 'one peptide length'):
            measure(manifest, props)
        props.write_text('id,rosetta_dG\na,-1\n')
        with self.assertRaisesRegex(ValueError, 'property row'):
            measure(manifest, props)

    def test_protocol_requires_explicit_output_score_function(self):
        xml = self.root / 'score.xml'
        xml.write_text('<ROSETTASCRIPTS><SCOREFXNS><ScoreFunction name="score" weights="ref2015"/></SCOREFXNS></ROSETTASCRIPTS>')
        with self.assertRaisesRegex(ValueError, 'OUTPUT'):
            protocol_metadata(xml, [], self.root / 'binary')

    def test_coverage_distinguishes_ineligible_from_missing_measurements(self):
        rows = [dict(id=sid, receptor_id=rid, cyclization_distance=4, rosetta_dG=energy,
                     vina_score=score) for sid, rid, energy, score in
                [('a', 'r1', -1, -2), ('b', 'r1', -1, -4), ('c', 'r2', 1, -900), ('d', 'r3', -1, -6)]]
        conv = dict(cyclization_distance='fixture', sequence_distance='fixture', ca_rmsd='fixture')
        pairwise = dict(format='apcyc.evaluation_distances.v1', receptors={
            rid: dict(ids=ids, sequence_distance=np.zeros((len(ids), len(ids))).tolist(),
                      ca_rmsd=np.zeros((len(ids), len(ids))).tolist())
            for rid, ids in [('r1', ['a', 'b']), ('r2', ['c']), ('r3', ['d'])]})
        result = evaluate(rows, conv, pairwise, metrics=['vina_score'])
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['aggregate']['vina_score'], -4.5)
        self.assertEqual(result['aggregate']['cramers_v'], 1.)
        self.assertEqual(result['coverage']['vina_score']['eligible_receptors'], 2)
        self.assertEqual(result['coverage']['D_co']['eligible_receptors'], 1)
        self.assertEqual(result['coverage']['energy_success']['eligible_receptors'], 3)
        rows[-1]['vina_score'] = ''
        incomplete = evaluate(rows, conv, pairwise, metrics=['vina_score'])
        self.assertIsNone(incomplete['aggregate']['vina_score'])
        self.assertEqual(incomplete['status'], 'incomplete')
        self.assertEqual(incomplete['coverage']['vina_score']['reported_receptors'], 1)

    def test_degenerate_consistency_and_all_invalid_population(self):
        self.assertEqual(cramers_v([0, 0], [1, 1]), 1.)
        self.assertEqual(cramers_v([0, 0], [1, 2]), 0.)
        self.assertEqual(cramers_v([0, 1], [2, 2]), 0.)
        self.assertIsNone(cramers_v([0], [0]))
        rows = [dict(id='a', receptor_id='r', cyclization_distance=20, rosetta_dG=1)]
        report = evaluate(rows, {'cyclization_distance': 'fixture'}, metrics=['vina_score'])
        self.assertEqual(report['status'], 'complete')
        self.assertEqual(report['aggregate']['valid_fraction'], 0)
        self.assertIsNone(report['aggregate']['vina_score'])
        self.assertEqual(report['coverage']['vina_score']['eligible_receptors'], 0)

    def test_guidance_descends_energy_and_keeps_receptor_fixed(self):
        case = fixtures.PaperFixes(); case.setUp()
        try:
            d, inputs = case.dpm(), case.fixture()
            args = dict(inputs); args['H'] = args.pop('H_0'); args['X'] = args.pop('X_0')
            class LinearProperty(torch.nn.Module):
                n_properties = 1
                def forward(self, **features):
                    mask = features['mask_generate']
                    score = features['H_noisy'][mask].sum() + features['X_noisy'][mask].sum()
                    return score.reshape(1, 1)
            torch.manual_seed(123)
            plain, _ = d.sample(**args, t_hard=1)
            torch.manual_seed(123)
            guided, _ = d.sample(**args, t_hard=1, use_cg_guidance=True, property_predictor=LinearProperty(),
                                 guidance_w=[1.], lambda_max=.2, t_guidance_start=3, t_guidance_end=3)
            mask = args['mask_generate']
            expected = .2 * float(d.trans_h.var_sched.sigmas[3] ** 2)
            delta = guided[2][1] - plain[2][1]
            self.assertTrue(torch.allclose(delta[mask], torch.full_like(delta[mask], expected), atol=1.1e-4, rtol=0))
            self.assertTrue(torch.equal(delta[~mask], torch.zeros_like(delta[~mask])))
            self.assertGreater(guided[2][1][mask].sum().item(), plain[2][1][mask].sum().item())
            variance = sigma2_from_transition(d.trans_h, 1, torch.zeros(7, dtype=torch.long), (7, 1))
            self.assertEqual(variance.abs().sum().item(), 0.)
        finally:
            case.tearDown()


    def test_measurement_rejects_explicit_invalid_ids_and_property_sequence_mismatch(self):
        manifest, props = self.fixture()
        records = [json.loads(line) for line in manifest.read_text().splitlines()]
        for field in ['id', 'receptor_id']:
            for invalid in [0, False, None, '']:
                changed = [dict(record) for record in records]
                changed[0][field] = invalid
                manifest.write_text(''.join(json.dumps(row) + '\n' for row in changed))
                with self.assertRaises(ValueError): measure(manifest, props)
        manifest.write_text(''.join(json.dumps(row) + '\n' for row in records))
        props.write_text('id,sequence,rosetta_dG\na,GGGG,-1\nb,AAAA,-2\n')
        with self.assertRaisesRegex(ValueError, 'Property CSV sequence'):
            measure(manifest, props)
        props.write_text(props.read_text().replace('GGGG', 'AAAA'))
        self.assertEqual(len(measure(manifest, props)[0]), 2)

    def test_evaluation_commands_preserve_aliased_input_files(self):
        from pipelines.measure_paper import main as measure_main
        from pipelines.evaluate_paper import main as evaluate_main
        manifest, props = self.fixture()
        destination = self.root/'measured'; destination.mkdir()
        input_copy = destination/'measurements.csv'; input_copy.write_bytes(props.read_bytes())
        before = input_copy.read_bytes()
        with patch.object(sys, 'argv', ['measure', '--manifest', str(manifest), '--properties', str(input_copy),
                                       '--out-dir', str(destination)]), self.assertRaisesRegex(ValueError, 'overwrite'):
            measure_main()
        self.assertEqual(input_copy.read_bytes(), before)
        with patch.object(sys, 'argv', ['measure', '--manifest', str(manifest), '--properties', str(props),
                                       '--out-dir', str(destination)]): measure_main()
        conventions = destination/'conventions.json'
        for target in [conventions, input_copy]:
            original = target.read_bytes()
            with patch.object(sys, 'argv', ['evaluate', '--measurements', str(input_copy), '--conventions', str(conventions),
                                           '--metrics', 'rosetta_dG', '--output', str(target)]), self.assertRaisesRegex(ValueError, 'overwrite'):
                evaluate_main()
            self.assertEqual(target.read_bytes(), original)
        hardlink = self.root/'alias.json'; os.link(conventions, hardlink)
        with patch.object(sys, 'argv', ['evaluate', '--measurements', str(input_copy), '--conventions', str(conventions),
                                       '--metrics', 'rosetta_dG', '--output', str(hardlink)]), self.assertRaises(ValueError):
            evaluate_main()
        output = self.root/'evaluation.json'
        with patch.object(sys, 'argv', ['evaluate', '--measurements', str(input_copy), '--conventions', str(conventions),
                                       '--metrics', 'rosetta_dG', '--output', str(output)]):
            self.assertEqual(evaluate_main(), 0)
        self.assertEqual(json.loads(output.read_text())['aggregate']['rosetta_dG'], -1.5)

if __name__ == '__main__':
    unittest.main()
