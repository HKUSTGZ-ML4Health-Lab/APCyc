"""Public command, configuration, cache and target-schema contracts."""
import copy
import csv
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd
import torch

import test_regressions as fixtures
from apcyc.guidance import load_property_predictors, parse_guidance_w, weights_from_map
import pipelines.collect_properties as collector
from pipelines.collect_properties import collect, merge_stage
from pipelines.evaluate_paper import evaluate
from pipelines.preflight_properties import inspect
from pipelines.score_properties import fit_normalizer, score


class DeliveryContracts(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.ReleaseRegressions(); self.fixture.setUp()
        self.root = self.fixture.root

    def tearDown(self):
        self.fixture.tearDown()

    def inputs(self, spec):
        pdb = self.fixture.structure()
        manifest = self.root / 'manifest.jsonl'
        manifest.write_text(json.dumps(dict(id='a', pdb=str(pdb), peptide_chain='L')) + '\n')
        config = self.root / 'config.json'
        config.write_text(json.dumps(dict(stages={'probe': spec})))
        return SimpleNamespace(config=config, manifest=manifest, out_dir=self.root/'output', only=None, run_external=True, strict=True)

    def test_preflight_import_ignores_external_environment_and_checks_headers(self):
        source = self.root/'source.csv';source.write_text('id,score\na,1\n')
        args = self.inputs(dict(**{'import': 'source.csv'}, required_env=['APCYC_NOT_CONFIGURED'], columns={'value':'score'}))
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(inspect(args.config)['ready'])
        collect(args)
        source.write_text('id,score,score\na,1,2\n')
        self.assertFalse(inspect(args.config)['ready'])

    def test_disabled_stages_and_path_executables(self):
        args = self.inputs(dict(enabled=False, required_env=['APCYC_NOT_CONFIGURED'], columns={'value':'score'}))
        self.assertTrue(inspect(args.config)['ready'])
        collect(args)
        status=json.loads((args.out_dir/'status.json').read_text())
        self.assertEqual(status['stages']['probe']['status'],'skipped')
        args.config.write_text(json.dumps(dict(stages={'probe':dict(enabled=True,argv=[Path(sys.executable).name],columns={'value':'score'})})))
        with patch.dict(os.environ, {'PATH':str(Path(sys.executable).parent)}):
            self.assertTrue(inspect(args.config)['ready'])
        self.assertTrue(inspect(args.config,['basic'])['ready'])

    def worker(self, sidecar=True):
        worker=self.root/'worker.py'
        worker.write_text("from pathlib import Path\np=Path('runs');p.write_text(str(int(p.read_text())+1 if p.exists() else 1))\nPath('results.csv').write_text('id,score\\na,1\\n')\n" + ("Path('metadata.json').write_text('{}')\n" if sidecar else ''))
        return dict(enabled=True,argv=[sys.executable,str(worker)],dependencies=[str(worker)],columns={'value':'score'},artifacts=['{stage_dir}/metadata.json'])

    def test_cache_requires_unchanged_sidecars(self):
        args=self.inputs(self.worker());collect(args);collect(args)
        stage=args.out_dir/'stages/probe'
        self.assertEqual((stage/'runs').read_text(),'1')
        (stage/'metadata.json').unlink();collect(args)
        self.assertEqual((stage/'runs').read_text(),'2')
        (stage/'metadata.json').write_text('tampered');collect(args)
        self.assertEqual((stage/'runs').read_text(),'3')

    def test_missing_sidecar_is_a_stage_failure(self):
        args=self.inputs(self.worker(sidecar=False))
        with self.assertRaises(SystemExit):collect(args)
        status=json.loads((args.out_dir/'status.json').read_text())
        self.assertEqual(status['stages']['probe']['status'],'failed')

    def test_duplicate_csv_headers_are_rejected(self):
        path=self.root/'input.csv';path.write_text('id,score,score\na,1,2\n')
        with self.assertRaisesRegex(ValueError,'duplicate'):
            merge_stage(path,{'columns':{'value':'score'}},{'a'})

    def test_scoring_checks_ids_and_saved_statistics(self):
        df=pd.DataFrame({'id':['a','b'],'logP':[1.,3.]})
        normalizer=fit_normalizer(df,{'a','b'})
        self.assertEqual(len(score(df,normalizer)),2)
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            score(pd.concat([df,df]),normalizer)
        bad=df.copy();bad.loc[0,'id']=''
        with self.assertRaisesRegex(ValueError,'nonempty'):score(bad,normalizer)
        corrupted=copy.deepcopy(normalizer);corrupted['statistics']['logP']['std']=-1
        with self.assertRaisesRegex(ValueError,'nonnegative'):score(df,corrupted)
        corrupted=copy.deepcopy(normalizer);corrupted['log_epsilon']=0
        with self.assertRaisesRegex(ValueError,'epsilon'):score(df,corrupted)

    def test_evaluation_rejects_empty_conventions_and_reserved_metric_names(self):
        rows=[dict(id='a',receptor_id='r',cyclization_distance=4,rosetta_dG=-1)]
        for value in [None,False,{},' ']:
            with self.assertRaisesRegex(ValueError,'convention'):
                evaluate(rows,{'cyclization_distance':value},metrics=['rosetta_dG'])
        with self.assertRaisesRegex(ValueError,'automatically'):
            evaluate(rows,{'cyclization_distance':'CA'},metrics=['valid_fraction'])

    def test_property_labels_must_match_trained_targets(self):
        class Head(torch.nn.Module):
            n_properties=1
            def __init__(self,key):super().__init__();self.property_keys=[key]
        heads=[Head('affinity'),Head('Pproxy')]
        with patch('apcyc.guidance.load_property_predictor',side_effect=heads):
            with self.assertRaisesRegex(ValueError,'names differ'):
                load_property_predictors(['a','b'],torch.device('cpu'),keys=['Pproxy','affinity'])
        with patch('apcyc.guidance.load_property_predictor',side_effect=heads):
            predictor=load_property_predictors(['a','b'],torch.device('cpu'),keys=['affinity','Pproxy'])
        self.assertEqual(weights_from_map(predictor,{'Pproxy':2,'affinity':3}),[3,2])
        with self.assertRaisesRegex(ValueError,'exactly'):
            weights_from_map(predictor,{'affinity':3})

    def test_guidance_weights_require_finite_matching_vector(self):
        for weights in [[1],[[1,2]],[float('nan'),1],torch.tensor([1.,float('inf')])]:
            with self.assertRaisesRegex(ValueError,'finite'):
                parse_guidance_w(weights,2,torch.device('cpu'))


    def test_cache_tracks_prepared_fasta_and_reuses_identical_preparation(self):
        worker = self.root/'fasta_worker.py'
        worker.write_text("from pathlib import Path\nimport sys\nfasta=Path(sys.argv[1]).read_text()\np=Path('runs');p.write_text(str(int(p.read_text())+1 if p.exists() else 1))\nPath('results.csv').write_text('id,score\\na,'+str(fasta.count('A'))+'\\n')\n")
        args = self.inputs(dict(enabled=True, argv=[sys.executable, str(worker), '{fasta}'],
                                dependencies=[str(worker)], columns={'value':'score'}))
        collect(args)
        stage = args.out_dir/'stages/probe'
        original_prepare = collector.prepare
        def revised_preparation(records, out):
            result = original_prepare(records, out)
            path = out/'sequences.fasta'; path.write_text(path.read_text().replace('AAAA', 'AAA'))
            return result
        with patch.object(collector, 'prepare', side_effect=revised_preparation):
            collect(args)
            self.assertEqual((stage/'runs').read_text(), '2')
            status = json.loads((args.out_dir/'status.json').read_text())
            self.assertFalse(status['stages']['probe']['cached'])
            with (args.out_dir/'raw_properties.csv').open() as handle:
                self.assertEqual(next(csv.DictReader(handle))['value'], '3')
            collect(args)
        self.assertEqual((stage/'runs').read_text(), '2')
        self.assertTrue(json.loads((args.out_dir/'status.json').read_text())['stages']['probe']['cached'])

if __name__=='__main__':unittest.main()
