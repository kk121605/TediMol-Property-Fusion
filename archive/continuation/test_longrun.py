"""Resume-contract and real small-model continuity checks; no cloud-performance claim."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
COMMON = HERE.parent / '12_三架构融合代码与学习/00_共用第一阶段底座与训练工具'
sys.path.insert(0, str(COMMON))
sys.path.insert(0, str(HERE.parent / '13_S2_10k统一筛选'))


class ResumeTests(unittest.TestCase):
    def fixture(self):
        import longrun_contract as c
        config = json.loads((HERE.parents[1] / '06_AutoDL实验结果/S2_10k四分支统一筛选_20261012/01_轻量证据包_已解包/runs_s2_10k_seed19991009/a3_adaln_zero/config.json').read_text(encoding='utf-8'))
        return dict(format=c.FORMAT, step=10000, config=config,
                    provenance={'s2_sources': {'old.py': 'abc'}, 'runtime_versions': {'torch': 'original'}},
                    model={}, optimizer={}, ema={}, sampler={'cursor': 16, 'epoch': 7, 'permutation': [2, 0, 1]},
                    initialization_report={})

    def test_promote_extends_budget_without_resetting_state_or_schedule(self):
        import longrun_contract as c
        cp = self.fixture()
        config, provenance = c.promote(cp, 'a3_adaln_zero', 'a' * 64, {'new.py': 'def'}, '/old/latest.pt')
        self.assertEqual(config['budget_steps'], 1200000)
        self.assertEqual(cp['step'], 10000)
        self.assertEqual(cp['config']['budget_steps'], 10000)
        self.assertEqual(config['lr_horizon_steps'], 1200000)
        self.assertEqual(config['lr'], 1e-5)
        self.assertEqual(provenance['longrun']['origin_step'], 10000)
        extended = dict(cp, config=config, provenance=provenance, step=100000)
        c.validate(extended, 'a3_adaln_zero', {'new.py': 'def'})
        with self.assertRaises(ValueError):
            c.validate(extended, 'a3_adaln_zero', {'new.py': 'changed'})

    def test_a30_contract_saves_every_100k_through_1200k(self):
        import longrun_contract as c
        self.assertEqual(c.TARGET, 1200000)
        self.assertEqual(c.MILESTONES, tuple(range(100000, 1200001, 100000)))
        self.assertEqual(c.FIXED['lr_horizon_steps'], 1200000)

    def test_changed_hyperparameters_incomplete_state_and_wrong_mode_are_rejected(self):
        import longrun_contract as c
        for key, value in [('lr', 5e-5), ('lr_horizon_steps', 200000), ('ema_rate', 0.99), ('microbatch', 1)]:
            cp = self.fixture(); cp['config'][key] = value
            with self.assertRaises(ValueError, msg=key):
                c.promote(cp, 'a3_adaln_zero', 'a' * 64, {}, 'old')
        cp = self.fixture(); del cp['optimizer']
        with self.assertRaises(ValueError): c.promote(cp, 'a3_adaln_zero', 'a' * 64, {}, 'old')
        with self.assertRaises(ValueError): c.promote(self.fixture(), 'a1_prefix', 'a' * 64, {}, 'old')

    def test_restore_reproduces_next_real_diffusion_update_for_all_four_modes(self):
        import torch
        from comparison_runtime import Bridge, MODES, update
        from improved_diffusion.resample import UniformSampler
        from runtime import make_diffusion, seed_all
        from train_longrun import restore_training_state, next_batch
        spec = __import__('importlib.util', fromlist=['spec_from_file_location'])
        source = COMMON.parent / '04_本地验收与来源/test_three_methods_cpu.py'
        module_spec = spec.spec_from_file_location('three_cpu', source)
        m = spec.module_from_spec(module_spec); module_spec.loader.exec_module(m)
        torch.set_num_threads(1)
        seed_all(17)
        base = m.TransformerNetModel2(323, **m.CONFIG)
        parent = {'model': copy.deepcopy(base.state_dict()), 'ema': {n: p.detach().clone() for n,p in base.named_parameters()}}
        cfg = dict(lr=1e-5, lr_horizon_steps=1200000, microbatch=1, seed=19991009, grad_clip=2.4, ema_rate=0.9999)
        molecules = torch.randint(0, 323, (2, 12))
        batch = (molecules, torch.randn(2,10,768), torch.ones(2,10), torch.tensor([m.MEANS,m.MEANS]), torch.ones(2,3,dtype=torch.bool))
        for mode in MODES:
            model = m.build(mode, parent).train(); optimizer, ema = m.setup(model)
            diffusion = make_diffusion(); sampler = UniformSampler(diffusion)
            update(model, Bridge(model,mode), optimizer, ema, diffusion, sampler, batch, 9999, cfg, 'cpu')
            checkpoint = copy.deepcopy(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), ema=ema,
                config=cfg, sampler=dict(permutation=torch.tensor([2,0,1]),cursor=2,epoch=7)))
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)/'saved.pt'; torch.save(checkpoint,path)
                saved = torch.load(path,weights_only=True,mmap=True)
            expected = update(model, Bridge(model,mode), optimizer, ema, diffusion, sampler, batch, 10000, cfg, 'cpu')
            resumed = m.build(mode,parent).train(); resumed.load_state_dict(saved['model'])
            opt2, ema2, order, cursor, epoch = restore_training_state(resumed,saved,'cpu')
            # Optimizer restoration must detach step tensors from the mmap file.
            saved_step = next(iter(saved['optimizer']['state'].values()))['step'].clone()
            self.assertEqual((cursor,epoch),(2,7)); self.assertTrue(torch.equal(order,torch.tensor([2,0,1])))
            actual = update(resumed,Bridge(resumed,mode),opt2,ema2,diffusion,sampler,batch,10000,cfg,'cpu')
            self.assertTrue(torch.equal(saved_step,next(iter(saved['optimizer']['state'].values()))['step']))
            self.assertEqual(expected['loss'],actual['loss'])
            self.assertEqual(actual['lr'], 1e-5*(1-10000/1200000))
            self.assertTrue(all(torch.equal(v,resumed.state_dict()[n]) for n,v in model.state_dict().items()))
            self.assertTrue(all(torch.equal(v,ema2[n]) for n,v in ema.items()))
            # Same retained sampler boundary produces exactly the same next epoch.
            a = next_batch(order,cursor,epoch,2,3,0)
            b = next_batch(checkpoint['sampler']['permutation'],2,7,2,3,0)
            self.assertEqual(a[1:],b[1:]); self.assertTrue(torch.equal(a[0],b[0]))

    def test_sweep_summary_does_not_confuse_changes_with_direction_and_package_excludes_models(self):
        import hashlib
        import tarfile
        from summarize_longrun import summarize,package
        root=HERE.parents[1]/'06_AutoDL实验结果/S2_10k四分支统一筛选_20261012/01_轻量证据包_已解包/runs_s2_10k_seed19991009'
        a0=summarize(root/'a0c/a0c_RAW_step10000.csv')
        a2=summarize(root/'a2_rt_numeric/a2_rt_numeric_EMA_step10000.csv')
        self.assertEqual(a0['sweep_numeric_response_groups'],0)
        self.assertEqual(a2['valid'],58)
        self.assertEqual(a2['sweep_numeric_response_groups'],2)
        self.assertEqual(a2['sweep_strictly_increasing_groups'],0)
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)/'runs';base.mkdir();(base/'latest.pt').write_bytes(b'not-evidence')
            (base/'status.json').write_text('{"step":10000}')
            package(base,Path(tmp)/'out')
            archive=next((Path(tmp)/'out').glob('*.tar.gz'))
            with tarfile.open(archive) as tar:
                names=tar.getnames(); self.assertFalse(any(n.endswith('.pt') for n in names))
                manifest=json.load(tar.extractfile('runs/evidence_manifest.json'))
                for item in manifest['files']:
                    body=tar.extractfile('runs/'+item['path']).read()
                    self.assertEqual(hashlib.sha256(body).hexdigest(),item['sha256'])

    def test_interrupted_initialization_is_preserved_and_failed_save_keeps_latest(self):
        import torch
        from train_longrun import durable_save,preserve_incomplete_initialization
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'a1';run.mkdir()
            (run/'config.json').write_text('{}');(run/'latest.pt.tmp').write_bytes(b'interrupted')
            for name in ('config.json.tmp','provenance.json.tmp','initialization_report.json.tmp','status.json.tmp'):
                (run/name).write_text('interrupted-json')
            preserve_incomplete_initialization(run)
            self.assertFalse(run.exists())
            self.assertEqual(len(list(root.glob('a1_incomplete_initialization_*'))),1)
            path=root/'latest.pt';durable_save({'step':1},path)
            with self.assertRaises(Exception): durable_save({'bad':lambda:None},path)
            self.assertEqual(torch.load(path,weights_only=True)['step'],1)
            durable_save({'step':2},path)
            self.assertEqual(torch.load(path,weights_only=True)['step'],2)

    def test_sampling_cache_rejects_different_checkpoint_weights_or_changed_csv(self):
        import hashlib
        from sample_longrun import reusable
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'samples.csv';output.write_bytes(b'valid\n1\n')
            metadata=dict(checkpoint_sha256='a'*64,checkpoint_step=50000,weights='RAW',
                longrun_sources={'runner':'123'},output_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
            output.with_suffix('.metadata.json').write_text(json.dumps(metadata))
            self.assertTrue(reusable(output,'a'*64,50000,'RAW',{'runner':'123'}))
            self.assertFalse(reusable(output,'b'*64,50000,'RAW',{'runner':'123'}))
            self.assertFalse(reusable(output,'a'*64,50000,'EMA',{'runner':'123'}))
            output.write_bytes(b'valid\n0\n')
            self.assertFalse(reusable(output,'a'*64,50000,'RAW',{'runner':'123'}))

    def test_a30_cluster_entrypoints_use_shared_root_and_100k_segments(self):
        env = (HERE / 'a30_env.example').read_text(encoding='utf-8')
        submit = (HERE / 'submit_a30_chain.sh').read_text(encoding='utf-8')
        segment = (HERE / 'a30_segment.sbatch').read_text(encoding='utf-8')
        self.assertIn('/nfs/home/yuanhang/lyk', env)
        self.assertIn('envs/tedimol310_cf/bin/python', env)
        self.assertIn('100000', submit)
        self.assertIn('1200000', submit)
        self.assertIn('sbatch', submit)
        self.assertIn('train_longrun.py', segment)
        self.assertIn('sample_longrun.py', segment)


if __name__ == '__main__':
    unittest.main(verbosity=2)
