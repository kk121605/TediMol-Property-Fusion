"""Fixed engineering panel for S2 checkpoints; seen-train evidence only."""
from __future__ import annotations
import argparse, csv, gc, json, math, os, time
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from common import open_cache, MoleculeTokenizer, sha256, write_json
from property_data import load_aligned_properties
from property_runtime import DEFAULT_PROPERTY_DIR
from runtime import make_diffusion, seed_all
from comparison_runtime import NAMES, load, source_receipt, verify_train_cache
from property_metrics import compute_generated_properties
from train_s2 import S2_PROTOCOL, s2_sources

SWEEP_INDICES = (0, 100, 1000, 10000)
FIELDS = ['regime','source_index','id','sweep_property','z','generated_selfies','generated_smiles','valid','exact_reference','morgan_similarity']
for name in NAMES: FIELDS += [f'target_{name}', f'generated_{name}', f'abs_error_{name}']

def evaluate(value):
    try:
        smiles, valid, computed = compute_generated_properties(value, list(NAMES))
        mol = Chem.MolFromSmiles(smiles) if valid else None
        valid = bool(mol is not None and mol.GetNumAtoms() > 0 and all(math.isfinite(computed[n]) for n in NAMES))
        return smiles, valid, computed if valid else {}
    except (TypeError, ValueError, RuntimeError, OverflowError, ZeroDivisionError):
        return '', False, {}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--weights',choices=('RAW','EMA'),required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--expected-steps',type=int,default=10000); a=p.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    model, checkpoint = load(a.checkpoint, 'cuda', a.weights)
    if checkpoint['step'] != a.expected_steps: raise RuntimeError('checkpoint step mismatch')
    if checkpoint['config'].get('protocol') != S2_PROTOCOL or checkpoint['provenance'].get('s2_sources') != s2_sources():
        raise RuntimeError('S2 source/protocol mismatch')
    rows, states, masks, cache_meta = open_cache('train_ChemBI', 0)
    verify_train_cache(checkpoint['provenance']['train_text_cache_sha256'], cache_meta['cache_sha256'])
    values, observed = load_aligned_properties(DEFAULT_PROPERTY_DIR/'train_properties.csv',[r['id'] for r in rows],NAMES)
    means=np.asarray(checkpoint['config']['property_means'],dtype=np.float32); stds=np.asarray(checkpoint['config']['property_stds'],dtype=np.float32)
    tokenizer=MoleculeTokenizer(); model.eval(); diffusion=make_diffusion(200); fp=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
    a.output.parent.mkdir(parents=True,exist_ok=True); partial=a.output.with_suffix('.csv.partial'); metadata=a.output.with_suffix('.metadata.json'); start=time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    for path in (a.output, partial, metadata):
        if path.exists():
            path.rename(path.with_name(path.name + f'.preserved_{time.time_ns()}'))
    with partial.open('w',encoding='utf-8',newline='') as stream, torch.inference_mode():
        writer=csv.DictWriter(stream,fieldnames=FIELDS); writer.writeheader()
        def generate(indices,targets,regime,prop='',z='',noise=None,seed=121):
            seed_all(seed); desc=(states[indices].to('cuda'),masks[indices].to('cuda')); kwargs={} if checkpoint['config']['mode']=='a0c' else {'chemical_metric':torch.from_numpy(targets).to('cuda'),'property_observed':torch.ones(len(indices),3,dtype=torch.bool,device='cuda')}
            sample=diffusion.p_sample_loop(model,(len(indices),258,model.in_channels),noise=noise,device='cuda',clip_denoised=False,top_p=1.,progress=False,desc=desc,model_kwargs=kwargs)
            for local,(index,sequence) in enumerate(zip(indices,model.get_logits(sample).argmax(-1).cpu().tolist())):
                selfies=tokenizer.decode(sequence); smiles,valid,computed=evaluate(selfies); ref=Chem.MolFromSmiles(rows[index]['smiles']); gen=Chem.MolFromSmiles(smiles) if valid else None
                exact=int(valid and Chem.MolToSmiles(ref,isomericSmiles=True)==smiles); sim=DataStructs.TanimotoSimilarity(fp.GetFingerprint(ref),fp.GetFingerprint(gen)) if valid else math.nan
                row={'regime':regime,'source_index':index,'id':rows[index]['id'],'sweep_property':prop,'z':z,'generated_selfies':selfies,'generated_smiles':smiles,'valid':int(valid),'exact_reference':exact,'morgan_similarity':sim}
                for k,n in enumerate(NAMES): row.update({f'target_{n}':float(targets[local,k]),f'generated_{n}':computed.get(n,math.nan),f'abs_error_{n}':abs(computed[n]-float(targets[local,k])) if n in computed else math.nan})
                writer.writerow(row)
            stream.flush()
        for start_i in range(0,32,8): generate(list(range(start_i,start_i+8)),values[start_i:start_i+8],'reference',seed=121+start_i)
        for index in SWEEP_INDICES:
            seed=121+index; seed_all(seed); noise=torch.randn(1,258,model.in_channels,device='cuda')
            for k,n in enumerate(NAMES):
                for z in (-1,0,1):
                    target=values[index:index+1].copy(); target[0,k]=means[k]+z*stds[k]; generate([index],target,'sweep',n,z,noise=noise,seed=seed)
    os.replace(partial,a.output)
    with a.output.open(encoding='utf-8', newline='') as f:
        assert len(list(csv.DictReader(f))) == 68, 'incomplete S2 sampling panel'
    write_json(metadata,{'output_sha256':sha256(a.output),'checkpoint_sha256':sha256(a.checkpoint),'weights':a.weights,'mode':checkpoint['config']['mode'],'checkpoint_step':a.expected_steps,'source_sha256':source_receipt(),'s2_sources':s2_sources(),'sample_seed':121,'sample_steps':200,'sample_length':258,'reference_indices':list(range(32)),'sweep_indices':list(SWEEP_INDICES),'sweep_z':[-1,0,1],'text_cache_sha256':cache_meta['cache_sha256'],'stress_sweep_may_conflict_with_text':True,'elapsed_seconds':time.perf_counter()-start,'max_gpu_gib':torch.cuda.max_memory_allocated()/2**30,'seen_train_panel':True,'not_a_heldout_or_superiority_result':True})
    print('S2_SAMPLE_PASS:',checkpoint['config']['mode'],a.weights,a.output,flush=True); del model,checkpoint; gc.collect()

if __name__=='__main__': main()

