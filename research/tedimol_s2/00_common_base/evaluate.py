"""Paired ChemBI evaluation with explicit denominators and no remote services."""
import argparse
import csv
import warnings
from pathlib import Path
import numpy as np
import selfies as sf
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator
from nltk.translate.bleu_score import corpus_bleu
from Levenshtein import distance
from common import ROOT, read_rows, read_json, sha256, write_json


def parse_molecule(value):
    if not value:
        return None
    try:
        mol = Chem.MolFromSmiles(value)
        return mol if mol is not None and mol.GetNumAtoms() else None
    except Exception:
        return None


def score_rows(predictions, references, training_rows):
    """Compare each generated molecule ONLY with its own description's reference."""
    RDLogger.DisableLog('rdApp.*')
    ref_by_id = {r['id']:r for r in references}
    if not predictions or len({r['id'] for r in predictions}) != len(predictions):
        raise ValueError('Empty or duplicate prediction IDs')
    train_set = set()
    for r in training_rows:
        mol = parse_molecule(r['smiles'])
        if mol is not None:
            train_set.add(Chem.MolToSmiles(mol, isomericSmiles=True))
    pairs = []
    valid_canonical = []
    refs, hyps = [], []
    invalid = []
    raw_exact, canonical_exact = 0, 0
    edit = []
    morgan = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    per_sample = []
    for pred in predictions:
        cid = pred['id']
        if cid not in ref_by_id:
            raise ValueError(f'Unknown prediction ID: {cid}')
        ref = ref_by_id[cid]
        if pred['reference_smiles'] != ref['smiles']:
            raise ValueError(f'Reference mismatch for ID {cid}')
        gen = ''
        error = ''
        try:
            gen = sf.decoder(pred['generated_selfies'])
        except Exception as e:
            error = type(e).__name__
        mol, target = parse_molecule(gen), parse_molecule(ref['smiles'])
        refs.append([list(ref['smiles'])])
        hyps.append(list(gen))
        edit.append(distance(gen, ref['smiles']))
        raw_exact += gen == ref['smiles']
        out = dict(id=cid, generated_smiles=gen, valid=mol is not None, error=error,
                   maccs=None, rdk=None, morgan=None)
        if mol is None:
            invalid.append(cid)
        else:
            canon = Chem.MolToSmiles(mol, isomericSmiles=True)
            valid_canonical.append(canon)
            if target is not None:
                target_canon = Chem.MolToSmiles(target, isomericSmiles=True)
                canonical_exact += canon == target_canon
                out['maccs'] = DataStructs.TanimotoSimilarity(MACCSkeys.GenMACCSKeys(mol),MACCSkeys.GenMACCSKeys(target))
                out['rdk'] = DataStructs.TanimotoSimilarity(Chem.RDKFingerprint(mol),Chem.RDKFingerprint(target))
                out['morgan'] = DataStructs.TanimotoSimilarity(morgan.GetFingerprint(mol),morgan.GetFingerprint(target))
                pairs.append((out['maccs'],out['rdk'],out['morgan']))
        per_sample.append(out)
    unique = set(valid_canonical)
    n = len(predictions)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        bleu = float(corpus_bleu(refs,hyps))
    result = dict(n_generated=n,n_valid=len(valid_canonical),n_valid_reference_pairs=len(pairs),
        validity=len(valid_canonical)/n, exact_raw_smiles=raw_exact/n, exact_canonical_isomeric_smiles=canonical_exact/n,
        bleu_character_smiles=bleu, levenshtein_character_smiles=float(np.mean(edit)),
        maccs_paired_valid_mean=float(np.mean([p[0] for p in pairs])) if pairs else None,
        rdk_paired_valid_mean=float(np.mean([p[1] for p in pairs])) if pairs else None,
        morgan_r2_2048_paired_valid_mean=float(np.mean([p[2] for p in pairs])) if pairs else None,
        uniqueness_valid_canonical=len(unique)/len(valid_canonical) if valid_canonical else None,
        novelty_unique_valid_canonical=len(unique-train_set)/len(unique) if unique else None,
        eos_found_fraction=sum(str(p.get('eos_found','')).lower()=='true' for p in predictions)/n,
        invalid_ids=invalid,
        fcd=None, text2mol=None,
        metrics_scope='seven basic paper-related metrics with explicit local tokenization; not exact official evaluator',
        missing_metric_reason={'fcd':'Optional --fcd, fcd-torch package required',
                               'text2mol':'Supplied archive lacks the Text2Mol evaluator weights/tokenizer/implementation; not fabricated'},
        definitions={'BLEU':'corpus 1-4 gram, character SMILES, no smoothing',
                     'Exact raw':'decoded uncanonicalized SMILES equality, denominator all generated',
                     'fingerprints':'paired by molecule ID; valid generated + valid reference pairs only',
                     'uniqueness':'unique canonical isomeric SMILES / valid outputs',
                     'novelty':'unique valid canonical SMILES absent from ALL supplied training SMILES / unique valid'})
    return result,per_sample


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--allow-subset',action='store_true')
    p.add_argument('--fcd',action='store_true')
    args=p.parse_args()
    meta=read_json(args.input.with_suffix('.metadata.json'))
    if sha256(args.input)!=meta['output_sha256']:
        raise ValueError('Sample file hash mismatch')
    if sha256(ROOT/'data/train_ChemBI.csv')!=meta['train_csv_sha256']:
        raise ValueError('Training reference changed; novelty must use the checkpoint training CSV')
    if sha256(ROOT/'data/selfies_vocab_CHEBI.txt')!=meta['vocab_sha256']:
        raise ValueError('Vocabulary changed since generation')
    split=meta['split']
    ref_path=ROOT/'data'/(split+'.csv')
    if sha256(ref_path)!=meta['csv_sha256']:
        raise ValueError('Reference CSV changed')
    with args.input.open(encoding='utf-8',newline='') as f:
        predictions=list(csv.DictReader(f))
    refs=read_rows(ref_path)
    if len(predictions)!=meta['count']:
        raise ValueError('Generation count mismatch')
    if (meta['subset'] or len(predictions)!=len(refs)) and not args.allow_subset:
        raise ValueError('Partial results: use --allow-subset explicitly for smoke evaluation')
    if not meta['subset'] and {r['id'] for r in predictions}!={r['id'] for r in refs}:
        raise ValueError('Full-split evaluation requires all IDs exactly once')
    result,details=score_rows(predictions,refs,read_rows(ROOT/'data/train_ChemBI.csv'))
    if args.fcd:
        from fcd_torch import FCD
        valid=[d['generated_smiles'] for d in details if d['valid']]
        evaluated_ids={r['id'] for r in predictions}
        truth=[r['smiles'] for r in refs if r['id'] in evaluated_ids and parse_molecule(r['smiles']) is not None]
        if len(valid)<2:
            result['missing_metric_reason']['fcd']='Fewer than two valid generated molecules'
        else:
            result['fcd']=float(FCD(device='cpu',n_jobs=1)(truth,valid))
            result['missing_metric_reason'].pop('fcd')
    result['generation']=meta
    result['subset']=meta['subset']
    result['training_scope']='QA_ONLY' if meta['qa_training'] else ('FULL_DATA_TARGET_REACHED' if meta['training_complete'] else 'FULL_DATA_INTERMEDIATE_CHECKPOINT')
    result['evaluation_sha256']=sha256(Path(__file__))
    output=args.input.with_suffix('.metrics.json')
    write_json(output,result)
    write_json(args.input.with_suffix('.per_sample.json'),details)
    print({k:result[k] for k in ('training_scope','n_generated','n_valid','validity','exact_canonical_isomeric_smiles','morgan_r2_2048_paired_valid_mean')})
    print(f'EVALUATION PASS: {output}')


if __name__=='__main__':
    main()
