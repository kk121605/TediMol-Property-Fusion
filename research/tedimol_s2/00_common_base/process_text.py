"""Senior-compatible .pt dictionary, local BioLinkBERT, original unmasked encoder call.

Only read text/IDs; no molecule target is required to build features.
Legacy mask behavior is retained for baseline parity and documented, not called correct masking.
"""
import argparse
import os
import psutil
import shutil
import torch
from transformers import AutoModel, AutoTokenizer
from common import ROOT, read_rows, read_json, sha256, write_json, cache_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input',required=True, choices=['train_ChemBI','test_ChemBI'])
    parser.add_argument('--batch-size',type=int,default=1, help='Original encoder batch size is 1')
    parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    args = parser.parse_args()
    if args.batch_size < 1 or args.limit < 0:
        parser.error('Invalid batch or limit')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    if args.device == 'cpu' and not 0 < args.limit <= 16:
        raise ValueError('Local CPU encoding is limited to 16 QA rows')
    os.environ['HF_HUB_OFFLINE']='1'
    directory = ROOT/'models/BioLinkBERT-base'
    encoder_files = ('config.json','pytorch_model.bin','tokenizer.json','tokenizer_config.json','vocab.txt','special_tokens_map.json')
    hashes = {n:sha256(directory/n) for n in encoder_files}
    rows = read_rows(ROOT/'data'/(args.input+'.csv'))
    if args.limit:
        rows = rows[:args.limit]
    dest = cache_path(args.input,args.limit)
    sidecar = dest.with_suffix('.metadata.json')
    identity = dict(csv_sha256=sha256(ROOT/'data'/(args.input+'.csv')), ids=[r['id'] for r in rows],
        encoder_hashes=hashes, length=256, dtype='float32', attention_mask_passed=False,
        transformers_version=__import__('transformers').__version__, torch_version=torch.__version__,
        batch_size=args.batch_size, preprocess_sha256=sha256(__file__))
    if dest.exists() or sidecar.exists():
        if dest.exists() and sidecar.exists():
            old = read_json(sidecar)
            if all(old.get(k)==v for k,v in identity.items()) and old['complete'] and sha256(dest)==old['cache_sha256']:
                print('CACHE ALREADY COMPLETE:',dest)
                return
        raise FileExistsError('Cache conflict. Preserve/rename the existing cache before retrying.')
    tensor_bytes = len(rows)*256*(768*4+8)
    if psutil.virtual_memory().available < tensor_bytes + 4*2**30:
        raise MemoryError(f'Encoding requires at least {(tensor_bytes+4*2**30)/2**30:.1f} GiB AVAILABLE host RAM; use a larger-RAM instance')
    if shutil.disk_usage(ROOT).free < tensor_bytes + 2*2**30:
        raise OSError('Not enough free disk for the complete .pt text cache')
    model = AutoModel.from_pretrained(directory,local_files_only=True).eval().to(args.device)
    model.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(directory,local_files_only=True)
    features, truncated = {}, 0
    with torch.inference_mode():
        for start in range(0,len(rows),args.batch_size):
            batch = rows[start:start+args.batch_size]
            texts = [r['text'] for r in batch]
            truncated += sum(len(x)>256 for x in tokenizer(texts,truncation=False,padding=False)['input_ids'])
            inputs = tokenizer(texts,max_length=256,truncation=True,padding='max_length',return_tensors='pt')
            # Exact input convention of supplied process_text.py: no attention_mask/token_type_ids.
            states = model(inputs['input_ids'].to(args.device)).last_hidden_state.float().cpu()
            if not torch.isfinite(states).all():
                raise ValueError('Nonfinite encoder features')
            for i,row in enumerate(batch):
                features[row['id']] = dict(states=states[i:i+1].clone(),mask=inputs['attention_mask'][i:i+1].clone())
            done=start+len(batch)
            if start==0 or done%190==0 or done==len(rows):
                print(f'{args.input}: {done}/{len(rows)}',flush=True)
    temp=dest.with_suffix('.pt.partial')
    torch.save(features,temp)
    os.replace(temp,dest)
    write_json(sidecar,identity|dict(complete=True,cache_sha256=sha256(dest),cache_bytes=dest.stat().st_size,
        truncated_text_rows=truncated,device=args.device,legacy_mask_behavior=True))
    print('CACHE PASS:',dest,flush=True)


if __name__=='__main__':
    main()
