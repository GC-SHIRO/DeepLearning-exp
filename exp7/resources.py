"""下载指定版本的原始数据与 BERT 权重，复用已校验的文件。"""
from pathlib import Path
import hashlib
import requests

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
BERT = ROOT.parent / '.cache' / 'exp7' / 'bert-tiny'
IMDB_REV = 'e6281661ce1c48d982bc483cf8a173c1bbeb5d31'
SST_REV = '8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb'
BERT_REV = '30b0a37ccaaa32f332884b96992754e246e48c5f'

def prepare():
    resources = []
    for dataset, folder, rev, splits in [('imdb','plain_text',IMDB_REV,['train','test']),('sst2','data',SST_REV,['train','validation'])]:
        for split in splits:
            resources.append((f'datasets/stanfordnlp/{dataset}',rev,f'{folder}/{split}-00000-of-00001.parquet',DATA/f'{dataset}_{split}.parquet'))
    for name in ['config.json','vocab.txt','model.safetensors']:
        resources.append(('google/bert_uncased_L-2_H-128_A-2',BERT_REV,name,BERT/name))
    for repo,rev,name,path in resources:
        path.parent.mkdir(parents=True,exist_ok=True)
        proof = path.with_name(path.name+'.sha256')
        if path.exists() and proof.exists() and hashlib.sha256(path.read_bytes()).hexdigest()==proof.read_text():
            continue
        for host in ['hf-mirror.com','huggingface.co']:
            try:
                url=f'https://{host}/{repo}/resolve/{rev}/{name}'
                response=requests.get(url,timeout=(10,120)); response.raise_for_status()
                content = response.content
                if name.endswith('.parquet') and not (content.startswith(b'PAR1') and content.endswith(b'PAR1')):
                    raise ValueError('Parquet 文件不完整')
                if 'text/html' in response.headers.get('content-type',''): raise ValueError('收到 HTML 页面')
                tmp=path.with_suffix('.part'); tmp.write_bytes(content); tmp.replace(path)
                proof.write_text(hashlib.sha256(content).hexdigest()); print(path.name,len(content),flush=True); break
            except (requests.RequestException,ValueError):
                if host=='huggingface.co': raise

if __name__=='__main__': prepare()
