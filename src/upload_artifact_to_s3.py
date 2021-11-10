### This script upload W&B artifacts to AWS S3:// 
### For more information, refer here: https://wandb.ai/aarora/reports/reports/How-Weights-and-Biases-can-help-you-with-Audits-and-Regulatory-Guidelines--VmlldzoxMTc1ODk4#but-what-about-sharing-model-artifacts-with-clients?

import hashlib 
import argparse
import logging
import tempfile
import warnings 
import boto3
from timm.utils.log import setup_default_logging
import wandb
import os
import yaml

_logger = logging.getLogger('train')
s3 = boto3.client('s3')


def main(args):
    api = wandb.Api()
    artifact = api.artifact(f'{args.project}/{args.filename}:{args.alias}')

    with tempfile.TemporaryDirectory() as tmpdir:
        path = artifact.download(tmpdir)
        fname = os.listdir(path)[0]
        fpath = path + '/' + fname

        # create hash dict
        f = open(fpath,"rb")
        bytes = f.read() 
        readable_hash = hashlib.sha256(bytes).hexdigest()
        hash_dict = {'filename': fname, 'hash': readable_hash}
        # dump dict to YAML
        with open(tmpdir+'/hash_dict.yaml', 'w') as outfile:
            yaml.dump(hash_dict, outfile)

        # get aws hash if file is present in bucket
        s3_objects = s3.list_objects(Bucket=args.bucket, Prefix=args.key)
        hash_dict_s3 = {'hash': -1}
        if 'Contents' in s3_objects.keys():
            try: s3.download_file(args.bucket, 'hash_dict.yaml', tmpdir+'/hash_dict_s3.yaml')
            except: raise ValueError(f"File hash_dict.yaml not found in s3://{args.bucket}")
            with open (tmpdir+'/hash_dict_s3.yaml') as f:
                hash_dict_s3 = yaml.load(f, Loader=yaml.FullLoader)
            
            # compare hash
            if hash_dict['hash'] == hash_dict_s3['hash']: 
                warnings.warn(f'File {fname} already present in s3://{args.bucket}. Nothing to do.')
        
        # upload file to S3
        if hash_dict_s3['hash']!=hash_dict['hash']:
            s3.upload_file(fpath, args.bucket, fname)
            s3.upload_file(tmpdir+'/hash_dict.yaml', args.bucket, 'hash_dict.yaml')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=str, required=True)
    parser.add_argument('--filename', type=str, required=True)
    parser.add_argument('--alias', type=str, required=True)
    parser.add_argument('--bucket', type=str, required=True)
    parser.add_argument('--key', type=str, default='', required=False)

    args = parser.parse_args()

    setup_default_logging()

    main(args)
