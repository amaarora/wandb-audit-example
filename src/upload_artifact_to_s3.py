### This script upload W&B artifacts to AWS S3:// 
### For more information, refer here: https://wandb.ai/aarora/reports/reports/How-Weights-and-Biases-can-help-you-with-Audits-and-Regulatory-Guidelines--VmlldzoxMTc1ODk4#but-what-about-sharing-model-artifacts-with-clients?

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
api = wandb.Api()


def main(args):
    artifact = api.artifact(f'{args.project}/{args.filename}:{args.alias}')
    digest = artifact.digest

    with tempfile.TemporaryDirectory() as tmpdir:
        path = artifact.download(tmpdir)
        fname = os.listdir(path)[0]
        fpath = path + '/' + fname

        _logger.info(f"Downloaded artifact {fname} to {fpath} locally.")

        try: 
            metadata = s3.head_object(Bucket=args.bucket, Key=fname)['Metadata']
        except: 
            warnings.warn(f"""File {fname} does not already exist in Bucket {args.bucket} on AWS.\
                            Cleaning up AWS bucket for any existing files, and uploading new \
                            artifact.""")
            bucket = boto3.resource('s3').Bucket(args.bucket)
            bucket.objects.all().delete()
            metadata = {'digest': -1}
        
        # upload files to S3 if digests are different 
        if metadata['digest']!=digest:
            s3.upload_file(fpath, args.bucket, fname, ExtraArgs={"Metadata": {"digest": digest}})
        else: 
            _logger.info(f"File {fname} already exists in Bucket {args.bucket} on AWS with same digest. Nothing to do.")

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
