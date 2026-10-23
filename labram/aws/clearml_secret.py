"""Store this machine's ClearML credentials in AWS Secrets Manager for SageMaker jobs.

Jobs receive only the secret's *name* (``LABRAM_CLEARML_SECRET``) and read the
values with their execution role, so credentials never appear in run configs,
S3, the job definition, ClearML or logs.

    python -m labram.aws.clearml_secret put      # create/update from clearml.conf or env
    python -m labram.aws.clearml_secret check    # list stored keys (never values)

After rotating the ClearML key: update ``clearml.conf``, then run ``put`` again.
The execution role needs ``secretsmanager:GetSecretValue`` on the secret.
"""
import argparse
import json
import os
import sys

from labram.utils.secrets import (
    CLEARML_SECRET_KEYS, DEFAULT_CLEARML_SECRET, missing_keys, put_clearml_secret,
)


def _session(profile, region):
    import boto3
    return boto3.Session(profile_name=profile or None, region_name=region or None)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("action", choices=["put", "check"])
    ap.add_argument("--name", default=DEFAULT_CLEARML_SECRET)
    ap.add_argument("--profile", default="")
    ap.add_argument("--region", default="")
    args = ap.parse_args(argv)
    session = _session(args.profile, args.region)
    if args.action == "put":
        from labram.runs.submit_sagemaker import clearml_conf_credentials
        conf = clearml_conf_credentials()
        values = {k: os.environ.get(k) or conf.get(k) for k in CLEARML_SECRET_KEYS}
        missing = missing_keys(values, CLEARML_SECRET_KEYS[:3])
        if missing:
            print(f"Cannot resolve {missing} from the environment or clearml.conf", file=sys.stderr)
            return 1
        arn = put_clearml_secret(args.name, values, session)
        print(f"Stored {sorted(k for k, v in values.items() if v)} in {arn}")
        return 0
    raw = session.client("secretsmanager").get_secret_value(SecretId=args.name)["SecretString"]
    print(f"{args.name}: keys {sorted(json.loads(raw))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
