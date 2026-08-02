#!/usr/bin/env bash
# Fast upload of the TUH Abnormal EEG corpus from EC2 local storage to S3.
#
#   ./upload_tuab_to_s3.sh          # upload, then verify
#   ./upload_tuab_to_s3.sh verify   # verify only (no upload)
#
# ~409k files averaging 0.5 MB, so request concurrency is the bottleneck, not
# bandwidth. s5cmd issues hundreds of parallel requests; `aws s3 sync` (CLI v1)
# issues ~10 and would take many hours longer.
set -euo pipefail

SRC="${SRC:-/data/datasets/EEG-public/TAUB/TUH_Abnormal}"
BUCKET="${BUCKET:-eeg-data-public}"
PREFIX="${PREFIX:-TUH_Abnormal}"
WORKERS="${WORKERS:-256}"
SAMPLE="${SAMPLE:-25}"          # files to checksum-verify
LOG="${LOG:-$HOME/tuab_upload.log}"

SRC="${SRC%/}"
DST="s3://$BUCKET/$PREFIX"

# --- s5cmd -------------------------------------------------------------------
if ! command -v s5cmd >/dev/null; then
  echo "==> installing s5cmd into ~/.local/bin"
  mkdir -p "$HOME/.local/bin"
  curl -sSL https://github.com/peak/s5cmd/releases/download/v2.2.2/s5cmd_2.2.2_Linux-64bit.tar.gz \
    | tar -xz -C "$HOME/.local/bin" s5cmd
fi
export PATH="$HOME/.local/bin:$PATH"

# --- upload ------------------------------------------------------------------
if [[ "${1:-upload}" != "verify" ]]; then
  echo "==> uploading $SRC -> $DST  (${WORKERS} workers)"
  # Trailing /* keeps the relative directory structure under $DST.
  # --if-size-differ makes reruns resumable: matching objects are skipped.
  time s5cmd --numworkers "$WORKERS" --log error \
    cp --if-size-differ "$SRC/*" "$DST/" 2>&1 | tee "$LOG"
fi

# --- verify: object count and total bytes ------------------------------------
echo "==> verifying"

read -r LOCAL_N LOCAL_B < <(
  find "$SRC" -type f -printf '%s\n' | awk '{n++; b+=$1} END {print n+0, b+0}'
)
read -r REMOTE_N REMOTE_B < <(
  aws s3 ls "$DST/" --recursive --summarize \
    | awk '/Total Objects:/ {n=$3} /Total Size:/ {b=$3} END {print n+0, b+0}'
)

printf '  local : %10s files  %18s bytes\n' "$LOCAL_N" "$LOCAL_B"
printf '  s3    : %10s files  %18s bytes\n' "$REMOTE_N" "$REMOTE_B"

if [[ "$LOCAL_N" != "$REMOTE_N" || "$LOCAL_B" != "$REMOTE_B" ]]; then
  echo "FAIL: count/size mismatch — rerun the upload to fill the gaps." >&2
  exit 1
fi

# --- verify: content spot-check ----------------------------------------------
# For single-part uploads the S3 ETag is the object's MD5, so a random sample
# catches silent corruption that matching sizes would miss.
echo "==> checksumming $SAMPLE random files"
fails=0
while read -r f; do
  key="$PREFIX/${f#"$SRC"/}"
  etag=$(aws s3api head-object --bucket "$BUCKET" --key "$key" \
           --query ETag --output text) || { echo "  MISSING $key" >&2; fails=$((fails+1)); continue; }
  etag=${etag//\"/}
  [[ "$etag" == *-* ]] && continue          # multipart: ETag is not an MD5
  md5=$(md5sum "$f" | cut -d' ' -f1)
  if [[ "$etag" != "$md5" ]]; then
    echo "  MISMATCH $key  (local $md5 != s3 $etag)" >&2
    fails=$((fails+1))
  fi
done < <(find "$SRC" -type f | shuf -n "$SAMPLE")

if (( fails > 0 )); then
  echo "FAIL: $fails/$SAMPLE spot-checks failed" >&2
  exit 1
fi

echo "OK: $LOCAL_N files / $LOCAL_B bytes uploaded and verified."
