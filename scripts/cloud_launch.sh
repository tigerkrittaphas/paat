#!/usr/bin/env bash
# Upload the local bundle to a Prime Intellect pod and launch Stage 2 in tmux.
#
# Prerequisites:
#   1. SSH alias `pi-paat` configured in ~/.ssh/config (see docs/cloud-setup.md
#      or the README for the recommended stanza).  Override with PI_HOST.
#   2. Local bundle exists (run scripts/pack_bundle.sh first), OR pass
#      AUTO_PACK=1 to build it as part of this run.
#
# What this script does on the pod:
#   - Unpacks the bundle into /workspace/paat/
#   - Runs scripts/setup_pod.sh (uv sync, CUDA check, data symlink)
#   - Launches Stage 2 (LM pretrain + downstream eval) inside tmux session
#     `paat-exp2` with the chosen MODELS + PRESET
#   - Returns immediately so you can ssh in to monitor or detach/reconnect
#
# Usage:
#   bash scripts/cloud_launch.sh                           # 6-model long preset
#   PRESET=standard bash scripts/cloud_launch.sh           # 1B tokens (cheaper)
#   MODELS="parity_bpe paat_a100_l0" bash scripts/cloud_launch.sh   # subset only
#   AUTO_PACK=1 bash scripts/cloud_launch.sh               # build bundle first
#   DRY_RUN=1 bash scripts/cloud_launch.sh                 # plan only, no upload
#
# Cost estimates (A100-80GB at ~$1.80/hr):
#   PRESET=standard (1B tok)  × 6 models  ≈ ~$15  ~9h
#   PRESET=long     (3B tok)  × 6 models  ≈ ~$45  ~25h
#   custom (5B tok) × 6 models             ≈ ~$75  ~40h

set -euo pipefail

# ── Config ─────────────────────────────────────────────────────────────────
PI_HOST="${PI_HOST:-pi-paat}"
REMOTE_ROOT="${REMOTE_ROOT:-/workspace}"
REMOTE_REPO="${REMOTE_REPO:-/workspace/paat}"
BUNDLE_PATH="${BUNDLE_PATH:-paat_bundle.tar.zst}"

# Persistent-volume mode — set USE_VOLUME=1 to skip the bundle upload and
# instead pull data + tokenizers from a pre-populated PI persistent volume
# mounted at $VOLUME_MOUNT on the pod.  Run scripts/init_data_volume.sh
# first to populate the volume.  After init, every subsequent cloud run
# only ships the source code (a few MB) instead of the ~30 GB bundle.
USE_VOLUME="${USE_VOLUME:-0}"
VOLUME_MOUNT="${VOLUME_MOUNT:-/data}"

# Stage 2 knobs (override via env).
PRESET="${PRESET:-long}"                      # smoke | pilot | standard | long
MODELS="${MODELS:-bpe parity_bpe adat unigram paat_a10 paat_a100_l0}"
TMUX_SESSION="${TMUX_SESSION:-paat-exp2}"

AUTO_PACK="${AUTO_PACK:-0}"
DRY_RUN="${DRY_RUN:-0}"

# ── Banner ─────────────────────────────────────────────────────────────────
if [[ "$USE_VOLUME" == "1" ]]; then
    transport_line="  data source:   persistent volume at $VOLUME_MOUNT (USE_VOLUME=1)"
else
    transport_line="  bundle:        $BUNDLE_PATH"
fi
cat <<EOF

================================================================
  PAAT — Cloud launch (Stage 2)
  pod host:      $PI_HOST
  remote repo:   $REMOTE_REPO
$transport_line
  preset:        $PRESET
  models:        $MODELS
  tmux session:  $TMUX_SESSION
================================================================
EOF

err() { echo "[FATAL] $*" >&2; exit 1; }
say() { echo ""; echo "▶ $*"; }

# ── Preflight ──────────────────────────────────────────────────────────────
say "Preflight"

# 1. Bundle (only required for the bundle path).
if [[ "$USE_VOLUME" != "1" ]]; then
    if [[ ! -f "$BUNDLE_PATH" ]]; then
        if [[ "$AUTO_PACK" == "1" ]]; then
            say "Bundle not found — building (AUTO_PACK=1)"
            bash scripts/pack_bundle.sh
        else
            err "Bundle not found at $BUNDLE_PATH.  Run scripts/pack_bundle.sh first or pass AUTO_PACK=1."
        fi
    fi
    bundle_size=$(du -h "$BUNDLE_PATH" | awk '{print $1}')
    echo "  bundle:       $BUNDLE_PATH ($bundle_size)"
else
    echo "  bundle:       skipped (USE_VOLUME=1)"
fi

# 2. SSH reachability.
if ! ssh -o ConnectTimeout=10 -o BatchMode=yes "$PI_HOST" 'echo ok' >/dev/null 2>&1; then
    err "Cannot SSH to '$PI_HOST'.  Check ~/.ssh/config has the pod entry, or set PI_HOST."
fi
echo "  ssh:          $PI_HOST reachable"

# 3. Volume mount (only required for USE_VOLUME mode).
if [[ "$USE_VOLUME" == "1" ]]; then
    vol_check=$(ssh "$PI_HOST" "test -d $VOLUME_MOUNT && find $VOLUME_MOUNT -maxdepth 1 -type d 2>/dev/null | wc -l")
    if [[ "$vol_check" -lt 4 ]]; then
        err "$VOLUME_MOUNT on $PI_HOST is missing or unpopulated (found <4 dirs).  Run scripts/init_data_volume.sh first."
    fi
    echo "  volume:       $VOLUME_MOUNT populated on pod ($vol_check dirs)"
fi

# 3. Cost-tier confirmation (only nag once per session).
case "$PRESET" in
    smoke)    est_cost="< \$1   (~10 min)" ;;
    standard) est_cost="~\$15  (~9 h)" ;;
    long)     est_cost="~\$45  (~25 h)" ;;
    *)        est_cost="unknown" ;;
esac
echo "  est. cost:    $est_cost (A100-80GB at ~\$1.80/hr × ${MODELS} models)"

if [[ "$DRY_RUN" == "1" ]]; then
    echo ""
    echo "[DRY_RUN=1] would upload + launch.  Stopping here."
    exit 0
fi

# ── Upload (bundle OR code-only) ──────────────────────────────────────────
if [[ "$USE_VOLUME" == "1" ]]; then
    say "Syncing source code only (data/tokenizers come from volume)"

    # rsync only the things that change between runs.  Excludes everything
    # that's either on the volume, ephemeral, or never wanted on the pod.
    ssh "$PI_HOST" "mkdir -p '$REMOTE_REPO'"
    rsync -avP --delete \
        --exclude='.git/' \
        --exclude='.venv/' \
        --exclude='__pycache__/' \
        --exclude='*.pyc' \
        --exclude='data/' \
        --exclude='models/' \
        --exclude='results/' \
        --exclude='logs/' \
        --exclude='paat_bundle.tar.zst*' \
        ./ "$PI_HOST:$REMOTE_REPO/"
else
    say "Uploading bundle (rsync --partial — resumes on disconnect)"

    # The .sha256 is tiny; tag it along so we can verify on the pod.
    sha_path="${BUNDLE_PATH}.sha256"
    if [[ ! -f "$sha_path" ]]; then
        sha256sum "$BUNDLE_PATH" > "$sha_path"
    fi

    rsync -avP --partial \
        "$BUNDLE_PATH" "$sha_path" \
        "${PI_HOST}:${REMOTE_ROOT}/"
fi

# ── Setup + launch on the pod ──────────────────────────────────────────────
say "Setup + launch on pod"

# We dispatch one of two slightly different setup scripts depending on
# whether data is in a bundle (extract) or a volume (symlink).  The launch
# tail (uv sync, tmux, etc.) is identical.

if [[ "$USE_VOLUME" == "1" ]]; then
    # Volume mode: code is already rsynced in place at $REMOTE_REPO; we just
    # need to wire up the symlinks from the volume into the repo so existing
    # scripts find data/tokenizers at their default paths.
    ssh "$PI_HOST" bash -se <<EOF
set -euo pipefail

cd "$REMOTE_REPO"

# Verify the volume is populated (in case the user destroyed init-data state).
for d in mc4_sampled flores tokenizers; do
    [[ -d "$VOLUME_MOUNT/\$d" ]] || {
        echo "[FATAL] $VOLUME_MOUNT/\$d missing — re-run scripts/init_data_volume.sh" >&2
        exit 1
    }
done

# Symlink data + tokenizers from the volume into the repo at the paths
# the rest of the codebase already expects.  -sf is idempotent.
echo "▶ Wiring volume → repo symlinks ..."
mkdir -p data/raw models results
ln -sfn "$VOLUME_MOUNT/mc4_sampled" data/raw/mc4_sampled
ln -sfn "$VOLUME_MOUNT/mc4_sampled" data/raw/mc4
ln -sfn "$VOLUME_MOUNT/flores"      data/raw/flores
ln -sfn "$VOLUME_MOUNT/tokenizers"  models/tokenizers
[[ -d "$VOLUME_MOUNT/parity" ]] && ln -sfn "$VOLUME_MOUNT/parity" results/parity

ls -ld data/raw/mc4 data/raw/flores models/tokenizers 2>/dev/null \
    | awk '{print "  " \$0}'

# Setup deps + CUDA.  Pass no bundle path — there's nothing to unpack.
echo "▶ Running setup_pod.sh (uv sync + CUDA check) ..."
bash scripts/setup_pod.sh

# Launch Stage 2.
if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    echo "[FATAL] tmux session '$TMUX_SESSION' already exists — kill it first" >&2
    echo "        ssh $PI_HOST 'tmux kill-session -t $TMUX_SESSION'" >&2
    exit 1
fi

mkdir -p logs/exp_2
tmux new-session -d -s "$TMUX_SESSION" \
    "MODELS='$MODELS' PRESET='$PRESET' \
     bash scripts/exp_2_pretrain_eval.sh 2>&1 | tee logs/exp_2/cloud_run.log; \
     echo ''; echo '═══ Stage 2 finished — sleeping so you can read the output ═══'; \
     sleep 3600"

echo ""
echo "▶ Launched Stage 2 in tmux session '$TMUX_SESSION' (volume-backed data)."
tmux ls
EOF
else
    ssh "$PI_HOST" bash -se <<EOF
set -euo pipefail

cd "$REMOTE_ROOT"

# Verify upload integrity before doing anything else.
echo "▶ Verifying bundle checksum ..."
sha256sum -c "$(basename "$sha_path")"

# Fresh extract (idempotent — overwrites old code/data).
echo "▶ Extracting bundle ..."
mkdir -p "$REMOTE_REPO"
tar --use-compress-program=unzstd -xf "$(basename "$BUNDLE_PATH")" -C "$REMOTE_REPO"

cd "$REMOTE_REPO"

# Run setup (installs uv + deps, CUDA check).  Pass empty bundle path so
# setup_pod.sh skips the unpack — we've already extracted.
echo "▶ Running setup_pod.sh ..."
bash scripts/setup_pod.sh

# Make sure data/raw/mc4 symlink exists (setup_pod.sh skips this when
# called without a bundle path).
if [[ ! -e data/raw/mc4 && -d data/raw/mc4_sampled ]]; then
    ln -sf mc4_sampled data/raw/mc4
    echo "▶ symlinked data/raw/mc4 -> mc4_sampled"
fi

# Launch Stage 2 in a detached tmux session.  If the session already
# exists, complain instead of clobbering it.
if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    echo "[FATAL] tmux session '$TMUX_SESSION' already exists — kill it first" >&2
    echo "        ssh $PI_HOST 'tmux kill-session -t $TMUX_SESSION'" >&2
    exit 1
fi

mkdir -p logs/exp_2

tmux new-session -d -s "$TMUX_SESSION" \
    "MODELS='$MODELS' PRESET='$PRESET' \
     bash scripts/exp_2_pretrain_eval.sh 2>&1 | tee logs/exp_2/cloud_run.log; \
     echo ''; echo '═══ Stage 2 finished — sleeping so you can read the output ═══'; \
     sleep 3600"

echo ""
echo "▶ Launched Stage 2 in tmux session '$TMUX_SESSION'."
echo ""
tmux ls
EOF
fi

# ── Done ───────────────────────────────────────────────────────────────────
cat <<EOF

================================================================
  Launched.
================================================================

Monitor:
  ssh $PI_HOST tmux attach -t $TMUX_SESSION
  # detach without killing: Ctrl-b d

Live tail of the log only:
  ssh $PI_HOST tail -f $REMOTE_REPO/logs/exp_2/cloud_run.log

When it's done — pull results back:
  rsync -avP --exclude='checkpoint-*' \\
      $PI_HOST:$REMOTE_REPO/models/lm/ ./models/lm_pi/
  rsync -avP $PI_HOST:$REMOTE_REPO/results/ ./results_pi/
  rsync -avP $PI_HOST:$REMOTE_REPO/logs/  ./logs_pi/

Then DESTROY the pod from the PI dashboard — it bills by the second.
EOF
