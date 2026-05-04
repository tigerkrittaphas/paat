#!/usr/bin/env bash
# One-time: upload data + tokenizers + parity reports to a Prime Intellect
# persistent volume so future cloud runs don't have to re-upload them.
#
# Prerequisites:
#   1. Created a persistent volume in the PI dashboard.
#   2. Spun up any pod (CPU is fine; cheaper) with that volume attached at
#      $VOLUME_MOUNT (default /workspace).  The mount path is set when
#      you create the pod, not when you create the volume.
#   3. SSH alias `pi-paat` (or set PI_HOST) configured in ~/.ssh/config.
#
# Two upload modes:
#   - Default (BUNDLE_MODE=1): pack everything locally with zstd, upload one
#     ~10 GB compressed file, extract on the pod.  Fastest for the initial
#     upload — JSONL compresses ~3:1 (~30 GB raw → ~10 GB on the wire).
#   - BUNDLE_MODE=0 (raw rsync): per-component rsync of uncompressed files.
#     Slower for the initial upload but right for incremental refreshes
#     (only changed files transfer).
#
# After this completes:
#   - Destroy the upload pod.  Volume + data persist.
#   - Future cloud runs use:  USE_VOLUME=1 bash scripts/cloud_launch.sh
#
# Usage:
#   bash scripts/init_data_volume.sh                        # default: bundle mode
#   BUNDLE_MODE=0 bash scripts/init_data_volume.sh          # rsync mode
#   PI_HOST=pi-paat-cpu bash scripts/init_data_volume.sh    # different SSH alias
#   VOLUME_MOUNT=/data    bash scripts/init_data_volume.sh
#   KEEP_LOCAL_BUNDLE=1   bash scripts/init_data_volume.sh  # don't delete bundle after upload
#   DRY_RUN=1            bash scripts/init_data_volume.sh   # plan only

set -euo pipefail

PI_HOST="${PI_HOST:-pi-paat}"
VOLUME_MOUNT="${VOLUME_MOUNT:-/data}"
BUNDLE_MODE="${BUNDLE_MODE:-1}"
LOCAL_BUNDLE="${LOCAL_BUNDLE:-paat_bundle.tar.zst}"
KEEP_LOCAL_BUNDLE="${KEEP_LOCAL_BUNDLE:-0}"

DRY_RUN="${DRY_RUN:-0}"

err() { echo "[FATAL] $*" >&2; exit 1; }
say() { echo ""; echo "▶ $*"; }

mode_label="bundle (zstd-compressed)"
[[ "$BUNDLE_MODE" == "0" ]] && mode_label="raw rsync (per-component, incremental-friendly)"

cat <<EOF

================================================================
  Init persistent data volume (one-time)
  pod host:      $PI_HOST
  volume mount:  $VOLUME_MOUNT
  upload mode:   $mode_label
================================================================
EOF

# ── Preflight ──────────────────────────────────────────────────────────────
say "Preflight"

# Local artifacts must exist.
[[ -d data/raw/mc4_sampled ]] \
    || err "data/raw/mc4_sampled missing — run scripts/sample_mc4.py first"
[[ -d data/raw/flores ]]      \
    || err "data/raw/flores missing — run scripts/download_flores.py first"
[[ -d models/tokenizers ]]    \
    || err "models/tokenizers missing — run scripts/exp_1_tokenizers.sh first"

# zstd needed for bundle mode.
if [[ "$BUNDLE_MODE" == "1" ]]; then
    command -v zstd >/dev/null 2>&1 \
        || err "zstd not installed — apt-get install zstd, or set BUNDLE_MODE=0"
fi

# SSH reachability.
ssh -o ConnectTimeout=10 -o BatchMode=yes "$PI_HOST" 'echo ok' >/dev/null 2>&1 \
    || err "Cannot SSH to '$PI_HOST'"

# Volume must already be mounted on the pod side (PI mounts it at pod creation).
mount_check=$(ssh "$PI_HOST" "test -d $VOLUME_MOUNT && echo OK || echo MISSING")
if [[ "$mount_check" != "OK" ]]; then
    err "$VOLUME_MOUNT not present on $PI_HOST.  Did you attach the volume when creating the pod?"
fi

# Local size summary.
echo "  local data:    $(du -sh data/raw/mc4_sampled 2>/dev/null | awk '{print $1}') mC4"
echo "                 $(du -sh data/raw/flores 2>/dev/null | awk '{print $1}') FLORES"
echo "                 $(du -sh models/tokenizers 2>/dev/null | awk '{print $1}') tokenizers"
[[ -d results/parity ]] && echo "                 $(du -sh results/parity 2>/dev/null | awk '{print $1}') parity"

# Bundle-mode disk preflight: need ~1/3 of raw size for the compressed bundle.
if [[ "$BUNDLE_MODE" == "1" ]]; then
    raw_kb=$(du -sk data/raw/mc4_sampled data/raw/flores models/tokenizers 2>/dev/null | awk '{s+=$1} END {print s}')
    [[ -d results/parity ]] && raw_kb=$((raw_kb + $(du -sk results/parity | awk '{print $1}')))
    avail_kb=$(df -P . | tail -1 | awk '{print $4}')
    need_kb=$((raw_kb / 3 + 1024 * 1024))    # bundle ≈ 1/3 raw + 1 GB safety
    if (( avail_kb < need_kb )); then
        echo "[WARN] Local disk has $((avail_kb / 1024 / 1024)) GB free,"
        echo "       bundle will need ~$((need_kb / 1024 / 1024)) GB.  Continuing anyway."
    fi
fi

if [[ "$DRY_RUN" == "1" ]]; then
    echo ""
    echo "[DRY_RUN=1] would $mode_label upload to ${PI_HOST}:${VOLUME_MOUNT}/.  Stopping."
    exit 0
fi

# ── Ensure target dirs exist on the volume ─────────────────────────────────
say "Preparing volume layout on pod"
ssh "$PI_HOST" "mkdir -p '$VOLUME_MOUNT'/{mc4_sampled,flores,tokenizers,parity}"

# ── Upload ─────────────────────────────────────────────────────────────────
if [[ "$BUNDLE_MODE" == "1" ]]; then
    # ── Bundle mode ────────────────────────────────────────────────────────
    # Reuse an existing bundle if one is already on disk and intact.  This
    # makes retries (e.g. after a flaky upload) and re-runs against a fresh
    # pod nearly instant — no 5–10 min repack.  Set FORCE_REPACK=1 to
    # rebuild even if the bundle exists (e.g. after re-sampling mC4).
    if [[ -f "$LOCAL_BUNDLE" && "${FORCE_REPACK:-0}" != "1" ]]; then
        say "Reusing existing bundle: $LOCAL_BUNDLE"
        bundle_size=$(du -h "$LOCAL_BUNDLE" | awk '{print $1}')
        echo "  bundle:    $LOCAL_BUNDLE ($bundle_size)"
        # Refresh checksum if missing or older than the bundle.
        if [[ ! -f "${LOCAL_BUNDLE}.sha256" || "$LOCAL_BUNDLE" -nt "${LOCAL_BUNDLE}.sha256" ]]; then
            echo "  computing checksum ..."
            sha256sum "$LOCAL_BUNDLE" > "${LOCAL_BUNDLE}.sha256"
        fi
        echo "  → set FORCE_REPACK=1 to rebuild from current source files"
    else
        say "Building data bundle (zstd -19) → $LOCAL_BUNDLE"

        # Use -C to flatten the layout: contents of data/raw/mc4_sampled land
        # at "mc4_sampled/" inside the tarball, NOT at "data/raw/mc4_sampled/".
        # That way extraction into $VOLUME_MOUNT lands files at the right paths.
        tar_args=(
            --owner=0 --group=0
            --exclude='__pycache__' --exclude='*.pyc' --exclude='.DS_Store'
            -cf -
            -C data/raw mc4_sampled flores
            -C "$(pwd)/models" tokenizers
        )
        if [[ -d results/parity ]]; then
            tar_args+=(-C "$(pwd)/results" parity)
        fi

        tar "${tar_args[@]}" | zstd -T0 -19 -o "$LOCAL_BUNDLE" --force

        sha256sum "$LOCAL_BUNDLE" > "${LOCAL_BUNDLE}.sha256"
        bundle_size=$(du -h "$LOCAL_BUNDLE" | awk '{print $1}')
        echo "  bundle:    $LOCAL_BUNDLE ($bundle_size)"
    fi

    say "Uploading bundle (rsync --partial — resumes on disconnect)"
    rsync -avP --partial \
        "$LOCAL_BUNDLE" "${LOCAL_BUNDLE}.sha256" \
        "${PI_HOST}:${VOLUME_MOUNT}/"

    say "Extracting bundle into volume"
    ssh "$PI_HOST" bash -se <<EOF
set -euo pipefail
cd "$VOLUME_MOUNT"

echo "▶ Verifying bundle checksum ..."
sha256sum -c "$(basename "${LOCAL_BUNDLE}.sha256")"

echo "▶ Extracting (this writes mc4_sampled/, flores/, tokenizers/, parity/) ..."
tar --use-compress-program=unzstd -xf "$(basename "$LOCAL_BUNDLE")"

echo "▶ Removing on-pod bundle file (data is now in place) ..."
rm -f "$(basename "$LOCAL_BUNDLE")" "$(basename "${LOCAL_BUNDLE}.sha256")"
EOF

    if [[ "$KEEP_LOCAL_BUNDLE" != "1" ]]; then
        echo ""
        echo "▶ Removing local bundle ($LOCAL_BUNDLE).  Set KEEP_LOCAL_BUNDLE=1 to retain."
        rm -f "$LOCAL_BUNDLE" "${LOCAL_BUNDLE}.sha256"
    fi
else
    # ── Raw rsync mode (incremental-friendly) ──────────────────────────────
    say "Per-component rsync (uncompressed, incremental)"
    rsync -avP --partial --delete \
        data/raw/mc4_sampled/   "$PI_HOST:$VOLUME_MOUNT/mc4_sampled/"
    rsync -avP --partial --delete \
        data/raw/flores/        "$PI_HOST:$VOLUME_MOUNT/flores/"
    rsync -avP --partial --delete \
        models/tokenizers/      "$PI_HOST:$VOLUME_MOUNT/tokenizers/"
    if [[ -d results/parity ]]; then
        rsync -avP --partial --delete \
            results/parity/     "$PI_HOST:$VOLUME_MOUNT/parity/"
    fi
fi

# ── Verify ─────────────────────────────────────────────────────────────────
say "Verifying upload"
ssh "$PI_HOST" bash -se <<EOF
set -euo pipefail
cd "$VOLUME_MOUNT"
echo "  mC4 langs:        \$(find mc4_sampled -maxdepth 1 -name '*.jsonl' | wc -l)"
echo "  FLORES langs:     \$(find flores      -maxdepth 1 -name '*.jsonl' | wc -l)"
echo "  tokenizer.jsons:  \$(find tokenizers  -name 'tokenizer.json'      | wc -l)"
echo "  parity reports:   \$(find parity      -maxdepth 1 -name '*.json'  2>/dev/null | wc -l)"
echo "  total size:       \$(du -sh . | awk '{print \$1}')"
EOF

cat <<EOF

================================================================
  Volume initialised.

  Now you can:
    1. Destroy this upload pod (PI dashboard) — data persists on the volume.
    2. Spin up future GPU pods with the SAME volume attached at $VOLUME_MOUNT.
    3. For each cloud run:
         USE_VOLUME=1 bash scripts/cloud_launch.sh
       This will rsync only the source code (a few MB) instead of
       re-uploading the ~30 GB of data.

  To refresh data later (e.g. after re-sampling mC4 or training new
  tokenizers), use the incremental mode so only changed files transfer:
       BUNDLE_MODE=0 bash scripts/init_data_volume.sh
================================================================
EOF
