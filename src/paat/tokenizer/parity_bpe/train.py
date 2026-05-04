"""
Parity-aware BPE training wrapper for PAAT.

Provides a :func:`train_parity_bpe` function with the same shape as
:func:`paat.tokenizer.train.train_bpe`, so the parity-aware BPE algorithm
(Foroutan et al., 2025) can be invoked the same way as the classic BPE
baseline: read mC4 JSONL files from ``data_dir``, write a single
``tokenizer.json`` to ``output_dir`` that ``eval_parity.py`` can load.

The upstream algorithm requires:
    * one input corpus per language,
    * a multi-parallel development set per language (we use FLORES+),
      OR a desired compression-ratio vector.

This wrapper streams the per-language mC4 JSONL files into the upstream
learner without materialising plain-text copies on disk, and converts
the resulting merges file into a HuggingFace BPE ``tokenizer.json``.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Callable, Iterable

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel, Whitespace
from tokenizers import pre_tokenizers

from paat.tokenizer.parity_bpe import parity_aware_learn_bpe as _upstream
from paat.tokenizer.parity_bpe.hf_tokenizer import build_vocab_from_merges


SPECIAL_TOKENS = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]


_PARALLEL_PATCH_INSTALLED = False


def _install_parallel_preprocess() -> None:
    """Replace ``_upstream.preprocess_input_data`` with a thread-pooled version.

    Idempotent — re-installing is a no-op.  The replacement is functionally
    identical to the original; it only swaps the serial loops over input/dev
    files for parallel ``ThreadPoolExecutor.map`` calls.  Safe because
    ``get_vocabulary`` does its work in Rust (HF tokenizers' ``pre_tokenize_str``)
    which releases the GIL.
    """
    global _PARALLEL_PATCH_INSTALLED
    if _PARALLEL_PATCH_INSTALLED:
        return

    import copy
    import functools
    import os
    from collections import defaultdict
    from concurrent.futures import ThreadPoolExecutor

    import numpy

    def parallel_preprocess_input_data(
        infiles, devfiles,
        is_dict=False, total_symbols=False, num_global=0,
        num_workers=1, bpe_file=None,
    ):
        bpe_codes = None
        if bpe_file is not None:
            line = bpe_file.readline()
            offset = 1
            if not line.startswith("version"):
                bpe_file.seek(0)
                offset = 0
            raw = [tuple(item.strip("\r\n ").split(" "))
                   for item in bpe_file.read().rstrip("\n").split("\n")]
            for i, item in enumerate(raw):
                if len(item) != 2:
                    raise ValueError(
                        f"Bad BPE codes line {i + offset}: {' '.join(item)}"
                    )
            bpe_codes = dict(
                (code, i) for (i, code) in reversed(list(enumerate(raw)))
            )

        def _build(f):
            v = _upstream.get_vocabulary(f, is_dict, num_workers=1)
            if bpe_codes is not None:
                v = _upstream.pre_merge(v, bpe_codes)
            return dict(((tuple(x,), y) for (x, y) in v.items()))

        # Pool size: bounded by both file count and CPU count.  CPU-bound
        # tasks (`pre_tokenize_str`) inside HF Rust release the GIL, so
        # threads scale near-linearly with cores up to ~16.
        n_workers = min(len(infiles) or 1, (os.cpu_count() or 4) * 2)
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            print(f"  [pbpe] preprocess inputs: {len(infiles)} files × "
                  f"{n_workers} threads ...", flush=True)
            vocabs = list(ex.map(_build, infiles))
            dev_vocabs = []
            if devfiles:
                print(f"  [pbpe] preprocess devs:   {len(devfiles)} files × "
                      f"{n_workers} threads ...", flush=True)
                dev_vocabs = list(ex.map(_build, devfiles))

        joint_keys = set().union(*(v.keys() for v in vocabs))
        dev_keys = set().union(*(v.keys() for v in dev_vocabs)) if dev_vocabs else set()

        array_length = len(vocabs) + (1 if num_global else 0)

        vocab = defaultdict(lambda: numpy.zeros(array_length, dtype=int))
        for i, v in enumerate(vocabs):
            for key in joint_keys:
                vocab[key][i] = v.get(key, 0)
        if num_global:
            for key in joint_keys:
                vocab[key][-1] = sum(vocab[key])

        dev_vocab = defaultdict(lambda: numpy.zeros(len(dev_vocabs), dtype=int))
        if dev_vocabs:
            for i, v in enumerate(dev_vocabs):
                for key in dev_keys:
                    dev_vocab[key][i] = v.get(key, 0)

        sorted_vocab = sorted(vocab.items(), key=lambda x: sum(x[1]), reverse=True)
        stats, indices = _upstream.get_pair_statistics(sorted_vocab)
        big_stats = copy.deepcopy(stats)

        if total_symbols:
            uniq_internal = set()
            uniq_final = set()
            for word in vocab:
                for char in word[:-1]:
                    uniq_internal.add(char)
                uniq_final.add(word[-1])

        threshold = numpy.zeros(array_length, dtype=int)
        for l in range(array_length):
            threshold[l] = stats[max(stats, key=lambda x: (stats[x][l], x))][l] / 10

        if dev_vocab:
            lengths = functools.reduce(
                numpy.add,
                [len(key) * value for key, value in dev_vocab.items()],
            )
        else:
            lengths = numpy.zeros(len(dev_vocabs) if dev_vocabs else 0, dtype=int)

        return (dev_vocab, sorted_vocab, stats, indices, big_stats,
                threshold, lengths, array_length)

    _upstream.preprocess_input_data = parallel_preprocess_input_data
    _PARALLEL_PATCH_INSTALLED = True


class _JsonlTextStream:
    """Adapts a ``<lang>.jsonl`` file to the file-like interface the
    upstream learner expects.

    Yields decoded text strings line-by-line so the learner sees one
    document per line. Exposes ``.name`` (any non-``<stdin>`` value) so
    the single-worker code path is taken in
    :func:`parity_aware_learn_bpe.get_vocabulary`.
    """

    def __init__(
        self,
        path: Path,
        key: str = "text",
        record_filter: Callable[[dict], bool] | None = None,
        max_docs: int | None = None,
    ):
        self.path = Path(path)
        self.name = str(self.path)
        self._key = key
        self._filter = record_filter
        self._max_docs = max_docs
        self._fh = None

    def __iter__(self):
        self._fh = self.path.open(encoding="utf-8")
        n = 0
        for line in self._fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if self._filter is not None and not self._filter(rec):
                continue
            yield rec[self._key]
            n += 1
            if self._max_docs is not None and n >= self._max_docs:
                break

    def close(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def _resolve_languages(data_dir: Path, languages: list[str] | None) -> list[str]:
    available = sorted(p.stem for p in data_dir.glob("*.jsonl"))
    if not available:
        raise FileNotFoundError(f"No JSONL files found in {data_dir}")
    if languages is None:
        return available
    missing = [l for l in languages if l not in available]
    if missing:
        raise FileNotFoundError(
            f"No mC4 file for languages: {missing} in {data_dir}"
        )
    return list(languages)


def _merges_to_tokenizer_json(merges_path: Path, output_dir: Path) -> Tokenizer:
    """Convert a parity-aware-BPE merges file into a HF ``tokenizer.json``.

    Uses the same vocab-construction logic as the upstream
    :func:`hf_tokenizer.build_vocab_from_merges` (256-entry ByteLevel
    alphabet + one entry per merge), then wraps it in a ``Tokenizer`` with
    matching pre-tokenizer/decoder so encoding round-trips correctly.
    """
    with merges_path.open("r", encoding="utf-8") as f:
        merge_lines = f.readlines()

    vocab = build_vocab_from_merges(merge_lines)

    output_dir.mkdir(parents=True, exist_ok=True)
    vocab_file = output_dir / "vocab.json"
    merges_file = output_dir / "merges.txt"
    with vocab_file.open("w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    with merges_file.open("w", encoding="utf-8") as f:
        for line in merge_lines:
            f.write(line)

    tokenizer = Tokenizer(BPE(vocab=str(vocab_file), merges=str(merges_file)))
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [Whitespace(), ByteLevel(use_regex=False)]
    )
    tokenizer.decoder = ByteLevelDecoder()
    tokenizer.add_special_tokens(SPECIAL_TOKENS)

    out_path = output_dir / "tokenizer.json"
    tokenizer.save(str(out_path))
    print(
        f"Tokenizer saved to {out_path}  "
        f"(vocab size: {tokenizer.get_vocab_size():,})"
    )
    return tokenizer


def train_parity_bpe(
    data_dir: Path,
    output_dir: Path,
    vocab_size: int = 32_000,
    min_frequency: int = 2,
    languages: list[str] | None = None,
    flores_dir: Path | None = Path("data/raw/flores"),
    variant: str = "base",
    global_merges: int = 0,
    window_size: int = 100,
    alpha: int = 2,
    ratio: list[float] | None = None,
    verbose: bool = False,
    total_docs: int | None = None,
) -> Tokenizer:
    """Train a parity-aware BPE tokenizer and save it to *output_dir*.

    Mirrors the signature of :func:`paat.tokenizer.train.train_bpe`.
    Reads one ``<lang>.jsonl`` per language from *data_dir* and uses
    FLORES+ sentences from *flores_dir* as the multi-parallel
    development set for parity computation (or *ratio* per language).

    Args:
        data_dir:      Directory with per-language ``<lang>.jsonl`` mC4
                       files (one document per line, ``{"text": ...}``).
        output_dir:    Where to write ``tokenizer.json`` (created if absent).
        vocab_size:    Target vocabulary size (incl. 256 ByteLevel base
                       chars and special tokens). Number of BPE merges
                       performed is ``vocab_size - 256 - len(SPECIAL_TOKENS)``.
        min_frequency: Minimum merge-pair frequency.
        languages:     Subset of language codes to train on (default: all
                       ``*.jsonl`` files in *data_dir*).
        flores_dir:    Directory with per-language FLORES+ JSONL files
                       used as the parity dev set. Only the ``dev`` split
                       is read; the ``devtest`` split is held back for
                       eval. Required unless *ratio* is given.
        variant:       ``"base"`` or ``"window"`` — the moving-window
                       balancing variant from Foroutan et al. 2025.
        global_merges: Run this many initial merges on global statistics
                       (standard-BPE-style) before switching to
                       parity-driven merging (Hybrid parity-aware BPE).
        window_size:   Window size for the ``"window"`` variant.
        alpha:         Window threshold ratio for the ``"window"`` variant.
        ratio:         Per-language target compression ratios. If given,
                       *flores_dir* is ignored. Length must equal the
                       number of languages.
        verbose:       Verbose progress output from the learner.

    Returns:
        The trained :class:`~tokenizers.Tokenizer` object.
    """
    if variant not in ("base", "window"):
        raise ValueError(f"variant must be 'base' or 'window', got {variant!r}")

    langs = _resolve_languages(data_dir, languages)
    print(f"Languages ({len(langs)}): {langs}")

    # Per-language doc caps proportional to MC4_NATURAL_COUNTS — without
    # this the upstream learner reads entire mC4 language files (gigabytes
    # each) and is unusable for both local dev and the cloud sample.
    docs_per_lang: dict[str, int] | None = None
    if total_docs is not None:
        from paat.data.languages import MC4_NATURAL_COUNTS
        total_natural = sum(MC4_NATURAL_COUNTS.get(l, 1) for l in langs)
        docs_per_lang = {
            l: max(1, round(total_docs * MC4_NATURAL_COUNTS.get(l, 1) / total_natural))
            for l in langs
        }
        print(f"Doc cap: {sum(docs_per_lang.values()):,} total "
              f"(proportional to MC4_NATURAL_COUNTS)")

    inputs: list[_JsonlTextStream] = [
        _JsonlTextStream(
            data_dir / f"{l}.jsonl", key="text",
            max_docs=docs_per_lang.get(l) if docs_per_lang else None,
        )
        for l in langs
    ]

    devs: list[_JsonlTextStream] | None = None
    if ratio is None:
        if flores_dir is None:
            raise ValueError(
                "Parity-aware BPE requires either flores_dir (dev sets) "
                "or ratio (per-language compression targets)."
            )
        missing = [l for l in langs if not (flores_dir / f"{l}.jsonl").exists()]
        if missing:
            raise FileNotFoundError(
                f"FLORES+ files missing for languages: {missing} in {flores_dir}. "
                "Run scripts/download_flores.py first."
            )
        # Use only the FLORES `dev` split for parity computation so the
        # `devtest` split is held out for eval (see
        # :func:`paat.parity.metrics.load_flores_sentences`). This avoids
        # train/test leakage when `eval_parity.py` measures compression
        # on the same sentences the learner optimized against.
        devs = [
            _JsonlTextStream(
                flores_dir / f"{l}.jsonl",
                key="sentence",
                record_filter=lambda r: r.get("split") == "dev",
            )
            for l in langs
        ]

    if ratio is not None and len(ratio) != len(langs):
        raise ValueError(
            f"ratio length ({len(ratio)}) must equal number of languages ({len(langs)})"
        )

    # ByteLevel base alphabet (256) + special tokens are added on top of
    # the merges. Convert vocab_size to merge count.
    base_alphabet = 256
    num_symbols = vocab_size - base_alphabet - len(SPECIAL_TOKENS)
    if num_symbols < 1:
        raise ValueError(
            f"vocab_size={vocab_size} is too small "
            f"(need > {base_alphabet + len(SPECIAL_TOKENS)})"
        )

    # The upstream module references `pre_tokenizer` as a module-level
    # global (defined only in its __main__ block). Set it here so the
    # learner can call `pre_tokenizer.pre_tokenize_str(line)`.
    _upstream.pre_tokenizer = pre_tokenizers.Sequence(
        [Whitespace(), ByteLevel(use_regex=False)]
    )

    # Monkey-patch upstream `preprocess_input_data` to parallelise the
    # per-language `get_vocabulary` calls.  Without this, all 96 input +
    # 96 dev files are processed sequentially in Python, which can take
    # 30–60 minutes before the merge loop even starts on a 10M-doc corpus.
    # `get_vocabulary` calls `pre_tokenize_str` (Rust, releases GIL), so a
    # plain ThreadPoolExecutor gets near-linear speedup over the file list.
    _install_parallel_preprocess()

    import numpy as np
    ratio_arr = None
    if ratio is not None:
        ratio_arr = np.array(ratio, dtype=float)
        ratio_arr = ratio_arr / ratio_arr[0]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".merges", delete=False, encoding="utf-8"
    ) as merges_fh:
        merges_path = Path(merges_fh.name)
        try:
            if variant == "base":
                _upstream.learn_bpe(
                    inputs,
                    merges_fh,
                    devs,
                    num_symbols,
                    min_frequency=min_frequency,
                    verbose=verbose,
                    is_dict=False,
                    total_symbols=False,
                    num_global=global_merges,
                    ratio=ratio_arr,
                    num_workers=1,
                    bpe_file=None,
                )
            else:
                _upstream.learn_bpe_moving_window(
                    inputs,
                    merges_fh,
                    devs,
                    num_symbols,
                    window_size=window_size,
                    alpha=alpha,
                    min_frequency=min_frequency,
                    verbose=verbose,
                    is_dict=False,
                    total_symbols=False,
                    num_global=global_merges,
                    ratio=ratio_arr,
                    num_workers=1,
                    bpe_file=None,
                )
        finally:
            for s in inputs:
                s.close()
            if devs:
                for s in devs:
                    s.close()

    try:
        return _merges_to_tokenizer_json(merges_path, output_dir)
    finally:
        merges_path.unlink(missing_ok=True)
