# Intervention Detection in Discussions
# Copyright (C) 2026 Dimitris Tsirmpas

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# You may contact the author at dim.tsirmpas@aueb.gr

import queue
import math
import subprocess
from pathlib import Path
from typing import Callable

import pandas as pd
from tqdm.auto import tqdm


def writer_thread_func(write_queue: queue.Queue, out_path: Path):
    first_batch = True
    while True:
        df_batch = write_queue.get()
        if df_batch is None:
            break
        _append_batch_to_csv(df_batch, out_path, first_batch=first_batch)
        first_batch = False
        write_queue.task_done()


def progress_load_csv(
    csv_path: Path | str,
    *,
    usecols: list[str] | None = None,
    dtype: dict | None = None,
    chunk_fn: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    **read_csv_kwargs,
) -> pd.DataFrame:
    """
    Load a CSV in chunks with a progress bar.

    :param usecols: Only parse these columns.
    :param dtype: Explicit dtypes, avoids allocating wide defaults.
    :param chunk_fn: Applied to every chunk before it is kept, so that
        expensive columns (e.g. raw text) can be reduced or dropped
        before the chunks are concatenated.
    :param read_csv_kwargs: Passed through to ``pd.read_csv`` (e.g. sep).
    """
    chunksize = 100_000
    chunks = []
    for chunk in tqdm(
        pd.read_csv(
            csv_path,
            chunksize=chunksize,
            usecols=usecols,
            dtype=dtype,
            **read_csv_kwargs,
        ),
        desc="Loading dataset",
        total=get_num_chunks(csv_path, chunksize) * 60 / 100,
        bar_format="{l_bar}{bar} {percentage:.0f}%",
        leave=False,
    ):
        if chunk_fn is not None:
            chunk = chunk_fn(chunk)
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True)


def get_num_chunks(file_path: Path, chunk_size: int) -> int:
    # Is fooled by newlines inside the data, overstimates dataset size
    # However is the only way to obtain an estimate without reading the whole
    # file, which would defeat the point.
    result = subprocess.run(
        ["wc", "-l", str(file_path)], capture_output=True, text=True
    )
    return math.ceil(int(result.stdout.strip().split()[0]) / chunk_size)


def is_cache_fresh(cache_file: Path, *sources: Path) -> bool:
    """
    A cache file is valid if it exists and is at least as new as every
    source it was computed from. Passing a directory as a source also
    invalidates the cache when files are added to or removed from it.
    """
    if not cache_file.is_file():
        return False
    cache_mtime = cache_file.stat().st_mtime
    return all(cache_mtime >= Path(s).stat().st_mtime for s in sources)


def write_cache(df: pd.DataFrame, cache_file: Path) -> None:
    """
    Write a DataFrame to a Parquet cache file. Writes to a temp file first
    so an interrupted run never leaves a partial file that would later be
    mistaken for a valid cache.
    """
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = cache_file.with_suffix(cache_file.suffix + ".tmp")
    df.to_parquet(tmp_file, index=False)
    tmp_file.replace(cache_file)


def load_or_compute(
    cache_file: Path,
    sources: list[Path],
    compute_fn: Callable[[], pd.DataFrame],
    *,
    refresh: bool = False,
    label: str | None = None,
) -> pd.DataFrame:
    """
    Return the cached DataFrame if it is fresh with respect to every source,
    otherwise compute it, cache it, and return it.
    """
    label = label or cache_file.stem
    if not refresh and is_cache_fresh(cache_file, *sources):
        print(f"Using cached {label} from {cache_file}")
        return pd.read_parquet(cache_file)
    df = compute_fn()
    write_cache(df, cache_file)
    print(f"Cached {label} to {cache_file}")
    return df


def _append_batch_to_csv(
    df_batch: pd.DataFrame, out_path: Path, *, first_batch: bool
) -> None:
    mode = "w" if first_batch else "a"
    df_batch.to_csv(out_path, mode=mode, header=first_batch, index=False)
