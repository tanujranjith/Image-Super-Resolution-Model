"""Scene-level exact and perceptual duplicate grouping for the independent SR project."""

from collections import defaultdict
import numpy as np


class HammingIndex:
    """Exact multi-index search: radius+1 disjoint chunks guarantee an unchanged chunk.

    Candidate distances use NumPy's vectorized 64-bit popcount. This avoids traversing
    a large BK-tree for every natural-image hash while keeping memory bounded.
    """

    def __init__(self, radius):
        self.radius = radius
        n = radius + 1
        widths = [64 // n + (i < 64 % n) for i in range(n)]
        self.parts, offset = [], 0
        for width in widths:
            self.parts.append((offset, (1 << width) - 1, {}))
            offset += width
        self.values, self.rows, self.exact = [], [], {}

    def add(self, value, row):
        if value in self.exact:
            return
        position = len(self.values)
        self.exact[value] = position
        self.values.append(value)
        self.rows.append(row)
        for offset, mask, buckets in self.parts:
            buckets.setdefault((value >> offset) & mask, []).append(position)

    def query(self, value):
        candidates = set()
        for offset, mask, buckets in self.parts:
            candidates.update(buckets.get((value >> offset) & mask, ()))
        positions = list(candidates)
        if not positions:
            return
        values = np.array([self.values[i] for i in positions], dtype=np.uint64)
        matches = np.bitwise_count(values ^ np.uint64(value)) <= self.radius
        for i in np.flatnonzero(matches):
            yield self.rows[positions[int(i)]]


def group_duplicates(rows, distance=6):
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(i)] = find(j)

    hashes = {}
    tree = HammingIndex(distance)
    for i, row in enumerate(rows):
        if row["sha256"] in hashes:
            union(i, hashes[row["sha256"]])
        hashes[row["sha256"]] = i
        values = {int(row["phash"], 16), int(row.get("phash_hflip", row["phash"]), 16)}
        for value in values:
            for j in tree.query(value):
                union(i, j)
        for value in values:
            tree.add(value, i)
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[find(i)].append(row)
    return list(groups.values())


def verify_split_isolation(rows, distance=6, batch_size=64):
    """Independent brute-force check of every held-out hash against all splits.

    Two views per scene; query batches bound temporary distance arrays to about
    32 MiB for this 32,000-scene corpus. This does not reuse the grouping index.
    """
    if batch_size < 1:
        raise ValueError("Positive hash-query batch size required")
    values = np.array(
        [int(row.get(key, row["phash"]), 16) for row in rows for key in ("phash", "phash_hflip")],
        dtype=np.uint64,
    )
    labels = np.repeat(np.array([row["split"] for row in rows]), 2)
    queries = np.flatnonzero(labels != "train")
    for start in range(0, len(queries), batch_size):
        positions = queries[start : start + batch_size]
        distances = np.bitwise_count(values[positions, None] ^ values[None, :])
        conflicts = (distances <= distance) & (labels[positions, None] != labels[None, :])
        if conflicts.any():
            raise ValueError("Full/mirror perceptual duplicate crosses splits")
    exact = {}
    for row in rows:
        previous = exact.setdefault(row["sha256"], row["split"])
        if previous != row["split"]:
            raise ValueError("Byte-identical image crosses splits")
