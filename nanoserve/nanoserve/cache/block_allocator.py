"""A trivial first-fit block allocator backing the paged KV cache.

Each block holds ``block_size`` tokens (per layer, for both K and V).
The allocator tracks a free list as a deque and supports allocation
and freeing of arbitrary contiguous *counts* (not contiguous block IDs:
fragmentation is fine because the cache is paged).
"""
from __future__ import annotations

from collections import deque
from typing import List


class BlockAllocator:
    def __init__(self, num_blocks: int):
        # Block 0 is reserved as a sentinel "padding" block.  Sequences that
        # have fewer logical blocks than ``max_blocks_per_seq`` use this
        # sentinel for unused slots in their block table, which lets us
        # build a uniform 2-D block_tables tensor without worrying about
        # bogus reads (the kernel masks based on seq_len anyway).
        self.num_blocks = num_blocks
        self._free: deque[int] = deque(range(1, num_blocks))  # skip 0
        self._padding_block = 0

    @property
    def padding_block(self) -> int:
        return self._padding_block

    @property
    def num_free_blocks(self) -> int:
        return len(self._free)

    def can_allocate(self, n: int) -> bool:
        return len(self._free) >= n

    def allocate(self, n: int) -> List[int]:
        if not self.can_allocate(n):
            raise RuntimeError(
                f"BlockAllocator OOM: requested {n}, free {len(self._free)}/{self.num_blocks}"
            )
        return [self._free.popleft() for _ in range(n)]

    def free(self, blocks: List[int]) -> None:
        for b in blocks:
            if b == self._padding_block:
                continue
            self._free.append(b)
