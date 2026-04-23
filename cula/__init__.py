# Copyright 2025-2026 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__version__ = "0.1.0"

from cula.ops.lightning_attn import LinearAttentionChunkwiseDecay

__all__ = [
    "LinearAttentionChunkwiseDecay",
]

# ------------------------------------------------------------------
# Monkey-patch backward-intra SM90 standalone extension into the
# main cula.cudac namespace so existing imports keep working.
# ------------------------------------------------------------------
try:
    import cula.cudac as _cudac
    import cula._kda_bwd_intra_sm90 as _bwd_ext

    _cudac.chunk_kda_bwd_intra_sm90 = _bwd_ext.chunk_kda_bwd_intra_sm90
except Exception:
    # Either extension is not built yet or SM90 is disabled; ignore
    pass
