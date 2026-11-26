# SPDX-License-Identifier: Apache-2.0
"""Auto-select PinnedNpuCacheContext for native PP KV cache registration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lmcache.v1.multiprocess.pp_layout import PPRegistrationInfo
    from lmcache.v1.platform.base.cache_context import BaseCacheContext
    from lmcache.v1.platform.devices.npu import NpuDeviceSpec


def _create_context_for_registration(
    self: NpuDeviceSpec,
    *args: Any,
    registration_info: PPRegistrationInfo | None,
    **kwargs: Any,
) -> BaseCacheContext:
    """Use host-pinned block index staging for native PP registrations."""
    if registration_info is None:
        return self.create_cache_context(*args, **kwargs)
    from lmcache_ascend.v1.pp_stage_pinning import PinnedNpuCacheContext

    return PinnedNpuCacheContext(*args, **kwargs)


def install_npu_registration_context() -> None:
    """Install the PP registration context hook without eager device imports."""
    from lmcache.v1.platform.devices.npu import NpuDeviceSpec

    NpuDeviceSpec.create_cache_context_for_registration = (
        _create_context_for_registration
    )
