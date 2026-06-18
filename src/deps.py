"""
Dependency wiring — PERSISTENCE_BACKEND controls which store backs the service.

memory → InMemoryRegistryStore (tests)
volume → VolumeRegistryStore   (production)
"""
from __future__ import annotations

import os
from functools import lru_cache

from src.core.ports.registry_store import RegistryStore
from src.core.services.registry_service import RegistryService


@lru_cache(maxsize=1)
def _get_store() -> RegistryStore:
    backend = os.environ.get("PERSISTENCE_BACKEND", "volume")
    if backend == "memory":
        from src.adapters.persistence.memory_store import InMemoryRegistryStore
        return InMemoryRegistryStore()
    elif backend == "volume":
        from src.adapters.persistence.volume_store import VolumeRegistryStore
        return VolumeRegistryStore()
    else:
        raise ValueError(f"Unknown PERSISTENCE_BACKEND='{backend}'")


def get_registry_service() -> RegistryService:
    return RegistryService(_get_store())
