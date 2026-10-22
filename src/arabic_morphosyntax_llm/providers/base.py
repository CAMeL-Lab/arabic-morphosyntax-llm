"""Provider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..records import GenerationRequest, GenerationResult


class Provider(ABC):
    @abstractmethod
    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate and parse one structured response."""

    async def close(self) -> None:
        """Release any provider-owned clients."""
