"""Factory that resolves the extraction provider from config."""

from __future__ import annotations

import logging
import os

from engram.config import ActivationConfig, EngramConfig
from engram.extraction.extractor import EntityExtractor

logger = logging.getLogger(__name__)


def create_extractor(config: EngramConfig):
    """Create the appropriate extractor based on ``extraction_provider`` config.

    Resolution order for ``"auto"`` (default):
    1. Anthropic — if ANTHROPIC_API_KEY is set
    2. Ollama — if reachable at configured base_url (sync probe)
    3. Narrow — always available, zero dependencies

    Returns an object with ``async extract(text) -> ExtractionResult``.
    """
    cfg = config.activation
    provider = cfg.extraction_provider

    if provider == "anthropic":
        return _try_anthropic(cfg, strict=True)

    if provider == "ollama":
        return _try_ollama(cfg, strict=True)

    if provider == "narrow":
        return _make_narrow(cfg)

    # auto: try anthropic → ollama → narrow
    extractor = _try_anthropic(cfg, strict=False)
    if extractor is not None:
        return extractor

    extractor = _try_ollama(cfg, strict=False)
    if extractor is not None:
        return extractor

    return _make_narrow(cfg)


def _try_anthropic(cfg: ActivationConfig, *, strict: bool):
    """Return EntityExtractor if API key is available."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if api_key:
        logger.info("Extraction provider: Anthropic (Claude Haiku)")
        return EntityExtractor()
    if strict:
        logger.warning(
            "extraction_provider='anthropic' but ANTHROPIC_API_KEY not set — "
            "falling back to narrow extraction"
        )
        return _make_narrow(cfg)
    return None


def _try_ollama(cfg: ActivationConfig, *, strict: bool):
    """Return OllamaExtractor if Ollama is reachable (sync probe)."""
    try:
        import httpx

        resp = httpx.get(f"{cfg.ollama_base_url.rstrip('/')}/api/tags", timeout=2.0)
        if resp.status_code == 200:
            from engram.extraction.ollama_extractor import OllamaExtractor

            logger.info(
                "Extraction provider: Ollama (%s at %s)",
                cfg.ollama_model,
                cfg.ollama_base_url,
            )
            return OllamaExtractor(
                model=cfg.ollama_model,
                base_url=cfg.ollama_base_url,
            )
    except Exception:
        pass

    if strict:
        logger.warning(
            "extraction_provider='ollama' but Ollama not reachable at %s — "
            "falling back to narrow extraction",
            cfg.ollama_base_url,
        )
        return _make_narrow(cfg)
    return None


def _make_narrow(cfg: ActivationConfig):
    """Create the deterministic narrow extraction adapter."""
    from engram.extraction.narrow_adapter import NarrowExtractorAdapter

    logger.info("Extraction provider: Narrow (deterministic, zero-dependency)")
    return NarrowExtractorAdapter(cfg)
