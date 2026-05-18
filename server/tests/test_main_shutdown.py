from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest


@pytest.mark.asyncio
async def test_rest_shutdown_closes_runtime_stores_through_manager(monkeypatch) -> None:
    from engram import main as main_module
    from engram.security import middleware

    monkeypatch.setattr(middleware, "_oidc_validator", None)

    redis_subscriber = SimpleNamespace(stop=AsyncMock())
    episode_worker = SimpleNamespace(stop=AsyncMock())
    pressure_accumulator = SimpleNamespace(stop=AsyncMock())
    consolidation_scheduler = SimpleNamespace(stop=AsyncMock())
    redis_metering = SimpleNamespace(aclose=AsyncMock())
    consolidation_store = SimpleNamespace(close=AsyncMock())
    evaluation_store = SimpleNamespace(close=AsyncMock())
    atlas_store = SimpleNamespace(close=AsyncMock())
    conversation_store = SimpleNamespace(close=AsyncMock())
    manager = SimpleNamespace(close_runtime_resources=AsyncMock())
    search_index = SimpleNamespace(close=AsyncMock())
    activation_store = SimpleNamespace(close=AsyncMock())
    graph_store = SimpleNamespace(close=AsyncMock())

    main_module._app_state.clear()
    main_module._app_state.update(
        {
            "redis_subscriber": redis_subscriber,
            "episode_worker": episode_worker,
            "pressure_accumulator": pressure_accumulator,
            "consolidation_scheduler": consolidation_scheduler,
            "redis_metering": redis_metering,
            "consolidation_store": consolidation_store,
            "evaluation_store": evaluation_store,
            "atlas_store": atlas_store,
            "conversation_store": conversation_store,
            "graph_manager": manager,
            "search_index": search_index,
            "activation_store": activation_store,
            "graph_store": graph_store,
        }
    )

    try:
        await main_module._shutdown()
    finally:
        main_module._app_state.clear()

    redis_subscriber.stop.assert_awaited_once()
    episode_worker.stop.assert_awaited_once()
    pressure_accumulator.stop.assert_awaited_once()
    consolidation_scheduler.stop.assert_awaited_once()
    redis_metering.aclose.assert_awaited_once()
    consolidation_store.close.assert_awaited_once()
    evaluation_store.close.assert_awaited_once()
    atlas_store.close.assert_awaited_once()
    conversation_store.close.assert_awaited_once()
    manager.close_runtime_resources.assert_awaited_once()
    search_index.close.assert_not_awaited()
    activation_store.close.assert_not_awaited()
    graph_store.close.assert_not_awaited()


@pytest.mark.asyncio
async def test_rest_shutdown_falls_back_to_direct_runtime_store_closes(monkeypatch) -> None:
    from engram import main as main_module
    from engram.security import middleware

    monkeypatch.setattr(middleware, "_oidc_validator", None)

    search_index = SimpleNamespace(close=AsyncMock())
    activation_store = SimpleNamespace(close=AsyncMock())
    graph_store = SimpleNamespace(close=AsyncMock())

    main_module._app_state.clear()
    main_module._app_state.update(
        {
            "search_index": search_index,
            "activation_store": activation_store,
            "graph_store": graph_store,
        }
    )

    try:
        await main_module._shutdown()
    finally:
        main_module._app_state.clear()

    search_index.close.assert_awaited_once()
    activation_store.close.assert_awaited_once()
    graph_store.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_rest_shutdown_delegates_consolidation_to_helper(monkeypatch) -> None:
    from engram import main as main_module
    from engram.config import EngramConfig
    from engram.security import middleware

    monkeypatch.setattr(middleware, "_oidc_validator", None)
    run_shutdown_consolidation = AsyncMock()
    monkeypatch.setattr(
        main_module,
        "run_shutdown_consolidation",
        run_shutdown_consolidation,
    )

    config = EngramConfig()
    engine = object()
    main_module._app_state.clear()
    main_module._app_state.update(
        {
            "config": config,
            "consolidation_engine": engine,
        }
    )

    try:
        await main_module._shutdown()
    finally:
        main_module._app_state.clear()

    run_shutdown_consolidation.assert_awaited_once_with(
        engine,
        config=config,
        logger=main_module.logger,
    )


@pytest.mark.asyncio
async def test_shutdown_consolidation_helper_runs_enabled_cycle() -> None:
    from engram.config import EngramConfig
    from engram.consolidation_trigger import run_shutdown_consolidation

    config = EngramConfig()
    config.activation.consolidation_enabled = True
    engine = SimpleNamespace(is_running=False, run_cycle=AsyncMock(), cancel=Mock())

    await run_shutdown_consolidation(engine, config=config)

    engine.run_cycle.assert_awaited_once_with(
        group_id=config.default_group_id,
        trigger="shutdown",
        dry_run=False,
    )
    engine.cancel.assert_not_called()


@pytest.mark.asyncio
async def test_shutdown_consolidation_helper_skips_disabled_cycle() -> None:
    from engram.config import EngramConfig
    from engram.consolidation_trigger import run_shutdown_consolidation

    config = EngramConfig()
    config.activation.consolidation_enabled = False
    engine = SimpleNamespace(is_running=False, run_cycle=AsyncMock(), cancel=Mock())

    await run_shutdown_consolidation(engine, config=config)

    engine.run_cycle.assert_not_awaited()
    engine.cancel.assert_not_called()


@pytest.mark.asyncio
async def test_shutdown_consolidation_helper_cancels_running_engine() -> None:
    from engram.config import EngramConfig
    from engram.consolidation_trigger import run_shutdown_consolidation

    config = EngramConfig()
    config.activation.consolidation_enabled = True
    engine = SimpleNamespace(is_running=True, run_cycle=AsyncMock(), cancel=Mock())

    await run_shutdown_consolidation(engine, config=config)

    engine.cancel.assert_called_once_with()
    engine.run_cycle.assert_not_awaited()


@pytest.mark.asyncio
async def test_shutdown_consolidation_helper_logs_failed_cycle() -> None:
    from engram.config import EngramConfig
    from engram.consolidation_trigger import run_shutdown_consolidation

    config = EngramConfig()
    config.activation.consolidation_enabled = True
    engine = SimpleNamespace(
        is_running=False,
        run_cycle=AsyncMock(side_effect=RuntimeError("shutdown failed")),
        cancel=Mock(),
    )
    logger = SimpleNamespace(warning=Mock())

    await run_shutdown_consolidation(engine, config=config, logger=logger)

    engine.run_cycle.assert_awaited_once()
    logger.warning.assert_called_once_with("Shutdown consolidation failed", exc_info=True)
