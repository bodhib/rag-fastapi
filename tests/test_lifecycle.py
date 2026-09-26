import asyncio

import app.lifecycle as lifecycle


def test_lifespan_initializes_retrieval_index(monkeypatch):
    calls = []

    def fake_get_retrieval_index_manager():
        calls.append("initialized")

    monkeypatch.setattr(
        lifecycle,
        "get_retrieval_index_manager",
        fake_get_retrieval_index_manager,
    )

    async def run_lifespan():
        async with lifecycle.lifespan(None):
            pass

    asyncio.run(run_lifespan())

    assert calls == ["initialized"]

    def test_lifespan_runs_startup_and_shutdown(monkeypatch):
        events = []

        def fake_get_retrieval_index_manager():
            events.append("startup")

        monkeypatch.setattr(
            lifecycle,
            "get_retrieval_index_manager",
            fake_get_retrieval_index_manager,
        )

        async def run_lifespan():
            async with lifecycle.lifespan(None):
                events.append("running")

            events.append("shutdown")

        asyncio.run(run_lifespan())

        assert events == [
            "startup",
            "running",
            "shutdown",
        ]