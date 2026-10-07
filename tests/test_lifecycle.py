import asyncio

import pytest

from app.lifecycle import lifespan


class FakeApp:
    pass


def test_lifespan_allows_application_to_start():
    app = FakeApp()

    async def run_test():
        async with lifespan(app):
            assert True

    asyncio.run(run_test())


def test_lifespan_clears_dependency_caches_on_shutdown(
    monkeypatch,
):
    import app.lifecycle as lifecycle

    cleanup_calls = []

    def fake_clear_dependency_caches():
        cleanup_calls.append(True)

    monkeypatch.setattr(
        lifecycle,
        "clear_dependency_caches",
        fake_clear_dependency_caches,
    )

    app = FakeApp()

    async def run_test():
        async with lifecycle.lifespan(app):
            assert cleanup_calls == []

    asyncio.run(run_test())

    assert cleanup_calls == [True]


def test_lifespan_clears_dependency_caches_when_application_fails(
    monkeypatch,
):
    import app.lifecycle as lifecycle

    cleanup_calls = []

    def fake_clear_dependency_caches():
        cleanup_calls.append(True)

    monkeypatch.setattr(
        lifecycle,
        "clear_dependency_caches",
        fake_clear_dependency_caches,
    )

    app = FakeApp()

    async def run_test():
        with pytest.raises(RuntimeError, match="application failure"):
            async with lifecycle.lifespan(app):
                raise RuntimeError("application failure")

    asyncio.run(run_test())

    assert cleanup_calls == [True]