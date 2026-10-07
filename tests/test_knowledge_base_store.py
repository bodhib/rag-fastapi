from app.knowledge_base_store import KnowledgeBaseStore


def test_create_knowledge_base(postgres_session_factory):
    store = KnowledgeBaseStore(
        session_factory=postgres_session_factory,
    )

    knowledge_base = store.create(
        knowledge_base_id="kb_test",
        name="Test Knowledge",
        slug="test-knowledge",
        description="Test knowledge base",
    )

    assert knowledge_base.knowledge_base_id == "kb_test"
    assert knowledge_base.name == "Test Knowledge"
    assert knowledge_base.slug == "test-knowledge"
    assert knowledge_base.description == "Test knowledge base"
    assert knowledge_base.active is True


def test_get_knowledge_base(postgres_session_factory):
    store = KnowledgeBaseStore(
        session_factory=postgres_session_factory,
    )

    store.create(
        knowledge_base_id="kb_get",
        name="Get Test",
        slug="get-test",
    )

    knowledge_base = store.get("kb_get")

    assert knowledge_base is not None
    assert knowledge_base.knowledge_base_id == "kb_get"


def test_get_by_slug(postgres_session_factory):
    store = KnowledgeBaseStore(
        session_factory=postgres_session_factory,
    )

    store.create(
        knowledge_base_id="kb_slug",
        name="Slug Test",
        slug="slug-test",
    )

    knowledge_base = store.get_by_slug("slug-test")

    assert knowledge_base is not None
    assert knowledge_base.knowledge_base_id == "kb_slug"


def test_get_missing_knowledge_base(postgres_session_factory):
    store = KnowledgeBaseStore(
        session_factory=postgres_session_factory,
    )

    knowledge_base = store.get("does-not-exist")

    assert knowledge_base is None


def test_list_knowledge_bases(postgres_session_factory):
    store = KnowledgeBaseStore(
        session_factory=postgres_session_factory,
    )

    store.create(
        knowledge_base_id="kb_list_1",
        name="First",
        slug="first",
    )

    store.create(
        knowledge_base_id="kb_list_2",
        name="Second",
        slug="second",
    )

    knowledge_bases = store.list()

    ids = {
        knowledge_base.knowledge_base_id
        for knowledge_base in knowledge_bases
    }

    assert "kb_list_1" in ids
    assert "kb_list_2" in ids


def test_list_active_only(postgres_session_factory):
    store = KnowledgeBaseStore(
        session_factory=postgres_session_factory,
    )

    store.create(
        knowledge_base_id="kb_active",
        name="Active",
        slug="active",
        active=True,
    )

    store.create(
        knowledge_base_id="kb_inactive",
        name="Inactive",
        slug="inactive",
        active=False,
    )

    knowledge_bases = store.list(active_only=True)

    ids = {
        knowledge_base.knowledge_base_id
        for knowledge_base in knowledge_bases
    }

    assert "kb_active" in ids
    assert "kb_inactive" not in ids