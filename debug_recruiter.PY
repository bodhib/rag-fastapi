print("DEBUG SCRIPT STARTED", flush=True)

try:
    print("Importing FastAPI app...", flush=True)

    from fastapi.testclient import TestClient
    from app.main import app

    print("App imported successfully", flush=True)

    payload = {
        "query": "Find Python developers with FastAPI backend experience",
        "knowledge_base_id": "kb_recruitment",
        "skills": ["python", "fastapi"],
        "min_years_experience": 3,
        "top_k": 10,
    }

    print("Creating TestClient...", flush=True)

    with TestClient(
        app,
        raise_server_exceptions=True,
    ) as client:

        print("Sending request...", flush=True)

        response = client.post(
            "/recruiter/search",
            json=payload,
        )

        print("REQUEST COMPLETED", flush=True)
        print("STATUS:", response.status_code, flush=True)
        print("BODY:", response.text, flush=True)

except Exception as exc:
    print("\n===== EXCEPTION =====", flush=True)
    print(type(exc).__name__, flush=True)
    print(str(exc), flush=True)

    import traceback

    traceback.print_exc()