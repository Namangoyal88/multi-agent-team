from pathlib import Path

APP = str(Path(__file__).resolve().parent.parent / "frontend" / "streamlit_app.py")


def test_streamlit_dashboard_renders_without_errors(monkeypatch):
    from streamlit.testing.v1 import AppTest
    import frontend.api_client as c

    monkeypatch.setattr(c.ResearchAPI, "list", lambda self: [])
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    rendered = "\n".join(getattr(m, "value", "") for m in at.markdown)
    assert "Big questions." in rendered
    assert any(getattr(b, "label", "") == "＋  New research" for b in at.button)
