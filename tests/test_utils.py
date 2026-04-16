"""
Unit tests for utils.py.

All heavy ML dependencies are mocked in conftest.py.
These tests verify the pure-Python business logic that lives in utils.py.
"""



# ─── extract_metadata_advanced ────────────────────────────────────────────────

def test_metadata_count_matches_chunks(tmp_path):
    """One metadata dict is returned per chunk."""
    from utils import extract_metadata_advanced

    fake_pdf = tmp_path / "report.pdf"
    fake_pdf.write_text("dummy")

    chunks = ["chunk_a", "chunk_b", "chunk_c"]
    result = extract_metadata_advanced(str(fake_pdf), tmp_path, chunks)

    assert len(result) == len(chunks)


def test_metadata_fields_are_present(tmp_path):
    """Every metadata dict contains the four required keys."""
    from utils import extract_metadata_advanced

    fake_pdf = tmp_path / "report.pdf"
    fake_pdf.write_text("dummy")

    result = extract_metadata_advanced(str(fake_pdf), tmp_path, ["only_chunk"])
    entry = result[0]

    assert entry["file_name"] == "report.pdf"
    assert entry["full_pdf_path"] == str(fake_pdf)
    assert entry["chunk_index"] == 0
    assert "file_modified" in entry


def test_metadata_chunk_index_is_sequential(tmp_path):
    """chunk_index must be 0-based and sequential."""
    from utils import extract_metadata_advanced

    fake_pdf = tmp_path / "data.pdf"
    fake_pdf.write_text("dummy")

    result = extract_metadata_advanced(str(fake_pdf), tmp_path, ["a", "b", "c", "d"])
    indices = [r["chunk_index"] for r in result]

    assert indices == [0, 1, 2, 3]


# ─── extract_structured_info_from_pdf ─────────────────────────────────────────

def test_parse_key_value_pairs(monkeypatch):
    """Standard key: value lines in a PDF are extracted as a dict."""
    import utils
    monkeypatch.setattr(utils, "get_md_from_pdf_path", lambda _: "nombre: Juan García\nfecha: 2024-01-15\n")

    result = utils.extract_structured_info_from_pdf("irrelevant.pdf")

    assert result.get("nombre") == "Juan García"
    assert result.get("fecha") == "2024-01-15"


def test_parse_returns_empty_dict_for_blank_content(monkeypatch):
    """Blank PDF content must return an empty dict, not raise."""
    import utils
    monkeypatch.setattr(utils, "get_md_from_pdf_path", lambda _: "")

    result = utils.extract_structured_info_from_pdf("empty.pdf")

    assert isinstance(result, dict)
    assert len(result) == 0


def test_parse_ignores_lines_without_values(monkeypatch):
    """Lines with a key but no value after the colon are skipped."""
    import utils
    monkeypatch.setattr(utils, "get_md_from_pdf_path", lambda _: "clave:\nbuena: presente\n")

    result = utils.extract_structured_info_from_pdf("partial.pdf")

    assert "clave" not in result
    assert result.get("buena") == "presente"


def test_parse_normalises_key_to_lowercase_snake_case(monkeypatch):
    """Keys must be lower-cased and spaces replaced by underscores."""
    import utils
    monkeypatch.setattr(utils, "get_md_from_pdf_path", lambda _: "Nombre Completo: Ana López\n")

    result = utils.extract_structured_info_from_pdf("doc.pdf")

    assert "nombre_completo" in result
    assert result["nombre_completo"] == "Ana López"
