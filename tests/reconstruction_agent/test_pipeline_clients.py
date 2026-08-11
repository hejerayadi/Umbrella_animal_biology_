from pathlib import Path

import httpx
import pytest

from backend.agents.reconstruction_agent.data_ingestion import SpeciesConfig, IngestionSummary
from backend.agents.reconstruction_agent.pipeline import IngestionPipeline
from backend.agents.reconstruction_agent.ncbi_client import NCBIClient, NCBIClientError
from backend.agents.reconstruction_agent.ena_client import ENAClient, ENAClientError
from backend.agents.reconstruction_agent.data_ingestion.ncbi_client import (
    NCBIClient as NCBIClientDI,
    NCBIClientError as NCBIClientErrorDI,
)
from backend.agents.reconstruction_agent.data_ingestion.ena_client import (
    ENAClient as ENAClientDI,
    ENAClientError as ENAClientErrorDI,
)


def test_ingestion_pipeline_runs_steps_in_order(monkeypatch, tmp_path):
    calls = []

    class FakeClient:
        def fetch_assembly_metadata(self, accession):
            return {"assembly_level": "Chromosome", "accession": accession}

        def download_fasta(self, accession, dest_path):
            dest_path.write_text(">seq\nA" * 2500, encoding="utf-8")
            return dest_path

    class FakeENAClient:
        def fetch_project_runs(self, project_accession):
            return [{"accession": "ERR1"}]

        def download_fasta(self, run_accession, dest_path):
            dest_path.write_text(">seq\nA" * 2500, encoding="utf-8")
            return dest_path

    pipeline = IngestionPipeline(raw_data_path=tmp_path, ncbi_client=FakeClient(), ena_client=FakeENAClient(), pg_conn=object(), qdrant_client=object())
    config = SpeciesConfig(accession="GCA_0001", species_id="sp1", source="ncbi", is_mammoth=False, partition="train")

    summary = pipeline.run([config])

    assert isinstance(summary, IngestionSummary)
    assert summary.processed_accessions == ["GCA_0001"]


def test_ncbi_client_raises_on_non_2xx(monkeypatch):
    client = NCBIClient(base_url="https://example.test")

    def fake_request(self, url):
        raise RuntimeError("boom")

    monkeypatch.setattr(NCBIClient, "_request_json", fake_request)

    try:
        client.fetch_assembly_metadata("ABC")
    except NCBIClientError:
        assert True
    else:
        assert False


# ---------------------------------------------------------------------------
# download_fasta tests (httpx.MockTransport — no real network calls)
# ---------------------------------------------------------------------------

def _ncbi_mock_handler(request: httpx.Request) -> httpx.Response:
    """Route mock NCBI efetch responses based on the 'id' query param."""
    accession = dict(request.url.params).get("id", "")
    if accession == "NC_002008.4":
        return httpx.Response(200, text=">NC_002008.4\nACGTACGT\n")
    if accession == "TIMEOUT":
        raise httpx.ReadTimeout("simulated timeout")
    # Invalid accession — NCBI returns 200 with an error message (not FASTA)
    return httpx.Response(200, text="Error: cannot find accession\n")


def _ena_mock_handler(request: httpx.Request) -> httpx.Response:
    """Route mock ENA responses based on the URL path."""
    path = request.url.path
    if path.endswith("/ERR1234567"):
        return httpx.Response(200, text=">ERR1234567\nGGCCTTAA\n")
    if path.endswith("/TIMEOUT"):
        raise httpx.ReadTimeout("simulated timeout")
    # Invalid accession
    return httpx.Response(200, text="")


def _mock_httpx_client_factory(transport):
    """Return a factory that creates httpx.Client with the given MockTransport."""
    _real_init = httpx.Client.__init__

    def patched_init(self, **kwargs):
        kwargs.pop("timeout", None)
        kwargs["transport"] = transport
        _real_init(self, **kwargs)

    return patched_init


# -- NCBI (outer copy) tests --

def test_ncbi_download_fasta_invalid_accession_raises(tmp_path, monkeypatch):
    transport = httpx.MockTransport(_ncbi_mock_handler)
    monkeypatch.setattr(httpx.Client, "__init__", _mock_httpx_client_factory(transport))

    client = NCBIClient()
    with pytest.raises(NCBIClientError) as exc_info:
        client.download_fasta("INVALID_ACC_XYZ", tmp_path / "out.fasta")
    assert exc_info.value.status_code == 404
    assert "INVALID_ACC_XYZ" in str(exc_info.value)


def test_ncbi_download_fasta_valid_accession_writes_file(tmp_path, monkeypatch):
    transport = httpx.MockTransport(_ncbi_mock_handler)
    monkeypatch.setattr(httpx.Client, "__init__", _mock_httpx_client_factory(transport))

    client = NCBIClient()
    dest = tmp_path / "out.fasta"
    result = client.download_fasta("NC_002008.4", dest)

    assert result == dest
    content = dest.read_text(encoding="utf-8")
    assert content.startswith(">NC_002008.4")
    assert "ACGTACGT" in content


def test_ncbi_download_fasta_http_error_raises(tmp_path, monkeypatch):
    def error_handler(request):
        return httpx.Response(500, text="Internal Server Error")

    transport = httpx.MockTransport(error_handler)
    monkeypatch.setattr(httpx.Client, "__init__", _mock_httpx_client_factory(transport))

    client = NCBIClient()
    with pytest.raises(NCBIClientError) as exc_info:
        client.download_fasta("ANY", tmp_path / "out.fasta")
    assert exc_info.value.status_code == 500


# -- NCBI (data_ingestion copy) tests --

def test_ncbi_di_download_fasta_invalid_accession_raises(tmp_path, monkeypatch):
    # This is the exact client imported by data_ingestion/pipeline.py in production
    transport = httpx.MockTransport(_ncbi_mock_handler)
    monkeypatch.setattr(httpx.Client, "__init__", _mock_httpx_client_factory(transport))

    client = NCBIClientDI()
    with pytest.raises(NCBIClientErrorDI) as exc_info:
        client.download_fasta("INVALID_ACC_XYZ", tmp_path / "out.fasta")
    assert exc_info.value.status_code == 404


def test_ncbi_di_download_fasta_valid_accession_writes_file(tmp_path, monkeypatch):
    # This is the exact client imported by data_ingestion/pipeline.py in production
    transport = httpx.MockTransport(_ncbi_mock_handler)
    monkeypatch.setattr(httpx.Client, "__init__", _mock_httpx_client_factory(transport))

    client = NCBIClientDI()
    dest = tmp_path / "out.fasta"
    result = client.download_fasta("NC_002008.4", dest)

    assert result == dest
    content = dest.read_text(encoding="utf-8")
    assert content.startswith(">NC_002008.4")
    assert "ACGTACGT" in content


def test_ncbi_di_download_fasta_http_error_raises(tmp_path, monkeypatch):
    # This is the exact client imported by data_ingestion/pipeline.py in production
    def error_handler(request):
        return httpx.Response(500, text="Internal Server Error")

    transport = httpx.MockTransport(error_handler)
    monkeypatch.setattr(httpx.Client, "__init__", _mock_httpx_client_factory(transport))

    client = NCBIClientDI()
    with pytest.raises(NCBIClientErrorDI) as exc_info:
        client.download_fasta("ANY", tmp_path / "out.fasta")
    assert exc_info.value.status_code == 500


# -- ENA (data_ingestion copy) tests --

def test_ena_di_download_fasta_invalid_accession_raises(tmp_path, monkeypatch):
    transport = httpx.MockTransport(_ena_mock_handler)
    monkeypatch.setattr(httpx.Client, "__init__", _mock_httpx_client_factory(transport))

    client = ENAClientDI()
    with pytest.raises(ENAClientErrorDI) as exc_info:
        client.download_fasta("INVALID_RUN_XYZ", tmp_path / "out.fasta")
    assert exc_info.value.status_code == 404


def test_ena_di_download_fasta_valid_accession_writes_file(tmp_path, monkeypatch):
    transport = httpx.MockTransport(_ena_mock_handler)
    monkeypatch.setattr(httpx.Client, "__init__", _mock_httpx_client_factory(transport))

    client = ENAClientDI()
    dest = tmp_path / "out.fasta"
    result = client.download_fasta("ERR1234567", dest)

    assert result == dest
    content = dest.read_text(encoding="utf-8")
    assert content.startswith(">ERR1234567")


def test_ena_di_download_fasta_http_error_raises(tmp_path, monkeypatch):
    def error_handler(request):
        return httpx.Response(503, text="Service Unavailable")

    transport = httpx.MockTransport(error_handler)
    monkeypatch.setattr(httpx.Client, "__init__", _mock_httpx_client_factory(transport))

    client = ENAClientDI()
    with pytest.raises(ENAClientErrorDI) as exc_info:
        client.download_fasta("ANY", tmp_path / "out.fasta")
    assert exc_info.value.status_code == 503


# -- ENA (outer copy) tests --

def test_ena_download_fasta_invalid_accession_raises(tmp_path, monkeypatch):
    transport = httpx.MockTransport(_ena_mock_handler)
    monkeypatch.setattr(httpx.Client, "__init__", _mock_httpx_client_factory(transport))

    client = ENAClient()
    with pytest.raises(ENAClientError) as exc_info:
        client.download_fasta("INVALID_RUN_XYZ", tmp_path / "out.fasta")
    assert exc_info.value.status_code == 404


def test_ena_download_fasta_valid_accession_writes_file(tmp_path, monkeypatch):
    transport = httpx.MockTransport(_ena_mock_handler)
    monkeypatch.setattr(httpx.Client, "__init__", _mock_httpx_client_factory(transport))

    client = ENAClient()
    dest = tmp_path / "out.fasta"
    result = client.download_fasta("ERR1234567", dest)

    assert result == dest
    content = dest.read_text(encoding="utf-8")
    assert content.startswith(">ERR1234567")


def test_ena_download_fasta_http_error_raises(tmp_path, monkeypatch):
    def error_handler(request):
        return httpx.Response(503, text="Service Unavailable")

    transport = httpx.MockTransport(error_handler)
    monkeypatch.setattr(httpx.Client, "__init__", _mock_httpx_client_factory(transport))

    client = ENAClient()
    with pytest.raises(ENAClientError) as exc_info:
        client.download_fasta("ANY", tmp_path / "out.fasta")
    assert exc_info.value.status_code == 503
