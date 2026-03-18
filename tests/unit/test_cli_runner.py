from __future__ import annotations

import contextlib
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

import paper_context.cli as cli_module
import paper_context.worker.loop as worker_loop_module
import paper_context.worker.runner as runner_module
from paper_context.queue.contracts import ClaimedIngestMessage, IngestionQueue, IngestQueuePayload
from paper_context.queue.pgmq import PgmqMessage

pytestmark = pytest.mark.unit


def make_settings() -> SimpleNamespace:
    return SimpleNamespace(
        log_level="INFO",
        storage=SimpleNamespace(root_path="/tmp/paper-context-tests"),
        queue=SimpleNamespace(
            name="document_ingest",
            visibility_timeout_seconds=120,
            max_poll_seconds=7,
            poll_interval_ms=80,
        ),
        providers=SimpleNamespace(
            index_version="mvp-v1",
            voyage_model="voyage-4-large",
            reranker_model="zerank-2",
        ),
        chunking=SimpleNamespace(
            version="phase1",
            min_tokens=300,
            max_tokens=700,
            overlap_fraction=0.15,
        ),
        runtime=SimpleNamespace(
            app_host="127.0.0.1",
            app_port=9000,
            worker_idle_sleep_seconds=0.25,
        ),
    )


def make_claimed_message() -> ClaimedIngestMessage:
    payload = IngestQueuePayload(ingest_job_id=uuid4(), document_id=uuid4())
    return ClaimedIngestMessage(
        message=PgmqMessage(
            msg_id=7,
            read_ct=1,
            enqueued_at=datetime.now(UTC),
            vt=datetime.now(UTC),
            message={
                "ingest_job_id": str(payload.ingest_job_id),
                "document_id": str(payload.document_id),
            },
        ),
        payload=payload,
    )


def test_build_parser_supports_worker_once_flag() -> None:
    parser = cli_module.build_parser()

    args = parser.parse_args(["worker", "--once"])

    assert args.command == "worker"
    assert args.once is True


def test_main_runs_serve_command(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = make_settings()
    configure_logging = MagicMock()
    uvicorn_run = MagicMock()
    monkeypatch.setattr(cli_module, "get_settings", lambda: settings)
    monkeypatch.setattr(cli_module, "configure_logging", configure_logging)
    monkeypatch.setattr(cli_module.uvicorn, "run", uvicorn_run)

    exit_code = cli_module.main(["serve"])

    assert exit_code == 0
    configure_logging.assert_called_once_with("INFO")
    uvicorn_run.assert_called_once_with(
        "paper_context.api.app:create_app",
        host="127.0.0.1",
        port=9000,
        factory=True,
    )


def test_main_runs_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = make_settings()
    run_worker = MagicMock()
    monkeypatch.setattr(cli_module, "get_settings", lambda: settings)
    monkeypatch.setattr(cli_module, "configure_logging", MagicMock())
    monkeypatch.setattr(cli_module, "run_worker", run_worker)

    exit_code = cli_module.main(["worker", "--once"])

    assert exit_code == 0
    run_worker.assert_called_once_with(once=True)


def test_main_writes_verification_report(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    settings = make_settings()
    report = {"handled_message": True, "document_id": str(uuid4())}
    monkeypatch.setattr(cli_module, "get_settings", lambda: settings)
    monkeypatch.setattr(cli_module, "configure_logging", MagicMock())
    monkeypatch.setattr(cli_module, "run_synthetic_job_verification", lambda: report)

    exit_code = cli_module.main(["verify-synthetic-job"])

    assert exit_code == 0
    assert capsys.readouterr().out == (
        f'{{\n  "handled_message": true,\n  "document_id": "{report["document_id"]}"\n}}\n'
    )


def test_main_returns_one_for_unexpected_command(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = make_settings()
    parser = MagicMock()
    parser.parse_args.return_value = SimpleNamespace(command="unexpected")
    monkeypatch.setattr(cli_module, "build_parser", lambda: parser)
    monkeypatch.setattr(cli_module, "get_settings", lambda: settings)
    monkeypatch.setattr(cli_module, "configure_logging", MagicMock())

    assert cli_module.main([]) == 1


def test_worker_does_not_archive_when_processor_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    claimed = make_claimed_message()
    queue = MagicMock(spec=IngestionQueue)
    queue.claim_ingest.return_value = claimed
    processor = MagicMock()
    processor.process.side_effect = RuntimeError("boom")
    lease = MagicMock()
    connection = MagicMock()
    monkeypatch.setattr(worker_loop_module, "LeaseExtender", MagicMock(return_value=lease))
    worker = worker_loop_module.IngestWorker(
        connection_factory=lambda: contextlib.nullcontext(connection),
        queue_adapter=queue,
        processor=processor,
    )

    with pytest.raises(RuntimeError, match="boom"):
        worker.run_once()

    lease.extend.assert_called_once_with()
    queue.archive_message.assert_not_called()


def test_build_worker_uses_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = make_settings()
    queue_cls = MagicMock()
    queue_instance = MagicMock()
    queue_cls.return_value = queue_instance
    storage_cls = MagicMock()
    storage_instance = MagicMock()
    storage_cls.return_value = storage_instance
    primary_parser_cls = MagicMock()
    primary_parser = MagicMock()
    primary_parser_cls.return_value = primary_parser
    fallback_parser_cls = MagicMock()
    fallback_parser = MagicMock()
    fallback_parser_cls.return_value = fallback_parser
    enricher_cls = MagicMock()
    enricher = MagicMock()
    enricher_cls.return_value = enricher
    retrieval_indexer_cls = MagicMock()
    retrieval_indexer = MagicMock()
    retrieval_indexer_cls.return_value = retrieval_indexer
    deterministic_embedding_cls = MagicMock()
    deterministic_embedding = MagicMock()
    deterministic_embedding_cls.return_value = deterministic_embedding
    heuristic_reranker_cls = MagicMock()
    heuristic_reranker = MagicMock()
    heuristic_reranker_cls.return_value = heuristic_reranker
    processor_cls = MagicMock()
    processor_instance = MagicMock()
    processor_cls.return_value = processor_instance
    worker_cls = MagicMock()
    engine = MagicMock()
    monkeypatch.setattr(runner_module, "get_settings", lambda: settings)
    monkeypatch.setattr(runner_module, "get_engine", lambda: engine)
    monkeypatch.setattr(runner_module, "IngestionQueue", queue_cls)
    monkeypatch.setattr(runner_module, "LocalFilesystemStorage", storage_cls)
    monkeypatch.setattr(runner_module, "DoclingPdfParser", primary_parser_cls)
    monkeypatch.setattr(runner_module, "PdfPlumberPdfParser", fallback_parser_cls)
    monkeypatch.setattr(runner_module, "NullMetadataEnricher", enricher_cls)
    monkeypatch.setattr(runner_module, "DocumentRetrievalIndexer", retrieval_indexer_cls)
    monkeypatch.setattr(runner_module, "DeterministicEmbeddingClient", deterministic_embedding_cls)
    monkeypatch.setattr(runner_module, "HeuristicRerankerClient", heuristic_reranker_cls)
    monkeypatch.setattr(runner_module, "VoyageEmbeddingClient", MagicMock())
    monkeypatch.setattr(runner_module, "ZeroEntropyRerankerClient", MagicMock())
    monkeypatch.setattr(runner_module, "DeterministicIngestProcessor", processor_cls)
    monkeypatch.setattr(runner_module, "IngestWorker", worker_cls)

    runner_module.build_worker()

    queue_cls.assert_called_once_with("document_ingest")
    storage_cls.assert_called_once_with("/tmp/paper-context-tests")
    storage_instance.ensure_root.assert_called_once_with()
    primary_parser_cls.assert_called_once_with()
    fallback_parser_cls.assert_called_once_with()
    enricher_cls.assert_called_once_with()
    processor_cls.assert_called_once_with(
        storage=storage_instance,
        primary_parser=primary_parser,
        fallback_parser=fallback_parser,
        metadata_enricher=enricher,
        index_version="mvp-v1",
        chunking_version="phase1",
        embedding_model="voyage-4-large",
        reranker_model="zerank-2",
        min_tokens=300,
        max_tokens=700,
        overlap_fraction=0.15,
        retrieval_indexer=retrieval_indexer,
    )
    worker_cls.assert_called_once()
    kwargs = worker_cls.call_args.kwargs
    assert kwargs["queue_adapter"] is queue_instance
    assert kwargs["processor"] is processor_instance
    assert kwargs["config"] == worker_loop_module.WorkerConfig(
        vt_seconds=120,
        max_poll_seconds=7,
        poll_interval_ms=80,
    )


def test_build_worker_connection_factory_opens_connection_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = make_settings()
    engine = object()
    entered = object()
    connection_scope = MagicMock(return_value=contextlib.nullcontext(entered))
    worker = MagicMock()
    worker_cls = MagicMock(return_value=worker)
    monkeypatch.setattr(runner_module, "get_settings", lambda: settings)
    monkeypatch.setattr(runner_module, "get_engine", lambda: engine)
    monkeypatch.setattr(runner_module, "connection_scope", connection_scope)
    monkeypatch.setattr(runner_module, "IngestionQueue", MagicMock())
    monkeypatch.setattr(
        runner_module,
        "LocalFilesystemStorage",
        MagicMock(return_value=MagicMock(ensure_root=MagicMock())),
    )
    monkeypatch.setattr(runner_module, "DoclingPdfParser", MagicMock())
    monkeypatch.setattr(runner_module, "PdfPlumberPdfParser", MagicMock())
    monkeypatch.setattr(runner_module, "NullMetadataEnricher", MagicMock())
    monkeypatch.setattr(runner_module, "DocumentRetrievalIndexer", MagicMock())
    monkeypatch.setattr(runner_module, "DeterministicEmbeddingClient", MagicMock())
    monkeypatch.setattr(runner_module, "HeuristicRerankerClient", MagicMock())
    monkeypatch.setattr(runner_module, "VoyageEmbeddingClient", MagicMock())
    monkeypatch.setattr(runner_module, "ZeroEntropyRerankerClient", MagicMock())
    monkeypatch.setattr(runner_module, "DeterministicIngestProcessor", MagicMock())
    monkeypatch.setattr(runner_module, "IngestWorker", worker_cls)

    runner_module.build_worker()
    connection_factory = worker_cls.call_args.kwargs["connection_factory"]
    with connection_factory() as connection:
        assert connection is entered

    connection_scope.assert_called_once_with(engine)


def test_run_worker_returns_immediately_in_once_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    worker = MagicMock()
    worker.run_once.return_value = object()
    monkeypatch.setattr(runner_module, "get_settings", make_settings)
    monkeypatch.setattr(runner_module, "build_worker", lambda: worker)
    sleep = MagicMock()
    monkeypatch.setattr(runner_module.time, "sleep", sleep)

    runner_module.run_worker(once=True)

    worker.run_once.assert_called_once_with()
    sleep.assert_not_called()


def test_run_worker_sleeps_when_idle(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = make_settings()
    worker = MagicMock()
    worker.run_once.side_effect = [None, RuntimeError("stop")]
    sleep = MagicMock(side_effect=RuntimeError("stop"))
    monkeypatch.setattr(runner_module, "get_settings", lambda: settings)
    monkeypatch.setattr(runner_module, "build_worker", lambda: worker)
    monkeypatch.setattr(runner_module.time, "sleep", sleep)

    with pytest.raises(RuntimeError, match="stop"):
        runner_module.run_worker()

    sleep.assert_called_once_with(0.25)


def test_run_synthetic_job_verification_returns_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = make_settings()
    document_id = uuid4()
    ingest_job_id = uuid4()
    queue_metrics = SimpleNamespace(
        queue_name="document_ingest",
        queue_length=2,
        total_messages=5,
    )
    queue = MagicMock()
    queue.queue_metrics.return_value = queue_metrics
    service = MagicMock()
    service.enqueue_document.return_value = (document_id, ingest_job_id)
    worker = MagicMock()
    worker.run_once.return_value = object()
    connection = MagicMock()
    connection.execute.return_value.mappings.return_value.one.return_value = {
        "status": "ready",
        "started_at": datetime(2025, 1, 1, tzinfo=UTC),
        "finished_at": datetime(2025, 1, 2, tzinfo=UTC),
    }
    engine = MagicMock()
    engine.begin.return_value = contextlib.nullcontext(connection)
    monkeypatch.setattr(runner_module, "get_settings", lambda: settings)
    monkeypatch.setattr(runner_module, "get_engine", lambda: engine)
    monkeypatch.setattr(runner_module, "IngestionQueue", MagicMock(return_value=queue))
    monkeypatch.setattr(
        runner_module,
        "IngestionQueueService",
        MagicMock(return_value=service),
    )
    monkeypatch.setattr(runner_module, "build_worker", lambda: worker)

    report = runner_module.run_synthetic_job_verification()

    assert report == {
        "document_id": str(document_id),
        "ingest_job_id": str(ingest_job_id),
        "handled_message": True,
        "job_status": "ready",
        "started_at": datetime(2025, 1, 1, tzinfo=UTC),
        "finished_at": datetime(2025, 1, 2, tzinfo=UTC),
        "queue_metrics": {
            "queue_name": "document_ingest",
            "queue_length": 2,
            "total_messages": 5,
        },
    }
