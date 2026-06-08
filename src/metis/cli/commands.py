# SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0


import importlib
import copy
import inspect
import json
import logging
from datetime import datetime
from pathlib import Path
from rich.markup import escape

from metis.engine.llm_triage_service import (
    DEFAULT_LLM_TRIAGE_BATCH_SIZE,
    DEFAULT_LLM_TRIAGE_MODEL,
    DEFAULT_LLM_TRIAGE_REASONING_EFFORT,
)
from metis.engine.options import ReviewOptions, TriageOptions
from .command_runtime import CommandRuntime
from .review_progress import ReviewCodeProgressReporter
from metis.utils import read_file_content, safe_decode_unicode
from metis.sarif.writer import generate_sarif
from metis.usage import usage_operation
from .triage_cli import run_triage_action
from .utils import (
    check_file_exists,
    with_spinner,
    with_timer,
    collect_reviews,
    iterate_with_progress,
    build_standard_progress,
    count_index_items,
    pretty_print_reviews,
    save_output,
    print_console,
)


def _print_no_index_warning(args, runtime: CommandRuntime):
    if runtime.use_retrieval_context:
        return
    if runtime.no_index_warning_emitted:
        return
    print_console(
        "[yellow]Warning:[/yellow] Running without index; relevant-context retrieval was skipped.",
        args.quiet,
    )
    runtime.no_index_warning_emitted = True


def _review_options_for_runtime(runtime: CommandRuntime) -> ReviewOptions:
    return ReviewOptions(use_retrieval_context=runtime.use_retrieval_context)


def _triage_options_for_runtime(args, runtime: CommandRuntime) -> TriageOptions:
    return TriageOptions(
        use_retrieval_context=runtime.use_retrieval_context,
        include_triaged=bool(getattr(args, "include_triaged", False)),
    )


def show_help(args=None):
    print_console("""
[bold blue]Metis CLI[/bold blue]

Type one of the following commands (with arguments):

- [cyan]index[/cyan]
- [cyan]review_patch mypatch.diff[/cyan]
- [cyan]review_file path_to_file/myfile.c[/cyan]
- [cyan]review_code[/cyan]
- [cyan]triage findings.sarif[/cyan]
- [cyan]update patch.diff[/cyan]
- [cyan]ask "Give me an overview of the code"[/cyan]
- [magenta]exit[/magenta]   (quit the tool)
- [magenta]help[/magenta]   (show this message)

Options:
    --backend chroma|postgres  Vector backend to use (default: chroma).
    --output-file PATH         Save analysis results to this file.
    --custom-prompt PATH       Custom prompt file (.md or .txt) to guide analysis.
    --threat-model PATH        Threat model document to scope review and LLM triage.
    --triage                   Triage findings and annotate SARIF output for review commands.
    --include-triaged          Include findings already triaged by Metis.
    --llm-triage               Run reasoning-model triage after review_file or review_code.
    --llm-triage-model MODEL   Model for --llm-triage (default: gpt-5.5).
    --llm-triage-reasoning-effort LEVEL  Reasoning effort for --llm-triage (default: high).
    --llm-triage-batch-size N  Findings per LLM triage batch after similarity sorting (default: 10).
    --llm-triage-output-file PATH  Save the p0-p4 LLM triage JSON to this path.
    --ignore-index             Allow review_file, review_code, review_patch, and triage to run without index-backed context.
    --project-schema SCHEMA    (Optional) Project identifier if postgresql is used.
    --chroma-dir DIR           (Optional) Directory to store ChromaDB data (default: ./chromadb).
    --verbose                  (Optional) Shows detailed output in the terminal window.
    --version                  (Optional) Show program version
""")


def show_version(args=None):
    version = importlib.metadata.version("metis")
    print_console("Metis [green]" + version + "[/green]")


def run_review(engine, patch_file, args, runtime: CommandRuntime):
    if not check_file_exists(patch_file):
        return
    _print_no_index_warning(args, runtime)
    options = _review_options_for_runtime(runtime)
    results = with_spinner(
        "Reviewing patch...",
        engine.review.review_patch,
        patch_file=patch_file,
        options=options,
        quiet=args.quiet,
    )
    _finalize_review_output(engine, results, args, runtime)


def run_file_review(engine, file_path, args, runtime: CommandRuntime):
    if not check_file_exists(file_path):
        return
    _print_no_index_warning(args, runtime)
    options = _review_options_for_runtime(runtime)
    raw_result = with_spinner(
        f"Reviewing file {file_path}...",
        engine.review.review_file,
        file_path=file_path,
        options=options,
        quiet=args.quiet,
    )

    if raw_result and isinstance(raw_result.get("reviews"), list):
        results = {"reviews": [raw_result]}
    else:
        results = {"reviews": []}

    _finalize_review_output(engine, results, args, runtime)


def run_review_code(engine, args, runtime: CommandRuntime):
    _print_no_index_warning(args, runtime)
    options = _review_options_for_runtime(runtime)
    if not args.quiet:
        code_files = list(engine.review.get_code_files(options=options))
        file_reviews = _collect_review_code_with_progress(
            engine,
            options,
            code_files,
        )
        results = {"reviews": file_reviews}
    elif args.verbose:
        code_files = list(engine.review.get_code_files(options=options))
        file_reviews = iterate_with_progress(
            len(code_files),
            _review_code_iter(engine.review, options, code_files=code_files),
        )
        results = {"reviews": file_reviews}
    else:
        results = with_spinner(
            "Reviewing codebase...",
            collect_reviews,
            engine,
            options=options,
            quiet=args.quiet,
        )
    _finalize_review_output(engine, results, args, runtime)


def _collect_review_code_with_progress(engine, options, code_files):
    results = []
    total = len(code_files)
    with build_standard_progress(transient=True) as progress:
        progress_reporter = ReviewCodeProgressReporter(
            progress,
            total_files=total,
        )
        for item in _review_code_iter(
            engine.review,
            options,
            progress_callback=progress_reporter,
            code_files=code_files,
        ):
            if item is not None:
                results.append(item)
            progress_reporter.review_result()
        progress_reporter.finish()
    return results


def _review_code_iter(review_domain, options, progress_callback=None, code_files=None):
    review_code = review_domain.review_code
    kwargs = {"options": options}
    try:
        signature = inspect.signature(review_code)
    except (TypeError, ValueError):
        signature = None
    if signature is not None:
        params = signature.parameters
        accepts_kwargs = any(
            param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()
        )
        if progress_callback is not None and (
            "progress_callback" in params or accepts_kwargs
        ):
            kwargs["progress_callback"] = progress_callback
        if code_files is not None and (
            "get_code_files_func" in params or accepts_kwargs
        ):
            kwargs["get_code_files_func"] = lambda: code_files
    elif progress_callback is not None:
        kwargs["progress_callback"] = progress_callback
    return review_code(**kwargs)


def run_index(engine, verbose=False, quiet=False):
    if verbose:
        print_console("[cyan]Indexing codebase...[/cyan]", quiet)
        total = count_index_items(engine)
        if total > 0:
            iterate_with_progress(total, engine.indexing.index_prepare_nodes_iter())
            with_timer(
                "Embedding indexes...",
                engine.indexing.index_finalize_embeddings,
                quiet=quiet,
            )
            print_console("[green]Indexing completed successfully.[/green]", quiet)
            return

    with_spinner("Indexing codebase...", engine.indexing.index_codebase, quiet=quiet)
    print_console("[green]Indexing completed successfully.[/green]", quiet)


def run_update(engine, patch_file, args, runtime: CommandRuntime):
    if not check_file_exists(patch_file):
        return
    file_diff = read_file_content(patch_file)
    with_spinner(
        "Updating index...",
        engine.indexing.update_index,
        file_diff,
        quiet=args.quiet,
    )
    print_console("[green]Index update completed.[/green]", args.quiet)


def run_ask(engine, question, args, runtime: CommandRuntime):
    answer = with_spinner(
        "Thinking...", engine.ask_question, question, quiet=args.quiet
    )
    print_console("[bold magenta]Metis Answer:[/bold magenta]\n")
    if isinstance(answer, dict):
        if "code" in answer:
            print_console(
                f"[bold yellow]Code Context:[/bold yellow] {escape(safe_decode_unicode(answer['code']))} \n",
            )
        if "docs" in answer:
            print_console(
                f"[bold blue]Documentation Context:[/bold blue] {escape(safe_decode_unicode(answer['docs']))}",
            )
    else:
        print_console(escape(str(answer)))
    save_output(args.output_file, answer, args.quiet)


def run_triage(engine, sarif_path, args, runtime: CommandRuntime):
    if not check_file_exists(sarif_path, quiet=args.quiet):
        return
    if Path(sarif_path).suffix.lower() != ".sarif":
        print_console("[red]Only .sarif input files are supported.[/red]", args.quiet)
        return
    _print_no_index_warning(args, runtime)
    print_console("[cyan]Loading SARIF findings...[/cyan]", args.quiet)
    options = _triage_options_for_runtime(args, runtime)

    output_target = None
    if args.output_file:
        sarif_targets = [
            p for p in args.output_file if str(p).lower().endswith(".sarif")
        ]
        if sarif_targets:
            output_target = sarif_targets[0]

    def _invoke(kwargs):
        return engine.triage_sarif_file(
            sarif_path,
            output_target,
            options=options,
            **kwargs,
        )

    saved_path = run_triage_action(
        args,
        action=_invoke,
        spinner_text="Triaging SARIF findings...",
    )
    print_console(
        f"[green]Triage complete. SARIF saved to {escape(str(saved_path))}[/green]",
        args.quiet,
    )


def _build_triaged_sarif_payload(engine, results, args, runtime: CommandRuntime):
    if not getattr(args, "triage", False):
        return None
    try:
        sarif_payload = generate_sarif(results)
        _print_no_index_warning(args, runtime)
        options = _triage_options_for_runtime(args, runtime)

        def _invoke(kwargs):
            return engine.triage_sarif_payload(
                sarif_payload,
                options=options,
                **kwargs,
            )

        with usage_operation("triage"):
            return run_triage_action(
                args,
                action=_invoke,
                spinner_text="Triaging findings...",
            )
    except Exception as exc:
        print_console(
            f"[yellow]Triage skipped due to error: {escape(str(exc))}[/yellow]",
            args.quiet,
        )
        return None


def _llm_triage_requested(args) -> bool:
    return bool(getattr(args, "llm_triage", False))


def _llm_triage_supported(runtime: CommandRuntime) -> bool:
    return runtime.command in {"review_code", "review_file"}


def _llm_triage_batch_size(args) -> int:
    try:
        batch_size = int(
            getattr(args, "llm_triage_batch_size", DEFAULT_LLM_TRIAGE_BATCH_SIZE)
            or DEFAULT_LLM_TRIAGE_BATCH_SIZE
        )
    except (TypeError, ValueError):
        return DEFAULT_LLM_TRIAGE_BATCH_SIZE
    return max(1, batch_size)


def _resolve_llm_triage_output_path(args, runtime: CommandRuntime) -> Path:
    requested = getattr(args, "llm_triage_output_file", None)
    if requested:
        return Path(str(requested))

    output_files = getattr(args, "output_file", None) or []
    if isinstance(output_files, (str, Path)):
        output_files = [output_files]
    if output_files:
        base = Path(str(output_files[0]))
        return base.with_name(f"{base.stem}_llm_triage.json")

    Path("results").mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("results") / f"{runtime.command}_llm_triage_{timestamp}.json"


def _write_llm_triage_output(payload: dict, output_path: Path, quiet: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=4)
    print_console(
        f"[blue]LLM triage saved to {escape(str(output_path))}[/blue]",
        quiet,
    )


def _run_llm_triage_if_requested(engine, results, args, runtime: CommandRuntime):
    if not _llm_triage_requested(args):
        return None
    if not _llm_triage_supported(runtime):
        print_console(
            "[yellow]LLM triage skipped:[/yellow] --llm-triage only applies to review_code and review_file.",
            args.quiet,
        )
        return None

    model = getattr(args, "llm_triage_model", None) or DEFAULT_LLM_TRIAGE_MODEL
    reasoning_effort = (
        getattr(args, "llm_triage_reasoning_effort", None)
        or DEFAULT_LLM_TRIAGE_REASONING_EFFORT
    )
    batch_size = _llm_triage_batch_size(args)

    def _progress(event):
        if not getattr(args, "verbose", False):
            return
        ev = str(event.get("event") or "")
        if ev == "llm_triage_start":
            print_console(
                f"[cyan]LLM triage: {event.get('findings', 0)} finding(s) in "
                f"{event.get('batches', 0)} batch(es)[/cyan]",
                args.quiet,
            )
        elif ev == "llm_triage_batch_done":
            print_console(
                f"[green]LLM triage batch {event.get('batch', 0)}/"
                f"{event.get('batches', 0)}: kept {event.get('kept', 0)}[/green]",
                args.quiet,
            )
        elif ev == "llm_triage_done":
            print_console(
                f"[green]LLM triage: kept {event.get('kept', 0)}, "
                f"filtered {event.get('filtered', 0)}[/green]",
                args.quiet,
            )

    try:
        with usage_operation("llm_triage"):
            payload = with_spinner(
                "LLM triaging findings...",
                engine.llm_triage_reviews,
                results,
                model=model,
                reasoning_effort=reasoning_effort,
                batch_size=batch_size,
                progress_callback=_progress,
                quiet=args.quiet,
            )
    except Exception as exc:
        print_console(
            (
                "[yellow]LLM triage failed closed due to error: "
                f"{escape(str(exc))}[/yellow]"
            ),
            args.quiet,
        )
        payload = _llm_triage_failure_payload(
            results,
            model=model,
            reasoning_effort=reasoning_effort,
            batch_size=batch_size,
            phase="cli_llm_triage",
            error=f"{type(exc).__name__}: {exc}",
        )

    try:
        _write_llm_triage_output(
            payload,
            _resolve_llm_triage_output_path(args, runtime),
            args.quiet,
        )
    except Exception as exc:
        print_console(
            (
                "[yellow]LLM triage output was not written due to error: "
                f"{escape(str(exc))}[/yellow]"
            ),
            args.quiet,
        )
    return payload


def _llm_triage_failure_payload(
    results: dict,
    *,
    model: str,
    reasoning_effort: str,
    batch_size: int,
    phase: str,
    error: str,
) -> dict:
    filtered_issues = []
    reviews = results.get("reviews") if isinstance(results, dict) else None
    if isinstance(reviews, list):
        next_id = 1
        for file_entry in reviews:
            if not isinstance(file_entry, dict):
                continue
            file_name = str(file_entry.get("file") or "")
            file_path = str(file_entry.get("file_path") or "")
            issues = file_entry.get("reviews")
            if not isinstance(issues, list):
                continue
            for issue in issues:
                if not isinstance(issue, dict):
                    continue
                issue_copy = copy.deepcopy(issue)
                issue_copy["id"] = f"F{next_id:03d}"
                issue_copy["priority"] = "p5"
                issue_copy["file"] = str(
                    issue_copy.get("file")
                    or issue_copy.get("primary_file")
                    or file_name
                )
                issue_copy["file_path"] = str(issue_copy.get("file_path") or file_path)
                issue_copy["llm_triage_reason"] = (
                    f"LLM triage failed during {phase}; filtered because "
                    "triage was requested and no positive triage decision was available."
                )
                issue_copy["llm_triage_exploitability"] = (
                    "Not assessed because LLM triage failed."
                )
                issue_copy["llm_triage_duplicate_of"] = None
                issue_copy["llm_triage_failure_phase"] = phase
                issue_copy["llm_triage_error"] = error
                issue_copy["llm_triage_filtered"] = True
                issue_copy["llm_triage_keep"] = False
                filtered_issues.append(issue_copy)
                next_id += 1

    return {
        "summary": {
            "schema_version": 1,
            "model": model,
            "reasoning_effort": reasoning_effort,
            "batch_size": batch_size,
            "total_input_findings": len(filtered_issues),
            "kept_findings": 0,
            "kept_input_findings": 0,
            "filtered_findings": len(filtered_issues),
            "additional_findings": 0,
            "omitted_findings": 0,
            "threat_model_provided": False,
            "dynamic_repo_search_rounds": 0,
            "dynamic_repo_search_requests": 0,
            "dynamic_repo_search_matches": 0,
            "dynamic_repo_search_insights": 0,
            "errors": [{"phase": phase, "error": error}],
        },
        "issues": [],
        "filtered_issues": filtered_issues,
        "dynamic_repo_checks": [],
        "dynamic_repo_insights": [],
    }


def _review_results_from_llm_triage_payload(results: dict, payload: dict | None) -> dict:
    if not isinstance(payload, dict):
        return results
    issues = payload.get("issues")
    if not isinstance(issues, list):
        return results

    grouped: dict[str, dict] = {}
    original_reviews = results.get("reviews") if isinstance(results, dict) else None
    if isinstance(original_reviews, list):
        for file_entry in original_reviews:
            if not isinstance(file_entry, dict):
                continue
            file_name = str(file_entry.get("file") or file_entry.get("file_path") or "")
            file_path = str(file_entry.get("file_path") or "")
            key = file_name or file_path
            if not key:
                continue
            grouped.setdefault(
                key,
                {
                    "file": file_name,
                    "file_path": file_path,
                    "reviews": [],
                    "llm_triage_filtered_reviews": [],
                },
            )

    def _entry_for_issue(issue_copy: dict) -> dict:
        file_name = str(
            issue_copy.get("file")
            or issue_copy.get("primary_file")
            or issue_copy.get("file_path")
            or "UNKNOWN FILE"
        )
        file_path = str(issue_copy.get("file_path") or "")
        key = file_name or file_path
        entry = grouped.setdefault(
            key,
            {
                "file": file_name,
                "file_path": file_path,
                "reviews": [],
                "llm_triage_filtered_reviews": [],
            },
        )
        if not entry.get("file") and file_name:
            entry["file"] = file_name
        if not entry.get("file_path") and file_path:
            entry["file_path"] = file_path
        entry.setdefault("llm_triage_filtered_reviews", [])
        return entry

    for issue in issues:
        if not isinstance(issue, dict):
            continue
        issue_copy = copy.deepcopy(issue)
        _entry_for_issue(issue_copy)["reviews"].append(issue_copy)

    filtered_issues = payload.get("filtered_issues")
    if isinstance(filtered_issues, list):
        for issue in filtered_issues:
            if not isinstance(issue, dict):
                continue
            issue_copy = copy.deepcopy(issue)
            _entry_for_issue(issue_copy)["llm_triage_filtered_reviews"].append(
                issue_copy
            )

    final_results = copy.deepcopy(results) if isinstance(results, dict) else {}
    llm_triage_summary = copy.deepcopy(payload.get("summary") or {})
    dynamic_repo_checks = payload.get("dynamic_repo_checks")
    copied_dynamic_repo_checks = copy.deepcopy(
        dynamic_repo_checks if isinstance(dynamic_repo_checks, list) else []
    )
    dynamic_repo_insights = payload.get("dynamic_repo_insights")
    copied_dynamic_repo_insights = copy.deepcopy(
        dynamic_repo_insights if isinstance(dynamic_repo_insights, list) else []
    )
    final_reviews = [
        entry
        for entry in grouped.values()
        if entry.get("reviews") or entry.get("llm_triage_filtered_reviews")
    ]
    for entry in final_reviews:
        entry["llm_triage_summary"] = copy.deepcopy(llm_triage_summary)
        entry_ids = _file_entry_finding_ids(entry)
        entry["dynamic_repo_checks"] = _dynamic_records_for_entry(
            copied_dynamic_repo_checks,
            entry_ids,
        )
        entry["dynamic_repo_insights"] = _dynamic_records_for_entry(
            copied_dynamic_repo_insights,
            entry_ids,
        )

    final_results["reviews"] = final_reviews
    final_results["llm_triage_summary"] = llm_triage_summary
    final_results["llm_triage_filtered_issues"] = copy.deepcopy(
        filtered_issues if isinstance(filtered_issues, list) else []
    )
    final_results["dynamic_repo_checks"] = copied_dynamic_repo_checks
    final_results["dynamic_repo_insights"] = copied_dynamic_repo_insights
    return final_results


def _file_entry_finding_ids(entry: dict) -> set[str]:
    finding_ids: set[str] = set()
    for key in ("reviews", "llm_triage_filtered_reviews"):
        issues = entry.get(key)
        if not isinstance(issues, list):
            continue
        for issue in issues:
            if not isinstance(issue, dict):
                continue
            finding_id = str(issue.get("id") or "").strip()
            if finding_id:
                finding_ids.add(finding_id)
    return finding_ids


def _dynamic_records_for_entry(records: list[dict], finding_ids: set[str]) -> list[dict]:
    if not finding_ids:
        return []
    selected = []
    for record in records:
        if not isinstance(record, dict):
            continue
        raw_ids = record.get("finding_ids")
        if isinstance(raw_ids, str):
            record_ids = {raw_ids}
        elif isinstance(raw_ids, (list, tuple, set)):
            record_ids = {str(item) for item in raw_ids if str(item or "").strip()}
        else:
            record_ids = set()
        if not record_ids or record_ids & finding_ids:
            selected.append(copy.deepcopy(record))
    return selected


def _finalize_review_output(engine, results, args, runtime: CommandRuntime):
    llm_triage_payload = _run_llm_triage_if_requested(engine, results, args, runtime)
    final_results = _review_results_from_llm_triage_payload(
        results, llm_triage_payload
    )
    pretty_print_reviews(final_results, args.quiet)
    sarif_payload = _build_triaged_sarif_payload(engine, final_results, args, runtime)
    save_output(
        args.output_file,
        final_results,
        args.quiet,
        sarif_payload=sarif_payload,
    )
