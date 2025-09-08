"""
Agentic API Testing & Monitoring CLI (single-file starter)

Features implemented in this starter:
- Typer-based CLI with commands: run, monitor, gen-tests (LLM-backed with OpenAI)
- Load collections from YAML (requests, variables, assertions)
- Async HTTP requests using httpx
- Basic assertions (status, jsonpath, header, latency)
- Collects metrics (p50/p95/avg, error rate)
- Simple anomaly detection (z-score on p95 over stored history file)
- LLM integration with OpenAI (needs OPENAI_API_KEY)
- Export run results to JSON report

Usage:
    pip install -r requirements.txt
    python main.py --help

This is a starting point — extend with Prometheus, OpenTelemetry, richer AI agenting, self-healing, load testing, mocking, etc.
"""

from __future__ import annotations
import os
import json
import time
import math
import asyncio
import statistics
from typing import Any, Dict, List, Optional

import httpx
import typer
import yaml
from pydantic import BaseModel, Field
from rich.console import Console
from rich.table import Table
# -------------------------
# Globals & Utilities
# -------------------------

# app = typer.Typer(help="Agentic CLI for API testing & monitoring")
# console = Console()




app = typer.Typer(help="🤖 APIgentMan – Your Agentic API Tester")

console = Console()

BANNER = r"""
    █████╗ ██████╗ ██╗ ██████╗ ███████╗███╗   ██╗████████╗
   ██╔══██╗██╔══██╗██║██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝
   ███████║██████╔╝██║██║  ███╗█████╗  ██╔██╗ ██║   ██║   
   ██╔══██║██╔═══╝ ██║██║   ██║██╔══╝  ██║╚██╗██║   ██║   
   ██║  ██║██║     ██║╚██████╔╝███████╗██║ ╚████║   ██║   
   ╚═╝  ╚═╝╚═╝     ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝   
           🤖 APIgentMan – Agentic API Tester 📬
"""

@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    if ctx.invoked_subcommand is None:
        console.print(BANNER, style="bold green")
        typer.echo(ctx.get_help())


DEFAULT_HISTORY_FILE = "agentic_history.json"

# -------------------------
# Models
# -------------------------

class Assertion(BaseModel):
    type: str  # status | jsonpath | header | latency
    expect: Any
    path: Optional[str] = None  # for jsonpath or header

class RequestDef(BaseModel):
    name: str
    method: str = "GET"
    url: str
    headers: Dict[str, str] = Field(default_factory=dict)
    params: Dict[str, Any] = Field(default_factory=dict)
    json: Optional[Any] = None
    data: Optional[Any] = None
    assertions: List[Assertion] = Field(default_factory=list)

class Collection(BaseModel):
    name: str
    variables: Dict[str, Any] = Field(default_factory=dict)
    requests: List[RequestDef]

class RunResult(BaseModel):
    request_name: str
    url: str
    method: str
    status: int
    latency_ms: float
    success: bool
    errors: List[str] = Field(default_factory=list)
    timestamp: float
    response_sample: Optional[Any] = None

# -------------------------
# YAML loader
# -------------------------

def load_collection(path: str) -> Collection:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    col = Collection(**raw)
    return col

# -------------------------
# Assertion helpers
# -------------------------

def apply_assertions(result_json: Any, resp_headers: Dict[str, Any], status: int, latency_ms: float, assertions: List[Assertion]) -> List[str]:
    errors: List[str] = []
    for a in assertions:
        if a.type == "status":
            if status != int(a.expect):
                errors.append(f"status: expected {a.expect} got {status}")
        elif a.type == "header":
            key = a.path
            if not key:
                errors.append("header assertion missing path")
                continue
            actual = resp_headers.get(key)
            if str(actual) != str(a.expect):
                errors.append(f"header {key}: expected {a.expect} got {actual}")
        elif a.type == "latency":
            threshold = float(a.expect)
            if latency_ms > threshold:
                errors.append(f"latency: expected <= {threshold}ms got {latency_ms}ms")
        elif a.type == "jsonpath":
            path = a.path or ""
            try:
                val = result_json
                for part in [p for p in path.split('.') if p]:
                    if isinstance(val, dict):
                        val = val.get(part)
                    elif isinstance(val, list):
                        idx = int(part)
                        val = val[idx]
                    else:
                        val = None
                if str(val) != str(a.expect):
                    errors.append(f"jsonpath {path}: expected {a.expect} got {val}")
            except Exception as e:
                errors.append(f"jsonpath {path}: error {e}")
        else:
            errors.append(f"unknown assertion type: {a.type}")
    return errors

# -------------------------
# Runner
# -------------------------

async def run_request(client: httpx.AsyncClient, r: RequestDef, variables: Dict[str, Any]) -> RunResult:
    url = r.url.format(**variables)
    headers = {k: v.format(**variables) for k, v in r.headers.items()}
    params = {k: (v.format(**variables) if isinstance(v, str) else v) for k, v in r.params.items()}
    json_body = r.json
    data_body = r.data

    start = time.perf_counter()
    try:
        resp = await client.request(r.method.upper(), url, headers=headers, params=params, json=json_body, data=data_body, timeout=30.0)
        latency_ms = (time.perf_counter() - start) * 1000.0
        success = resp.status_code < 400
        try:
            resp_json = resp.json()
        except Exception:
            resp_json = None
        errors = apply_assertions(resp_json, resp.headers, resp.status_code, latency_ms, r.assertions)
        success = success and (len(errors) == 0)
        rr = RunResult(
            request_name=r.name,
            url=url,
            method=r.method.upper(),
            status=resp.status_code,
            latency_ms=latency_ms,
            success=success,
            errors=errors,
            timestamp=time.time(),
            response_sample=(resp_json if resp_json is not None else resp.text[:200])
        )
        return rr
    except Exception as e:
        latency_ms = (time.perf_counter() - start) * 1000.0
        return RunResult(
            request_name=r.name,
            url=url,
            method=r.method.upper(),
            status=0,
            latency_ms=latency_ms,
            success=False,
            errors=[f"exception: {e}"],
            timestamp=time.time(),
            response_sample=None
        )

async def run_collection_async(collection: Collection, concurrency: int = 5) -> List[RunResult]:
    results: List[RunResult] = []
    limits = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient() as client:
        async def worker(r: RequestDef):
            async with limits:
                return await run_request(client, r, collection.variables)

        tasks = [asyncio.create_task(worker(r)) for r in collection.requests]
        for t in asyncio.as_completed(tasks):
            res = await t
            results.append(res)
    return results

# -------------------------
# Simple Anomaly Detector
# -------------------------

def load_history(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {"runs": []}
    with open(path, "r") as f:
        return json.load(f)

def save_history(path: str, doc: Dict[str, Any]):
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)

def detect_anomaly(history_path: str, current_results: List[RunResult]) -> Dict[str, Any]:
    history = load_history(history_path)
    past_p95: Dict[str, List[float]] = {}
    for run in history.get("runs", []):
        for rr in run.get("results", []):
            name = rr["request_name"]
            past_p95.setdefault(name, []).append(rr["latency_p95"]) if rr.get("latency_p95") else None
    anomalies = []
    summary = {}
    for name in {r.request_name for r in current_results}:
        latencies = [r.latency_ms for r in current_results if r.request_name == name]
        if not latencies:
            continue
        p50 = statistics.median(latencies)
        p95 = percentile(latencies, 95)
        error_rate = len([r for r in current_results if r.request_name == name and not r.success]) / len(latencies)
        summary[name] = {"p50": p50, "p95": p95, "error_rate": error_rate}
        past = past_p95.get(name, [])
        if len(past) >= 5:
            mean = statistics.mean(past)
            stdev = statistics.pstdev(past) if statistics.pstdev(past) > 0 else 1.0
            z = (p95 - mean) / stdev
            if z > 3:
                anomalies.append({"request": name, "z": z, "p95": p95, "mean_p95": mean})
    return {"summary": summary, "anomalies": anomalies, "history_len": len(history.get("runs", []))}

def percentile(data: List[float], pct: float) -> float:
    if not data:
        return 0.0
    data_sorted = sorted(data)
    k = (len(data_sorted)-1) * (pct/100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return data_sorted[int(k)]
    d0 = data_sorted[int(f)] * (c-k)
    d1 = data_sorted[int(c)] * (k-f)
    return d0 + d1

# -------------------------
# CLI Commands
# -------------------------

@app.command()
def run(collection: str = typer.Argument(..., help="Path to collection YAML"),
        concurrency: int = typer.Option(5, help="Parallel requests"),
        out: Optional[str] = typer.Option(None, help="Write JSON report to file")):
    col = load_collection(collection)
    r = asyncio.run(run_collection_async(col, concurrency=concurrency))
    table = Table(title=f"Run Results: {col.name}")
    table.add_column("Request")
    table.add_column("Status")
    table.add_column("Latency ms")
    table.add_column("Success")
    table.add_column("Errors")

    for rr in r:
        table.add_row(
            rr.request_name, str(rr.status),
            f"{rr.latency_ms:.2f}", str(rr.success),
            "; ".join(rr.errors) if rr.errors else ""
        )
    console.print(table)

    stats = compute_stats(r)
    report = {
        "collection": col.name, "timestamp": time.time(),
        "results": [rr.model_dump() for rr in r], "stats": stats
    }

    history = load_history(DEFAULT_HISTORY_FILE)
    history_entry = {"timestamp": time.time(), "results": []}
    for name, s in stats.items():
        history_entry["results"].append(
            {"request_name": name, "latency_p95": s["p95"],
            "p50": s["p50"],
            "avg": s["avg"],
            "error_rate": s["error_rate"]}
        )
    history.setdefault("runs", []).append(history_entry)
    save_history(DEFAULT_HISTORY_FILE, history)

    if out:
        with open(out, "w") as f:
            json.dump(report, f, indent=2)
        console.log(f"Wrote report to {out}")

    anom = detect_anomaly(DEFAULT_HISTORY_FILE, r)
    if anom.get("anomalies"):
        console.print("[bold red]Anomalies detected:[/bold red]")
        console.print(anom["anomalies"])

@app.command()
def monitor(collection: str = typer.Argument(..., help="Path to collection YAML"),
            interval: int = typer.Option(60, help="Seconds between runs")):
    col = load_collection(collection)
    console.log(f"Starting monitor for {col.name} every {interval}s. Ctrl-C to stop.")
    try:
        while True:
            r = asyncio.run(run_collection_async(col, concurrency=5))
            stats = compute_stats(r)
            console.log("Run summary:")
            for k,v in stats.items():
                console.log(
                    f"{k}: p50={v['p50']:.1f}ms p95={v['p95']:.1f}ms avg={v['avg']:.1f}ms errs={v['error_rate']:.2f}")
            history = load_history(DEFAULT_HISTORY_FILE)
            history.setdefault("runs", []).append(
                {
                    "timestamp": time.time(),
                    "results": [{"request_name": name,
                    "latency_p95": v['p95'],
                    "p50": v['p50'],
                    "avg": v['avg'], 
                    "error_rate": v['error_rate']} for name, v in stats.items()]
                }
            )
            save_history(DEFAULT_HISTORY_FILE, history)
            anom = detect_anomaly(DEFAULT_HISTORY_FILE, r)
            if anom.get("anomalies"):
                console.print("[red]Anomaly detected![/red]")
                console.print(anom["anomalies"])
            time.sleep(interval)
    except KeyboardInterrupt:
        console.log("Monitor stopped by user")

@app.command()
def gen_tests(openapi: Optional[str] = typer.Option(
    None, help="Path or URL to OpenAPI JSON/YAML"),
    prompt: Optional[str] = typer.Option(None, help="Short description of API")):
    console.log("Generating tests with LLM...")
    if openapi:
        console.log(f"Parsing OpenAPI at {openapi} (stub implementation)")
        console.print("TODO: implement OpenAPI parsing")
    elif prompt:
        result = call_llm_generate(prompt)
        console.print("[bold green]Generated test collection:[/bold green]")
        console.print(result)
    else:
        console.print("Provide --openapi or --prompt to generate tests")

@app.command()
def report(history_file: str = typer.Option(DEFAULT_HISTORY_FILE, help="History JSON"), out: str = typer.Option("report.json", help="Output file")):
    h = load_history(history_file)
    outdoc = {"generated_at": time.time(), "history_len": len(h.get("runs", [])), "per_request": {}}
    per = {}
    for run in h.get("runs", []):
        for r in run.get("results", []):
            name = r["request_name"]
            per.setdefault(name, []).append(r.get("latency_p95"))
    for name, arr in per.items():
        arr_clean = [v for v in arr if v is not None]
        outdoc["per_request"][name] = {"samples": len(arr_clean), "p95_mean": statistics.mean(arr_clean) if arr_clean else None}
    with open(out, "w") as f:
        json.dump(outdoc, f, indent=2)
    console.log(f"Wrote report: {out}")

# -------------------------
# Helpers
# -------------------------

def compute_stats(results: List[RunResult]) -> Dict[str, Dict[str, float]]:
    by_name: Dict[str, List[RunResult]] = {}
    for r in results:
        by_name.setdefault(r.request_name, []).append(r)
    out: Dict[str, Dict[str, float]] = {}
    for name, items in by_name.items():
        lat = [it.latency_ms for it in items]
        p50 = statistics.median(lat) if lat else 0.0
        p95 = percentile(lat, 95) if lat else 0.0
        avg = statistics.mean(lat) if lat else 0.0
        err = len([it for it in items if not it.success]) / len(items) if items else 0.0
        out[name] = {"p50": p50, "p95": p95, "avg": avg, "error_rate": err}
    return out

# -------------------------
# LLM Integration Hook
# -------------------------
def call_llm_generate(prompt: str) -> str:
    """Placeholder: calls an LLM to generate tests. Hook with OpenAI/Anthropic/HF.

    To enable: set OPENAI_API_KEY env var and implement an API call here.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "LLM not configured. Set OPENAI_API_KEY to enable."
    # Example (pseudocode):
    # import openai
    # openai.api_key = api_key
    # resp = openai.ChatCompletion.create(model="gpt-4o-mini", messages=[...])
    # return resp

    from groq import Groq

    client = Groq()
    completion = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
        {
            "role": "user",
            "content": f"{prompt}"
        }
        ],
        temperature=1,
        max_completion_tokens=8192,
        top_p=1,
        reasoning_effort="medium",
        stream=True,
        stop=None
    )
    res = ""
    for chunk in completion:
        print(chunk.choices[0].delta.content or "", end="")
        # res = res + chunk.choices[0].delta.content or ""


    return ""

# -------------------------
# Entry point
# -------------------------

if __name__ == "__main__":
    app()
