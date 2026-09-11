"""Intel inference benchmark for the bimanual dinner-table pipeline.

Measures, per OpenVINO device (CPU, Intel iGPU, NPU when present):
  * the language planner (Qwen2.5-1.5B-Instruct INT4 via OpenVINO GenAI):
    load/compile time, time-to-first-token, time-per-output-token, throughput
    and end-to-end plan latency over a fixed instruction set;
  * every visuomotor policy IR found in ``models/policy/*.xml`` (e.g. the ACT
    policy exported in FP32 and INT8): single-call latency mean / p50 / p95 and
    calls per second, with zero-filled inputs of the model's own shapes.

Writes ``out/benchmark.json`` and a Markdown table ``out/benchmark.md``.

Run:  .venv\\Scripts\\python.exe -m bench.benchmark --devices CPU GPU.0
"""
import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import openvino as ov

from planner.planner import MODEL_DIR, Planner

ROOT = Path(__file__).resolve().parent.parent
POLICY_DIR = ROOT / "models" / "policy"
OUT_DIR = ROOT / "out"

INSTRUCTIONS = [
    "Open the drawer, set the spoon, hand the fork from the left arm to the right arm, put the plate on "
    "the placemat, then hold the mug with the left arm and pour from the bottle with the right arm.",
    "Put the plate on the placemat.",
    "Take the spoon out of the drawer and set it on the table.",
    "Hold the cup with the right arm and pour water into it with the left arm.",
    "Pass the fork from the left hand to the right hand and set it down.",
]
POLICY_WARMUP = 10
POLICY_ITERS = 200


def percentile(values, q):
    return float(np.percentile(np.asarray(values), q))


def device_names(core):
    return {d: core.get_property(d, "FULL_DEVICE_NAME") for d in core.available_devices}


def bench_planner(device):
    planner = Planner(MODEL_DIR, device=device)
    import openvino_genai as ov_genai

    latencies, ttft, tpot, throughput, valid = [], [], [], [], 0
    for instruction in INSTRUCTIONS:
        started = time.perf_counter()
        plan, info = planner.plan(instruction)
        latencies.append(time.perf_counter() - started)
        valid += info["source"] != "keyword_fallback"
    # Token-level metrics from one direct generation of the longest prompt.
    prompt = planner._prompt(INSTRUCTIONS[0], None)
    with_metrics = planner.pipe.generate([prompt], planner.config)
    metrics = with_metrics.perf_metrics
    ttft.append(metrics.get_ttft().mean)
    tpot.append(metrics.get_tpot().mean)
    throughput.append(metrics.get_throughput().mean)
    del ov_genai
    return {
        "model": "Qwen2.5-1.5B-Instruct INT4 (OpenVINO IR)",
        "device": device,
        "load_s": round(planner.load_s, 2),
        "plan_latency_mean_s": round(statistics.mean(latencies), 3),
        "plan_latency_p95_s": round(percentile(latencies, 95), 3),
        "ttft_ms": round(ttft[0], 1),
        "tpot_ms": round(tpot[0], 1),
        "tokens_per_s": round(throughput[0], 1),
        "llm_plans_valid": f"{valid}/{len(INSTRUCTIONS)}",
    }


def _zero_inputs(model):
    feeds = {}
    for port in model.inputs:
        shape = [d.get_length() if d.is_static else 1 for d in port.get_partial_shape()]
        dtype = port.get_element_type().to_dtype()
        feeds[port.any_name] = np.zeros(shape, dtype=dtype)
    return feeds


def bench_policy(core, xml_path, device):
    model = core.read_model(xml_path)
    meta_path = xml_path.with_suffix(".json")
    if meta_path.exists():
        shapes = json.loads(meta_path.read_text(encoding="utf-8")).get("input_shapes")
        if shapes:  # pin the batch-1 shapes the policy runs with
            model.reshape({port: ov.PartialShape(shape) for port, shape in zip(model.inputs, shapes)})
    started = time.perf_counter()
    compiled = core.compile_model(model, device, {"PERFORMANCE_HINT": "LATENCY"})
    compile_s = time.perf_counter() - started
    request = compiled.create_infer_request()
    feeds = _zero_inputs(model)
    for _ in range(POLICY_WARMUP):
        request.infer(feeds)
    times = []
    for _ in range(POLICY_ITERS):
        started = time.perf_counter()
        request.infer(feeds)
        times.append((time.perf_counter() - started) * 1000)
    return {
        "model": xml_path.stem,
        "device": device,
        "compile_s": round(compile_s, 2),
        "latency_mean_ms": round(statistics.mean(times), 3),
        "latency_p50_ms": round(percentile(times, 50), 3),
        "latency_p95_ms": round(percentile(times, 95), 3),
        "calls_per_s": round(1000 / statistics.mean(times), 1),
    }


def to_markdown(report):
    lines = [f"# Intel inference benchmark", "",
             f"Host: {report['host']['cpu']} · OpenVINO {report['host']['openvino']}", "",
             "Devices: " + ", ".join(f"`{k}` {v}" for k, v in report["host"]["devices"].items()), ""]
    if report["planner"]:
        lines += ["## Language planner", "",
                  "| Device | Load (s) | Plan latency mean / p95 (s) | TTFT (ms) | TPOT (ms) | Tokens/s | Valid LLM plans |",
                  "|---|---|---|---|---|---|---|"]
        for r in report["planner"]:
            lines.append(f"| {r['device']} | {r['load_s']} | {r['plan_latency_mean_s']} / {r['plan_latency_p95_s']} | "
                         f"{r['ttft_ms']} | {r['tpot_ms']} | {r['tokens_per_s']} | {r['llm_plans_valid']} |")
        lines.append("")
    if report["policy"]:
        lines += ["## Visuomotor policy", "",
                  "| Model | Device | Compile (s) | Latency mean / p50 / p95 (ms) | Calls/s |", "|---|---|---|---|---|"]
        for r in report["policy"]:
            lines.append(f"| {r['model']} | {r['device']} | {r['compile_s']} | {r['latency_mean_ms']} / "
                         f"{r['latency_p50_ms']} / {r['latency_p95_ms']} | {r['calls_per_s']} |")
        lines.append("")
    for note in report["notes"]:
        lines.append(f"- {note}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Benchmark planner and policy inference on Intel devices.")
    parser.add_argument("--devices", nargs="+", default=None, help="OpenVINO devices (default: all Intel ones)")
    parser.add_argument("--skip-planner", action="store_true")
    parser.add_argument("--policy-dir", type=Path, default=POLICY_DIR,
                        help="directory with exported policy IRs (*.xml)")
    args = parser.parse_args()
    policy_dir = args.policy_dir

    core = ov.Core()
    names = device_names(core)
    devices = args.devices or [d for d, name in names.items() if "NVIDIA" not in name]
    report = {
        "host": {"cpu": platform.processor() or names.get("CPU", "?"), "openvino": ov.__version__,
                 "devices": names},
        "planner": [], "policy": [], "notes": [],
    }
    if "Core(TM) Ultra" not in names.get("CPU", ""):
        report["notes"].append("Host CPU is not an Intel Core Ultra; numbers are from "
                               f"{names.get('CPU')} and its integrated GPU. Re-run this script on a Core Ultra "
                               "Series 2/3 machine to reproduce the target-hardware figures.")
    for device in devices:
        if not args.skip_planner:
            print(f"planner on {device} ...")
            try:
                report["planner"].append(bench_planner(device))
            except RuntimeError as exc:
                report["notes"].append(f"planner on {device} failed: {exc}")
        for xml_path in sorted(policy_dir.glob("*.xml")):
            print(f"policy {xml_path.stem} on {device} ...")
            try:
                report["policy"].append(bench_policy(core, xml_path, device))
            except RuntimeError as exc:
                report["notes"].append(f"policy {xml_path.stem} on {device} failed: {exc}")
    if not policy_dir.exists() or not any(policy_dir.glob("*.xml")):
        report["notes"].append(f"no policy IR in {policy_dir}; export one with policy/export_openvino.py")

    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "benchmark.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    markdown = to_markdown(report)
    (OUT_DIR / "benchmark.md").write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
