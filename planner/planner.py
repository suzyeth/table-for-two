"""Natural-language instruction -> bimanual subtask plan.

A small instruction-tuned LLM (Qwen2.5-1.5B-Instruct, INT4, OpenVINO IR) runs
through OpenVINO GenAI. Instead of verbose JSON (which a 1.5B model tends to
break) it writes a compact plan language, one stage per line:

    open_drawer left ; pick_place right mug
    pick_place left fork
    handoff spoon left right

The parser turns that into the JSON plan ``sim.task.Executor`` runs. Every plan
passes two checks before execution:
  * syntax  - known skills, arms and objects, one subtask per arm per stage;
  * physics - a small simulation of what each hand holds (no pouring without
    the bottle in hand and the mug set down in its place, no placing something
    that is not held, utensils only after the drawer is open, both hands empty
    at the end).
Errors are fed back to the model for one retry; remaining bad steps are
dropped; if nothing sensible is left, a keyword planner builds the plan.

Run:  .venv\\Scripts\\python.exe -m planner.planner "Open the drawer and set the plate" --device CPU
"""
import argparse
import json
import re
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov"
MODEL_DIR = ROOT / "models" / "qwen2.5-1.5b-instruct-int4-ov"

ARMS = ("left", "right")
OBJECTS = ("plate", "spoon", "fork", "mug", "bottle")
UTENSILS = ("spoon", "fork")
MAX_NEW_TOKENS = 160
MAX_ATTEMPTS = 2

SYNONYMS = {
    "cup": "mug", "glass": "mug", "mugs": "mug", "cups": "mug",
    "dish": "plate", "plates": "plate",
    "spoons": "spoon", "forks": "fork",
    "jug": "bottle", "pitcher": "bottle", "bottles": "bottle",
    "l": "left", "r": "right",
}

# skill -> ordered argument names in the compact plan language
SKILL_ARGS = {
    "open_drawer": ("arm",),
    "pick_place": ("arm", "object"),
    "handoff": ("object", "giver", "receiver"),
    "pick_hold": ("arm", "object"),
    "pick_lift": ("arm", "object"),
    "pour": ("arm", "into"),
    "place": ("arm", "object"),
    "return": ("arm", "object"),
    "home": ("arm",),
}

SYSTEM_PROMPT = """You plan actions for two robot arms, left and right, that set a dinner table.
Write the plan one stage per line. Subtasks on the same line run at the same time and are separated
by " ; ", with at most one subtask per arm. Use only these subtasks (ARM is left or right):
  open_drawer ARM
  pick_place ARM OBJECT        (OBJECT: plate, spoon, fork or mug; picks it up AND puts it in its place)
  handoff OBJECT GIVER RECEIVER  (spoon or fork; the receiver puts it in place; alone on its line)
  pick_lift ARM bottle
  pour ARM mug
  return ARM bottle
  home ARM
Rules:
- A cup or glass is the mug.
- The spoon and fork start inside the closed drawer, so open_drawer comes before them.
- pick_place already puts the object down: never add place after it.
- To pour, the mug must already stand in its place (pick_place ARM mug); then, with one arm,
  "pick_lift ARM bottle", "pour ARM mug", "return ARM bottle", each on its own line.
- Use the arm the user names. Otherwise left handles the drawer, fork and plate, right handles the mug
  and the bottle, and the spoon goes from left to right with a handoff.
- Only include what the instruction asks for, skip steps listed as done, end with "home left ; home right".
Write only the plan lines, nothing else."""

EXAMPLE_INSTRUCTION = ("Open the drawer, put the mug in its place and the plate on the placemat, lay the "
                       "fork, pour the water from the bottle into the mug, then hand the spoon from the "
                       "left arm to the right arm.")
EXAMPLE_PLAN_TEXT = """open_drawer left ; pick_place right mug
pick_place left plate ; pick_lift right bottle
pick_place left fork ; pour right mug
return right bottle
handoff spoon left right
home left ; home right"""


class PlanValidationError(ValueError):
    pass


def _normalise(token):
    return SYNONYMS.get(token, token)


def parse_plan_text(text):
    """Parse the compact plan language into the executor's JSON plan (syntax-checked)."""
    plan = []
    for line in text.strip().splitlines():
        line = line.strip().strip("`").strip()
        if not line or line.startswith("#"):
            continue
        stage = []
        for chunk in line.split(";"):
            tokens = [_normalise(t) for t in chunk.lower().replace(",", " ").split()]
            if not tokens:
                continue
            skill = tokens[0]
            if skill not in SKILL_ARGS:
                raise PlanValidationError(f"unknown skill '{skill}' in line: {line}")
            names = SKILL_ARGS[skill]
            if len(tokens) - 1 != len(names):
                raise PlanValidationError(f"'{skill}' needs {len(names)} argument(s): {line}")
            if skill == "pour" and tokens[2] == "bottle":
                tokens[2] = "mug"  # the model sometimes names what it pours from, not into
            stage.append({"skill": skill, **dict(zip(names, tokens[1:]))})
        if stage:
            plan.append(stage)
    return validate(plan)


def validate(plan):
    """Syntax check: raise PlanValidationError unless every subtask is well formed."""
    if not isinstance(plan, list) or not plan:
        raise PlanValidationError("plan must be a non-empty list of stages")
    for index, stage in enumerate(plan):
        if not isinstance(stage, list) or not stage:
            raise PlanValidationError(f"stage {index} must be a non-empty list")
        arms_used = []
        for sub in stage:
            if not isinstance(sub, dict) or sub.get("skill") not in SKILL_ARGS:
                raise PlanValidationError(f"stage {index}: unknown subtask {sub}")
            if sub["skill"] == "handoff":
                if len(stage) != 1:
                    raise PlanValidationError(f"stage {index}: handoff must be alone in its stage")
                if sub.get("giver") not in ARMS or sub.get("receiver") not in ARMS or sub["giver"] == sub["receiver"]:
                    raise PlanValidationError(f"stage {index}: handoff needs two different arms")
                if sub.get("object") not in UTENSILS:
                    raise PlanValidationError(f"stage {index}: only the spoon or fork can be handed off")
                continue
            if sub.get("arm") not in ARMS:
                raise PlanValidationError(f"stage {index}: subtask needs arm left/right: {sub}")
            arms_used.append(sub["arm"])
            for key in ("object", "into"):
                if key in sub and sub[key] not in OBJECTS:
                    raise PlanValidationError(f"stage {index}: unknown object '{sub[key]}'")
        if len(arms_used) != len(set(arms_used)):
            raise PlanValidationError(f"stage {index}: an arm has two subtasks")
    return plan


def check_semantics(plan, scene_state=None):
    """Simulate what each hand holds; return [(stage, subtask_index or None, message)]."""
    done = set((scene_state or {}).get("done", []))
    drawer_open = "drawer_open" in done
    placed = {item.removesuffix("_placed") for item in done if item.endswith("_placed")}
    holding = {arm: None for arm in ARMS}
    problems = []
    for s, stage in enumerate(plan):
        for k, sub in enumerate(stage):
            skill, obj = sub["skill"], sub.get("object")
            arms = [sub["giver"], sub["receiver"]] if skill == "handoff" else [sub["arm"]]
            if skill in ("open_drawer", "pick_place", "handoff", "pick_hold", "pick_lift"):
                busy = [a for a in arms if holding[a]]
                if busy:
                    problems.append((s, k, f"the {busy[0]} arm is still holding the {holding[busy[0]]}"))
                    continue
            if skill in ("pick_place", "handoff") and obj in UTENSILS and not drawer_open:
                problems.append((s, k, f"open the drawer before taking the {obj}"))
                continue
            if skill == "open_drawer":
                if drawer_open:
                    problems.append((s, k, "the drawer is already open"))
                drawer_open = True
            elif skill == "pick_place":
                if obj not in ("plate", "spoon", "fork", "mug"):
                    problems.append((s, k, "pick_place only moves the plate, spoon, fork or mug"))
                else:
                    placed.add(obj)
            elif skill == "handoff":
                placed.add(obj)
            elif skill == "pick_hold":
                if obj != "mug":
                    problems.append((s, k, "pick_hold is only for the mug"))
                else:
                    holding[sub["arm"]] = "mug"
                    placed.discard("mug")
            elif skill == "pick_lift":
                if obj != "bottle":
                    problems.append((s, k, "pick_lift is only for the bottle"))
                else:
                    holding[sub["arm"]] = "bottle"
            elif skill == "pour":
                if sub.get("into") != "mug":
                    problems.append((s, k, "pour only goes into the mug"))
                elif holding[sub["arm"]] != "bottle":
                    problems.append((s, k, f"pour needs the bottle in the {sub['arm']} arm first"))
                elif "mug" not in placed:
                    problems.append((s, k, "set the mug down in its place before pouring"))
            elif skill in ("place", "return"):
                if holding[sub["arm"]] != obj:
                    problems.append((s, k, f"the {sub['arm']} arm is not holding the {obj}"))
                else:
                    holding[sub["arm"]] = None
                    if skill == "place":
                        placed.add(obj)
            elif skill == "home" and holding[sub["arm"]]:
                problems.append((s, k, f"the {sub['arm']} arm goes home still holding the {holding[sub['arm']]}"))
    for arm, obj in holding.items():
        if obj:
            problems.append((len(plan), None, f"the plan ends with the {arm} arm holding the {obj}"))
    return problems


def repair(plan, scene_state=None):
    """Drop subtasks flagged by check_semantics until the plan is clean; None if that fails."""
    plan = [list(stage) for stage in plan]
    for _ in range(len(plan) * 2 + 2):
        problems = check_semantics(plan, scene_state)
        if not problems:
            return [stage for stage in plan if stage] or None
        located = [(s, k) for s, k, _ in problems if k is not None]
        if not located:
            return None
        s, k = located[0]
        del plan[s][k]
        plan = [stage for stage in plan if stage]
    return None


EXAMPLE_PLAN = parse_plan_text(EXAMPLE_PLAN_TEXT)


def _arm_named_for(text, words):
    """First 'left'/'right' mentioned in the same clause as any of ``words``."""
    for clause in re.split(r"[,.;]|\bthen\b|\band\b", text):
        if any(re.search(rf"\b{w}", clause) for w in words):
            match = re.search(r"\b(left|right)\b", clause)
            if match:
                return match.group(1)
    return None


def keyword_plan(instruction, scene_state=None):
    """Deterministic fallback: canonical plan restricted to what the instruction mentions."""
    text = instruction.lower()
    done = set((scene_state or {}).get("done", []))

    def wants(*words):
        return any(re.search(rf"\b{w}", text) for w in words)

    plan = []
    utensils = [u for u in UTENSILS if wants(u, "cutlery", "utensil") and f"{u}_placed" not in done]
    if (wants("drawer") or utensils) and "drawer_open" not in done:
        plan.append([{"skill": "open_drawer", "arm": _arm_named_for(text, ("drawer",)) or "left"}])
    mug_wanted = wants("mug", "cup", "glass", "pour", "water") and "mug_placed" not in done
    if mug_wanted:
        plan.append([{"skill": "pick_place", "arm": _arm_named_for(text, ("mug", "cup", "glass")) or "right",
                      "object": "mug"}])
    for utensil in utensils:
        arm = _arm_named_for(text, (utensil,))
        if arm and not wants("hand", "pass", "give"):
            plan.append([{"skill": "pick_place", "arm": arm, "object": utensil}])
        elif utensil == "spoon" or wants("hand", "pass", "give"):
            giver = arm or "left"
            receiver = "right" if giver == "left" else "left"
            plan.append([{"skill": "handoff", "object": utensil, "giver": giver, "receiver": receiver}])
        else:
            plan.append([{"skill": "pick_place", "arm": "left", "object": utensil}])
    if wants("plate", "dish") and "plate_placed" not in done:
        plan.append([{"skill": "pick_place", "arm": _arm_named_for(text, ("plate", "dish")) or "left",
                      "object": "plate"}])
    if wants("pour", "water"):
        pour_arm = _arm_named_for(text, ("pour", "bottle", "water")) or "right"
        plan += [
            [{"skill": "pick_lift", "arm": pour_arm, "object": "bottle"}],
            [{"skill": "pour", "arm": pour_arm, "into": "mug"}],
            [{"skill": "return", "arm": pour_arm, "object": "bottle"}],
        ]
    plan.append([{"skill": "home", "arm": "left"}, {"skill": "home", "arm": "right"}])
    return validate(plan)


class Planner:
    """LLM planner on OpenVINO GenAI with validation, one retry, repair and a keyword fallback."""

    def __init__(self, model_dir=MODEL_DIR, device="CPU"):
        import openvino_genai as ov_genai

        if not Path(model_dir).exists():
            raise FileNotFoundError(f"{model_dir} not found - run: python -m planner.download_model")
        self.device = device
        started = time.perf_counter()
        self.pipe = ov_genai.LLMPipeline(str(model_dir), device)
        self.load_s = time.perf_counter() - started
        self.config = ov_genai.GenerationConfig()
        self.config.max_new_tokens = MAX_NEW_TOKENS
        self.config.do_sample = False

    def _prompt(self, instruction, scene_state, feedback=None):
        state = json.dumps(scene_state or {"done": []})
        prompt = (f"Example instruction: {EXAMPLE_INSTRUCTION}\nExample plan:\n{EXAMPLE_PLAN_TEXT}\n\n"
                  f"Scene state: {state}\nInstruction: {instruction}\nPlan:")
        if feedback:
            prompt += f"\n(Your previous plan was rejected: {feedback}. Write a corrected plan.)"
        return prompt

    def _generate(self, prompt):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)  # start_chat API still works in 2026.x
            self.pipe.start_chat(SYSTEM_PROMPT)
            try:
                return str(self.pipe.generate(prompt, self.config))
            finally:
                self.pipe.finish_chat()

    def plan(self, instruction, scene_state=None):
        """Return (plan, info); info records latency, attempts, errors and where the plan came from."""
        info = {"device": self.device, "attempts": 0, "latency_s": 0.0, "source": "llm", "errors": []}
        feedback, last_parsed = None, None
        for _ in range(MAX_ATTEMPTS):
            info["attempts"] += 1
            started = time.perf_counter()
            raw = self._generate(self._prompt(instruction, scene_state, feedback))
            info["latency_s"] += time.perf_counter() - started
            info["raw"] = raw
            try:
                parsed = parse_plan_text(raw)
            except PlanValidationError as exc:
                feedback = str(exc)
                info["errors"].append(feedback)
                continue
            problems = check_semantics(parsed, scene_state)
            if not problems:
                return parsed, info
            last_parsed = parsed
            feedback = "; ".join(message for _, _, message in problems[:3])
            info["errors"].append(feedback)
        if last_parsed is not None:
            repaired = repair(last_parsed, scene_state)
            if repaired:
                info["source"] = "llm_repaired"
                return repaired, info
        info["source"] = "keyword_fallback"
        return keyword_plan(instruction, scene_state), info


def main():
    parser = argparse.ArgumentParser(description="Plan a dinner-table instruction with the OpenVINO LLM.")
    parser.add_argument("instruction", nargs="?", default=EXAMPLE_INSTRUCTION)
    parser.add_argument("--device", default="CPU", help="OpenVINO device: CPU, GPU or NPU")
    parser.add_argument("--out", type=Path, help="write the plan JSON here")
    args = parser.parse_args()

    planner = Planner(device=args.device)
    plan, info = planner.plan(args.instruction)
    print(info.get("raw", "").strip())
    print(json.dumps(plan))
    print(f"source={info['source']} attempts={info['attempts']} latency={info['latency_s']:.2f}s "
          f"load={planner.load_s:.1f}s device={info['device']} errors={info['errors']}")
    if args.out:
        args.out.write_text(json.dumps(plan, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
