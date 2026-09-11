// Builds the 8-slide submission deck from deck_data.json.
// Run:  node build_deck.js   ->  table_for_two.pptx
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const data = JSON.parse(fs.readFileSync(path.join(__dirname, "deck_data.json"), "utf8"));
const OUT_DIR = path.join(__dirname, "..", "..", "out");
const img = (name) => path.join(OUT_DIR, name);

// Palette: SO-101 yellow is the single accent on a green-black / cool-grey sandwich.
const C = {
  dark: "1B2220", light: "F2F4F3", white: "FFFFFF", ink: "1B2220", muted: "5B6864",
  accent: "E5B20F", blue: "2C5A78", rule: "D5DCD9", good: "2F7A4F", soft: "FBF1CF",
};
const HEAD = "Cambria";
const BODY = "Calibri";
const W = 13.333;
const M = 0.6;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.title = data.title;

const pending = (value, fmt) => (value === null || value === undefined ? "pending" : fmt(value));

function title(slide, text, options = {}) {
  slide.addText(text, {
    x: M, y: 0.45, w: W - 2 * M, h: 0.8, fontFace: HEAD, fontSize: 34, bold: true,
    color: options.color || C.ink, margin: 0, isTextBox: true,
  });
}

function body(slide, runs, box) {
  slide.addText(runs, { fontFace: BODY, fontSize: 16, color: C.ink, valign: "top", margin: 0, isTextBox: true, ...box });
}

function bullets(items) {
  return items.map((text, i) => ({
    text, options: { bullet: true, breakLine: i < items.length - 1, paraSpaceAfter: 8 },
  }));
}

// 1 · Title -----------------------------------------------------------------
{
  const s = pres.addSlide();
  s.background = { color: C.dark };
  s.addImage({ path: img("cover.png"), x: 6.1, y: 0, w: 7.233, h: 7.5, sizing: { type: "cover", w: 7.233, h: 7.5 } });
  s.addText(data.title, { x: M, y: 1.9, w: 5.4, h: 1.2, fontFace: HEAD, fontSize: 54, bold: true, color: C.white, margin: 0, isTextBox: true });
  s.addText(data.subtitle, { x: M, y: 3.2, w: 5.2, h: 1.3, fontFace: BODY, fontSize: 20, color: "E4E9E7", margin: 0, isTextBox: true });
  s.addText(data.event, { x: M, y: 6.3, w: 5.2, h: 0.6, fontFace: BODY, fontSize: 12, color: C.accent, margin: 0, isTextBox: true });
  s.addNotes("One spoken sentence in, a fully set dinner table out. Two simulated SO-101 arms, a language planner and a visuomotor policy, all running on OpenVINO.");
}

// 2 · The task --------------------------------------------------------------
{
  const s = pres.addSlide();
  s.background = { color: C.light };
  title(s, "The brief: set a dinner table with two arms");
  body(s, bullets([
    "Open the drawer and take out the spoon and fork",
    "Hand the fork from one arm to the other",
    "Put the plate on the placemat",
    "Hold the mug with one arm while the other pours from the bottle",
    "Follow a natural-language instruction; see the scene through cameras",
    "Stay robust across 10 randomised scenes",
  ]), { x: M, y: 1.6, w: 5.6, h: 4.6 });
  s.addImage({ path: img("cover_alt.png"), x: 6.6, y: 1.6, w: 6.13, h: 3.45 });
  s.addText("MuJoCo · 2 × SO-101 · 12 position actuators · 3 cameras · 20 Hz control", {
    x: 6.6, y: 5.2, w: 6.13, h: 0.4, fontFace: BODY, fontSize: 12, color: C.muted, margin: 0, isTextBox: true,
  });
  s.addNotes("This is the brief's own scenario. We built the scene from scratch: the official SO-101 model twice, a cabinet with a sliding drawer, and the table-setting props.");
}

// 3 · Architecture ----------------------------------------------------------
{
  const s = pres.addSlide();
  s.background = { color: C.light };
  title(s, "How it works");
  const stages = [
    ["Speech or text", "Speechmatics transcribes a spoken instruction"],
    ["Planner", "Qwen2.5-1.5B INT4 on OpenVINO GenAI writes a checked plan"],
    ["Stage executor", "Subtask token for each stage; both arms run in parallel"],
    ["ACT policy", "4 cameras (2 scene + 2 wrist) + 12 joints → 12 joint targets at 10 Hz, OpenVINO INT8"],
    ["MuJoCo", "Dual SO-101 arms, contact-only grasps, drawer, plate, mug, bead water, utensils"],
  ];
  const boxW = 2.2;
  const boxY = 2.2;
  const boxH = 1.9;
  const gap = (W - 2 * M - stages.length * boxW) / (stages.length - 1);
  stages.forEach(([head, text], i) => {
    const x = M + i * (boxW + gap);
    const key = head === "ACT policy";
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x, y: boxY, w: boxW, h: boxH, rectRadius: 0.08, fill: { color: key ? C.soft : C.white },
      line: { color: key ? C.accent : C.rule, width: key ? 2.5 : 1 },
    });
    s.addText(head, { x: x + 0.18, y: boxY + 0.18, w: boxW - 0.36, h: 0.45, fontFace: HEAD, fontSize: 18, bold: true, color: C.ink, margin: 0, isTextBox: true });
    s.addText(text, { x: x + 0.18, y: boxY + 0.7, w: boxW - 0.36, h: boxH - 0.85, fontFace: BODY, fontSize: 14, color: C.muted, margin: 0, valign: "top", isTextBox: true });
    if (i < stages.length - 1) {
      s.addShape(pres.shapes.RIGHT_ARROW, { x: x + boxW + gap / 2 - 0.14, y: boxY + boxH / 2 - 0.15, w: 0.28, h: 0.3, fill: { color: C.muted }, line: { color: C.muted } });
    }
  });
  s.addText("Scripted IK skills generate the demonstrations and finish any stage the policy times out on (hybrid mode, reported separately).", {
    x: M, y: 4.6, w: W - 2 * M, h: 0.6, fontFace: BODY, fontSize: 15, italic: true, color: C.blue, margin: 0, isTextBox: true,
  });
  s.addNotes("Hierarchical by design: learning a minute-long bimanual task end to end in a week is not realistic, so language planning and visuomotor control are separate, and every model runs through OpenVINO.");
}

// 4 · Planner ---------------------------------------------------------------
{
  const s = pres.addSlide();
  s.background = { color: C.light };
  title(s, "A planner that cannot hand the robot a bad plan");
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: M, y: 1.6, w: 5.9, h: 3.1, rectRadius: 0.06, fill: { color: C.dark }, line: { color: C.dark } });
  s.addText([
    { text: "\"Pour a drink, then pass the spoon from the left arm to the right.\"", options: { color: C.accent, breakLine: true } },
    { text: " ", options: { breakLine: true } },
    { text: "open_drawer left ; pick_place right mug", options: { breakLine: true } },
    { text: "pick_lift right bottle", options: { breakLine: true } },
    { text: "pour right mug", options: { breakLine: true } },
    { text: "return right bottle", options: { breakLine: true } },
    { text: "handoff spoon left right", options: { breakLine: true } },
    { text: "home left ; home right" },
  ], { x: M + 0.3, y: 1.85, w: 5.4, h: 2.7, fontFace: "Courier New", fontSize: 14, color: "E4E9E7", margin: 0, valign: "top", isTextBox: true });
  s.addText("The model writes the compact plan; the parser turns it into the JSON the executor runs.", {
    x: M, y: 4.9, w: 5.9, h: 0.7, fontFace: BODY, fontSize: 14, color: C.muted, margin: 0, isTextBox: true,
  });
  const steps = [
    ["1", "Compact plan language", "One stage per line, ~60 tokens instead of ~400 of JSON a 1.5 B model breaks."],
    ["2", "Syntax + hand-state check", "Known skills and objects; simulate what each hand holds — no pouring without the bottle."],
    ["3", "Retry, repair, fall back", "Errors go back to the model once; bad steps are dropped; a keyword planner is last resort."],
  ];
  steps.forEach(([num, head, text], i) => {
    const y = 1.6 + i * 1.5;
    s.addShape(pres.shapes.OVAL, { x: 7.0, y, w: 0.55, h: 0.55, fill: { color: C.accent }, line: { color: C.accent } });
    s.addText(num, { x: 7.0, y, w: 0.55, h: 0.55, fontFace: HEAD, fontSize: 18, bold: true, color: C.ink, align: "center", valign: "middle", margin: 0, isTextBox: true });
    s.addText(head, { x: 7.75, y: y - 0.02, w: 5.0, h: 0.4, fontFace: HEAD, fontSize: 18, bold: true, color: C.ink, margin: 0, isTextBox: true });
    s.addText(text, { x: 7.75, y: y + 0.42, w: 5.0, h: 0.9, fontFace: BODY, fontSize: 14, color: C.muted, margin: 0, valign: "top", isTextBox: true });
  });
  s.addText(`${data.planner.cpu_latency_s} s per plan on CPU · ${data.planner.igpu_latency_s} s on Intel iGPU · ${data.planner.unit_tests} unit tests`, {
    x: 7.0, y: 6.05, w: 5.75, h: 0.4, fontFace: BODY, fontSize: 13, color: C.blue, margin: 0, isTextBox: true,
  });
  s.addNotes("Arms are named by the user when they want; the plan is simulated for hand state before anything moves.");
}

// 5 · Policy and data -------------------------------------------------------
{
  const s = pres.addSlide();
  s.background = { color: C.light };
  title(s, "ACT policy trained on randomised demonstrations");
  const stats = [
    [String(data.data.episodes), "demonstrations"],
    [data.data.frames.toLocaleString("en-GB"), "frames at 10 Hz"],
    [(data.data.train_steps / 1000) + "k", "training steps"],
    ["20", "future actions per chunk"],
  ];
  stats.forEach(([big, small], i) => {
    const x = M + i * 3.07;
    s.addText(big, { x, y: 1.6, w: 2.9, h: 1.0, fontFace: HEAD, fontSize: 48, bold: true, color: C.blue, margin: 0, isTextBox: true });
    s.addText(small, { x, y: 2.6, w: 2.9, h: 0.4, fontFace: BODY, fontSize: 14, color: C.muted, margin: 0, isTextBox: true });
  });
  body(s, bullets([
    "Inputs: overhead, operator and both wrist cameras (128 × 128), 12 joint positions, one-hot subtask from the planner",
    "Output: the next 20 joint-target vectors; the arm executes 10 and re-plans every second",
    `Demonstrations: scripted contact skills on randomised seeds (${data.data.scripted_success_on_train_seeds} kept); held-out seeds 0–9 never seen in training`,
    "No object is attached to a gripper: everything is held by finger contact and friction",
  ]), { x: M, y: 3.6, w: W - 2 * M, h: 2.8 });
  s.addNotes("Training seeds start at 100 so the ten evaluation seeds stay unseen. The planner's subtask token is how language reaches the policy.");
}

// 6 · Intel optimisation ----------------------------------------------------
{
  const s = pres.addSlide();
  s.background = { color: C.light };
  title(s, "OpenVINO on Intel: INT8 policy in 5.9 ms");
  // Horizontal bar charts draw the first category at the bottom; reverse so the list reads top-down.
  const rows = [...data.latency].reverse();
  s.addChart(pres.charts.BAR, [{
    name: "Latency per action chunk (ms)",
    labels: rows.map((d) => d.label),
    values: rows.map((d) => d.ms),
  }], {
    x: M, y: 1.5, w: 7.4, h: 4.9, barDir: "bar", chartColors: [C.blue],
    showTitle: true, title: "Latency per action chunk (ms, lower is better)", titleFontFace: BODY, titleFontSize: 14, titleColor: C.ink,
    showValue: true, dataLabelPosition: "outEnd", dataLabelColor: C.ink, dataLabelFontSize: 12, dataLabelFormatCode: "0.0",
    showLegend: false, catAxisLabelColor: C.ink, valAxisLabelColor: C.muted, catAxisLabelFontSize: 13,
    valGridLine: { color: C.rule, size: 0.5 }, catGridLine: { style: "none" },
  });
  s.addText(data.int8_speedup, { x: 8.5, y: 1.7, w: 4.2, h: 1.2, fontFace: HEAD, fontSize: 60, bold: true, color: C.accent, margin: 0, isTextBox: true });
  s.addText("faster with NNCF INT8 than FP32 on the same CPU", { x: 8.5, y: 2.9, w: 4.2, h: 0.7, fontFace: BODY, fontSize: 15, color: C.ink, margin: 0, isTextBox: true });
  body(s, bullets([
    `Action error after INT8: ${pending(data.int8_action_error_rad, (v) => v + " rad mean")}`,
    "Planner: INT4 LLM on OpenVINO GenAI, CPU and iGPU",
    "IRs keep dynamic shapes; runtimes pin batch-1 shapes at load",
  ]), { x: 8.5, y: 3.8, w: 4.2, h: 2.2, fontSize: 14 });
  s.addText(data.hardware_note, { x: M, y: 6.6, w: W - 2 * M, h: 0.45, fontFace: BODY, fontSize: 12, color: C.muted, margin: 0, isTextBox: true });
  s.addNotes("The policy re-plans once per second, so even FP32 uses under two percent of the control budget; INT8 frees the CPU for simulation and the planner.");
}

// 7 · Results ---------------------------------------------------------------
{
  const s = pres.addSlide();
  s.background = { color: C.light };
  title(s, "Results on 10 held-out randomised seeds");
  const header = ["Sub-goal", "Scripted", "Learned policy", "Hybrid"].map((text) => ({
    text, options: { bold: true, color: C.white, fill: { color: C.dark }, fontFace: BODY, fontSize: 14 },
  }));
  const cell = (list, i) => (list ? `${list[i]}/10` : "pending");
  const rows = data.subgoals.map((name, i) => [
    { text: name, options: { fontFace: BODY, fontSize: 14, color: C.ink } },
    { text: `${data.scripted_10_seeds[i]}/10`, options: { fontFace: BODY, fontSize: 14, color: C.good, bold: true, align: "center" } },
    { text: cell(data.policy_10_seeds, i), options: { fontFace: BODY, fontSize: 14, color: C.ink, align: "center" } },
    { text: cell(data.hybrid_10_seeds, i), options: { fontFace: BODY, fontSize: 14, color: C.ink, align: "center" } },
  ]);
  const full = (value) => (value === null || value === undefined ? "pending" : `${value}/10`);
  const bold = { fontFace: BODY, fontSize: 14, color: C.ink, bold: true, fill: { color: C.soft } };
  rows.push([
    { text: "Full task", options: bold },
    { text: full(data.scripted_full_task), options: { ...bold, color: C.good, align: "center" } },
    { text: full(data.policy_full_task), options: { ...bold, align: "center" } },
    { text: full(data.hybrid_full_task), options: { ...bold, align: "center" } },
  ]);
  s.addTable([header, ...rows], {
    x: M, y: 1.6, w: 7.6, colW: [3.1, 1.5, 1.5, 1.5], rowH: 0.5,
    border: { type: "solid", pt: 0.75, color: C.rule }, fill: { color: C.white },
  });
  if (data.failure_note) {
    s.addText(data.failure_note, {
      x: M, y: 5.85, w: 7.6, h: 0.8, fontFace: BODY, fontSize: 13, italic: true, color: C.blue, margin: 0, isTextBox: true,
    });
  }
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 8.7, y: 1.6, w: 4.03, h: 3.64, rectRadius: 0.06, fill: { color: C.white }, line: { color: C.rule, width: 1 } });
  s.addText("Honest limits", { x: 8.95, y: 1.8, w: 3.6, h: 0.45, fontFace: HEAD, fontSize: 18, bold: true, color: C.ink, margin: 0, isTextBox: true });
  body(s, bullets([
    "Water is 24 small beads, not a fluid",
    "Hybrid stages finished by a script are counted as assisted, never as policy wins",
    "Benchmarked on Core i9 + iGPU; Core Ultra run is one command",
  ]), { x: 8.95, y: 2.35, w: 3.6, h: 2.8, fontSize: 14 });
  s.addNotes("Randomisation per seed: object positions, masses, friction, lighting and table colour.");
}

// 8 · Next ------------------------------------------------------------------
{
  const s = pres.addSlide();
  s.background = { color: C.dark };
  title(s, "What's next", { color: C.white });
  const next = [
    ["Harder contact skills", "Particle-fluid pouring and full-size cutlery with the same contact-only physics"],
    ["Core Ultra NPU", "Run the INT8 policy on the NPU and the planner on the iGPU concurrently"],
    ["Real SO-101 arms", "Same plan language and policy interface on two physical arms"],
  ];
  next.forEach(([head, text], i) => {
    const x = M + i * 4.1;
    s.addShape(pres.shapes.OVAL, { x, y: 2.2, w: 0.5, h: 0.5, fill: { color: C.accent }, line: { color: C.accent } });
    s.addText(head, { x, y: 2.9, w: 3.7, h: 0.5, fontFace: HEAD, fontSize: 20, bold: true, color: C.white, margin: 0, isTextBox: true });
    s.addText(text, { x, y: 3.45, w: 3.7, h: 1.2, fontFace: BODY, fontSize: 15, color: "C9D1CE", margin: 0, valign: "top", isTextBox: true });
  });
  s.addText("Table for Two · code, benchmark and every number in this deck are reproducible from the repository", {
    x: M, y: 6.3, w: W - 2 * M, h: 0.5, fontFace: BODY, fontSize: 13, color: C.accent, margin: 0, isTextBox: true,
  });
  s.addNotes("Thank you. Everything shown is reproducible with the commands in the README.");
}

pres.writeFile({ fileName: path.join(__dirname, "table_for_two.pptx") }).then((file) => console.log("wrote " + file));
