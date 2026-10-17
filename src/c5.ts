/**
 * Patient 360 suite endpoints (coach-console INTEGRATION.md, "C5 Coaching calls"):
 *   GET /api/c5/calls[/:id]       supervisor-labelled demo calls, scored with MITI-style targets
 *   GET /api/c5/survivors[/:id]   one row per participant, status from their worst call
 *   GET /api/c5/demo              a safety call pre-played through the live engine (scripts/preplay.py)
 *   GET /api/c5/agreement         app score vs two supervisors, flag recall and false alarms (scripts/agreement.py)
 * The response shape is the console's own adapter, so the console switches over with no change
 * (it sends `x-c5-source: native`). The one difference: the flag kind is Clef's answer to the flag question.
 */
import PEOPLE from "./c5-people.json";
import AGREEMENT from "./agreement.json";
import DEMO from "./demo-safety-call.json";

interface CallLine {
  line_number: number; time_in_call: string; speaker: "coach" | "participant"; text: string;
  coach_move: string | null; participant_talk: string | null; needs_human: string | null; flag?: string;
}
interface Call {
  call_id: string; participant_id: string; coach_id: string; call_number: number; program_week: number;
  date: string; start_time: string; language: string; goal_topic: string; duration_min: number; lines: CallLine[];
}
interface Assets { fetch(req: Request): Promise<Response> }

const people = PEOPLE.participants as Record<string, Record<string, string>>;
const coaches = PEOPLE.coaches as Record<string, Record<string, string>>;
const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json", "cache-control": "public, max-age=60" } });

// MITI-style targets, as on the demo page (public/index.html updateScore), but each part is rounded before
// summing (the page rounds the sum), so scores can differ from the page by up to 2 points. Same as scripts/score.py.
const SUPPORT = new Set(["affirm", "seek collaboration", "emphasize autonomy", "ask permission"]);
function scoreMoves(moves: string[]) {
  const n = (t: string) => moves.filter((m) => m === t).length;
  const oq = n("open question"), cq = n("closed question"), sr = n("simple reflection"), cr = n("complex reflection");
  const ad = moves.filter((m) => SUPPORT.has(m)).length, na = n("persuade") + n("confront");
  const q = oq + cq, r = sr + cr;
  const parts = {
    reflectionsPerQuestion: { value: q ? +(r / q).toFixed(2) : null, target: 2, points: Math.round(30 * Math.min((q ? r / q : r ? 2 : 0) / 2, 1)), max: 30 },
    complexReflectionPct: { value: r ? Math.round((100 * cr) / r) : null, target: 40, points: Math.round(20 * Math.min((r ? cr / r : 0) / 0.4, 1)), max: 20 },
    openQuestionPct: { value: q ? Math.round((100 * oq) / q) : null, target: 50, points: Math.round(20 * Math.min((q ? oq / q : 0) / 0.5, 1)), max: 20 },
    miConsistent: { value: `${ad} vs ${na}`, target: "no persuade or confront", points: Math.round(30 * (ad + na ? ad / (ad + na) : 1)), max: 30 },
  };
  const score = moves.length < 4 ? null : Object.values(parts).reduce((s, p) => s + p.points, 0);
  return { score, parts };
}

function callView(c: Call) {
  const p = people[c.participant_id];
  const coach = coaches[c.coach_id];
  const { score, parts } = scoreMoves(c.lines.filter((l) => l.speaker === "coach" && l.coach_move).map((l) => l.coach_move as string));
  const flags = c.lines.filter((l) => l.needs_human === "yes")
    .map((l) => ({ line: l.line_number, t: l.time_in_call, kind: l.flag && l.flag !== "none" ? l.flag : "medical", text: l.text }));
  return {
    callId: c.call_id, survivorId: `c5-${c.participant_id}`, participantName: p?.first_name ?? c.participant_id,
    coach: { id: c.coach_id, name: coach?.name ?? c.coach_id, role: coach?.role ?? "" },
    date: c.date, start: c.start_time, language: c.language, goalTopic: c.goal_topic,
    callNumber: c.call_number, week: c.program_week, durationMin: c.duration_min,
    labels: "supervisor", score, scoreParts: parts, flags,
    lines: c.lines.map((l) => ({ n: l.line_number, t: l.time_in_call, speaker: l.speaker, text: l.text,
      supervisor: l.speaker === "coach" ? { move: l.coach_move } : { talk: l.participant_talk, needsHuman: l.needs_human === "yes" } })),
  };
}

type View = ReturnType<typeof callView>;
const RANK = { alert: 0, watch: 1, ok: 2 } as const;
const statusOf = (v: Pick<View, "flags" | "score">): keyof typeof RANK => (v.flags.length ? "alert" : v.score != null && v.score < 60 ? "watch" : "ok");
const headlineOf = (v: View) =>
  v.flags.length ? `${v.flags[0].kind[0].toUpperCase() + v.flags[0].kind.slice(1)} concern at ${v.flags[0].t} in call ${v.callNumber}: a person must follow up today`
  : v.score != null && v.score < 60 ? `Coach score ${v.score}/100 on call ${v.callNumber}: below the 60 target`
  : `Coach score ${v.score ?? "–"}/100 on call ${v.callNumber}`;
const latestFirst = (vs: View[]) => [...vs].sort((a, b) => b.date.localeCompare(a.date));
const worstOf = (vs: View[]) => [...vs].sort((a, b) => RANK[statusOf(a)] - RANK[statusOf(b)])[0];
const summaryRow = ({ lines, ...v }: View) => ({ ...v, status: statusOf(v), lines: lines.length });
const displayName = (id: string, name: string) => {
  const p = people[id];
  return `${name} · ${p ? `${p.age}${p.sex[0].toUpperCase()} · ${p.cancer_type}` : id}`;
};
const ACTIONS: Record<string, string> = {
  medical: "Tell the care team today and log the hand-off.",
  social: "Connect to a navigator today (food, money, transport).",
  distress: "Check in today and connect to psycho-oncology support.",
  dropout: "Call back this week to talk about staying in the program.",
};

let views: View[] | null = null;
async function loadViews(assets: Assets, origin: string) {
  if (!views) {
    const r = await assets.fetch(new Request(`${origin}/demo-calls.json`));
    views = ((await r.json()) as { calls: Call[] }).calls.map(callView);
  }
  return views;
}

export async function c5(assets: Assets, url: URL): Promise<Response> {
  const [kind, id] = url.pathname.replace(/^\/api\/c5\//, "").split("/").map(decodeURIComponent);
  if (kind === "demo" && !id) return json(DEMO);
  if (kind === "agreement" && !id) return json(AGREEMENT);
  const all = await loadViews(assets, url.origin);
  if (kind === "calls" && !id) {
    return json(all.map(summaryRow).sort((a, b) => b.flags.length - a.flags.length || (a.score ?? 0) - (b.score ?? 0)));
  }
  if (kind === "calls") {
    const v = all.find((x) => x.callId === id);
    return v ? json({ ...v, status: statusOf(v) }) : json({ error: "call not found", callId: id }, 404);
  }
  if (kind !== "survivors") return json({ error: "not found" }, 404);
  const byPerson = new Map<string, View[]>();
  for (const v of all) byPerson.set(v.survivorId, [...(byPerson.get(v.survivorId) ?? []), v]);
  if (!id) {
    const rows = [...byPerson.entries()].map(([sid, vs]) => {
      const worst = worstOf(vs);
      return { survivorId: sid, displayName: displayName(sid.slice(3), worst.participantName), status: statusOf(worst),
        headline: headlineOf(worst), updatedAt: latestFirst(vs)[0].date };
    });
    return json(rows.sort((a, b) => RANK[a.status] - RANK[b.status] || a.survivorId.localeCompare(b.survivorId)));
  }
  const vs = byPerson.get(id);
  if (!vs) return json({ error: "survivor not found", survivorId: id }, 404);
  const worst = worstOf(vs);
  const p = people[id.slice(3)];
  return json({
    survivorId: id, displayName: displayName(id.slice(3), worst.participantName),
    status: statusOf(worst), headline: headlineOf(worst),
    summary: p ? `${p.age}-year-old ${p.sex}, ${p.cancer_type} cancer. Program focus: ${p.program_focus}. Calls in ${p.call_language}. Coach ${worst.coach.name} (${worst.coach.role}).` : "",
    profile: p ? { firstName: p.first_name, age: +p.age, sex: p.sex, cancerType: p.cancer_type, language: p.call_language, focus: p.program_focus, coach: worst.coach.name } : {},
    evidence: [
      ...worst.flags.map((f) => ({ kind: `flag-${f.kind}`, text: `${f.t} “${f.text}”`, source: "supervisor label; kind from Clef" })),
      ...latestFirst(vs).map((v) => ({ kind: "call", text: `Call ${v.callNumber} (week ${v.week}, ${v.date}): coach score ${v.score ?? "–"}/100`, callId: v.callId })),
    ],
    recommendations: worst.flags.length
      ? worst.flags.map((f) => ({ action: ACTIONS[f.kind] ?? ACTIONS.medical, urgency: f.kind === "dropout" ? "this week" : "today" }))
      : statusOf(worst) === "watch" ? [{ action: "Review this call with the coach in supervision; start with the cue moments.", urgency: "this week" }]
      : [{ action: "No action. Keep the call schedule.", urgency: "routine" }],
    charts: { calls: latestFirst(vs).map(summaryRow) },
  });
}
