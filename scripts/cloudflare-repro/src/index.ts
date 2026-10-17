import { getSandbox, Sandbox } from "@cloudflare/sandbox";

export { Sandbox };

interface Env {
  Sandbox: DurableObjectNamespace<Sandbox>;
  RUN_SECRET: string;
}

const REPO = "https://github.com/zaycruz/coach-copilot";
const UV = "0.12.15";

// POST /run?commit=<sha>[&e2e=<baseUrl>]  body = Challenge5_Coaches_Copilot.zip  (Authorization: Bearer RUN_SECRET)
export default {
  async fetch(req: Request, env: Env): Promise<Response> {
    const url = new URL(req.url);
    if (req.headers.get("authorization") !== `Bearer ${env.RUN_SECRET}`) return new Response("unauthorized", { status: 401 });
    if (req.method !== "POST" || url.pathname !== "/run" || !req.body) return new Response("not found", { status: 404 });
    const commit = url.searchParams.get("commit") ?? "";
    if (!/^[0-9a-f]{40}$/.test(commit)) return Response.json({ error: "commit must be a full sha" }, { status: 400 });
    const e2e = url.searchParams.get("e2e");
    if (e2e && !/^https:\/\/[a-z0-9.-]+$/.test(e2e)) return Response.json({ error: "bad e2e url" }, { status: 400 });

    const sandbox = getSandbox(env.Sandbox, `repro-${commit.slice(0, 8)}`);
    const results: Record<string, unknown>[] = [];
    try {
      await sandbox.writeFile("/workspace/c5.zip", req.body);
      const steps: { name: string; cmd: string }[] = [
        { name: "environment", cmd: "set -e; uname -srm; . /etc/os-release && echo $PRETTY_NAME; nproc; free -m | sed -n 2p; python3 --version; git --version" },
        { name: "install uv", cmd: `set -e; curl -LsSf https://astral.sh/uv/${UV}/install.sh | env UV_NO_MODIFY_PATH=1 sh >/dev/null 2>&1; $HOME/.local/bin/uv --version` },
        { name: "fresh clone", cmd: `set -e; cd /workspace; rm -rf coach-copilot; git clone -q ${REPO}; cd coach-copilot; git checkout -q ${commit}; git log -1 --format='%H %cI %s'; git status --short | wc -l` },
        { name: "data step", cmd: "set -e; cd /workspace/coach-copilot; sha256sum /workspace/c5.zip; python3 -m zipfile -e /workspace/c5.zip data/; ls data" },
        { name: "reproduce", cmd: "cd /workspace/coach-copilot; export PATH=$HOME/.local/bin:$PATH; s=$(date +%s.%N); scripts/reproduce.sh; rc=$?; echo \"wall seconds: $(python3 -c \"print(round($(date +%s.%N)-$s,1))\")\"; uv run --frozen python -c 'import sys,pandas,numpy;print(\"python\",sys.version.split()[0],\"pandas\",pandas.__version__,\"numpy\",numpy.__version__)'; exit $rc" },
        { name: "diff committed metrics", cmd: "cd /workspace/coach-copilot; git diff --stat -- runs/metrics.json runs/agreement-per-call.json; git diff --quiet -- runs/metrics.json runs/agreement-per-call.json && echo 'runs/metrics.json and runs/agreement-per-call.json: byte-identical to the commit'" },
      ];
      if (e2e) steps.push({ name: "live e2e", cmd: `cd /workspace/coach-copilot; export PATH=$HOME/.local/bin:$PATH; E2E_RAN_FROM=cloudflare-containers-sandbox uv run --frozen python -I scripts/e2e_live.py ${e2e}; rc=$?; cat runs/e2e-live-*.json > /workspace/e2e.json; exit $rc` });
      for (const s of steps) {
        const t = Date.now();
        // Each step runs in its own bash, so `exit` or `set -e` cannot end the sandbox's shared session shell.
        await sandbox.writeFile("/workspace/step.sh", s.cmd);
        const r = await sandbox.exec("bash /workspace/step.sh", { timeout: 600_000 });
        results.push({ step: s.name, exitCode: r.exitCode, seconds: (Date.now() - t) / 1000, stdout: r.stdout, stderr: r.stderr.slice(-4000) });
        if (r.exitCode !== 0 && s.name !== "live e2e") break;
      }
      const e2eArtifact = e2e ? (await sandbox.exec("cat /workspace/e2e.json 2>/dev/null")).stdout : null;
      return Response.json({ ranOn: "Cloudflare Containers (Sandbox SDK 0.12.10, image cloudflare/sandbox:0.12.10-python, instance standard-1)", colo: req.cf?.colo ?? null, commit, results, e2eArtifact });
    } catch (err) {
      return Response.json({ error: String(err), stack: (err as Error)?.stack?.slice(0, 1500), results }, { status: 500 });
    } finally {
      await sandbox.destroy();
    }
  },
};
