import json
import os
import secrets
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from openai import AsyncOpenAI
from starlette.concurrency import run_in_threadpool

from luma_gym.documents import CaseDocuments
from luma_gym.environment import read_rules
from luma_gym.judge import JUDGE_PRICE, grade_answer
from luma_gym import RULESET_VERSION
from luma_gym.scoring import RUBRIC_VERSION
from luma_gym.tasks import Task
from luma_gym.cases import generate_cases
from luma_gym.submissions import submission_hash
from luma_gym.dashboard import (
    heat,
    load_home,
    load_report,
    load_rules,
    load_run_page,
    load_transcript,
    pct,
    signed,
    soft_heat,
    text_heat,
)


app = FastAPI(title="Luma Payroll Gym", version="1")
DATABASE = Path(os.environ.get("LUMA_DATABASE", "runtime/sandbox.sqlite"))
TEMPLATE_DIR = Path(__file__).with_name("templates")
if not TEMPLATE_DIR.exists():
    TEMPLATE_DIR = Path(__file__).resolve().parents[1] / "templates"
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))
templates.env.filters.update(pct=pct, heat=heat, soft_heat=soft_heat, text_heat=text_heat, signed=signed)


@contextmanager
def database():
    DATABASE.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DATABASE, timeout=15) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, tasks TEXT, created REAL, submitted INTEGER DEFAULT 0, result TEXT)")
        connection.execute("CREATE TABLE IF NOT EXISTS callers (caller TEXT, minute INTEGER, calls INTEGER, PRIMARY KEY(caller, minute))")
        columns = {row[1] for row in connection.execute("PRAGMA table_info(runs)")}
        if "answer_hash" not in columns:
            connection.execute("ALTER TABLE runs ADD COLUMN answer_hash TEXT")
        yield connection


def rate_limit(caller):
    minute = int(time.time() // 60)
    with database() as connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("DELETE FROM callers WHERE minute < ?", (minute - 10,))
        connection.execute("INSERT INTO callers VALUES (?, ?, 1) ON CONFLICT(caller, minute) DO UPDATE SET calls=calls+1", (caller, minute))
        count = connection.execute("SELECT calls FROM callers WHERE caller=? AND minute=?", (caller, minute)).fetchone()[0]
    if count > 12:
        raise HTTPException(429, "Rate limit reached")


@app.middleware("http")
async def limits(request: Request, call_next):
    size = request.headers.get("content-length")
    if size:
        try:
            if int(size) > 100000 or int(size) < 0:
                return HTMLResponse("Request too large", status_code=413)
        except ValueError:
            return HTMLResponse("Invalid Content-Length", status_code=400)
    if request.url.path.startswith(("/tasks", "/submit", "/runs")):
        try:
            await run_in_threadpool(rate_limit, request.client.host)
        except HTTPException as error:
            return HTMLResponse(error.detail, status_code=error.status_code)
    return await call_next(request)


@app.get("/health")
def health():
    return {"status": "ok", "ruleset_version": RULESET_VERSION, "rubric_version": RUBRIC_VERSION}


@app.get("/tasks")
def fetch_tasks(n: int = 3, tier: str = "all"):
    if not 1 <= n <= 10 or tier not in ("all", "1", "2", "3"):
        raise HTTPException(400, "Use 1-10 tasks and a valid tier")
    seed = secrets.randbelow(2**31)
    tiers = (1, 2, 3) if tier == "all" else (int(tier),)
    candidates = generate_cases(seed, calculations=n, traps=n, retrievals=n)
    tasks = [task for task in candidates if task.difficulty in tiers][:n]
    if len(tasks) != n:
        raise HTTPException(400, "Not enough distinct tasks for that tier")
    run_id = secrets.token_urlsafe(24)
    with database() as connection:
        connection.execute("DELETE FROM runs WHERE created < ?", (time.time() - 86400,))
        connection.execute("INSERT INTO runs(id,tasks,created) VALUES(?,?,?)",
                           (run_id, json.dumps([task.to_dict() for task in tasks]), time.time()))
    public = []
    for task in tasks:
        case = CaseDocuments(task)
        public.append({"id": task.id, "prompt": task.prompt, "documents": case.documents,
                       "reference_tables": case.tables})
    return {"run_id": run_id, "ruleset_version": RULESET_VERSION, "rubric_version": RUBRIC_VERSION, "seed": seed,
            "tier_mix": {str(t): sum(task.difficulty == t for task in tasks) for t in (1, 2, 3)},
            "rules": read_rules(), "tasks": public, "expires_in_seconds": 86400}


class SandboxJudge:
    async def get_native_response(self, prompt, model, sampling_args, tools=None, **kwargs):
        from luma_gym.budget import Budget
        from luma_gym.hosted import load_key

        if os.environ.get("SPACE_ID") and not os.environ.get("OPENROUTER_API_KEY"):
            raise ValueError("Hosted judge credentials are not configured")
        budget = Budget()
        reserved = (len(json.dumps(prompt).encode()) + 2048) * JUDGE_PRICE["input"] + 8192 * JUDGE_PRICE["output"]
        budget.reserve(reserved)
        key = os.environ.get("OPENROUTER_API_KEY") or load_key(Path.home() / ".env", "OPENROUTER_API_KEY")
        async with AsyncOpenAI(api_key=key, base_url="https://openrouter.ai/api/v1", max_retries=0) as client:
            response = await client.chat.completions.create(
                model=model, messages=prompt, max_tokens=8192, **sampling_args,
            )
        cost = getattr(response.usage, "cost", None)
        if type(cost) not in (int, float):
            raise ValueError("Unknown judge cost")
        budget.settle(reserved, cost)
        return response


@app.post("/submit")
async def submit(request: Request):
    raw = await request.body()
    if len(raw) > 100000:
        raise HTTPException(413, "Request too large")
    try:
        data = json.loads(raw)
        run_id = data["run_id"]
        answers = data["answers"]
        if not isinstance(run_id, str) or not isinstance(answers, list) or len(answers) > 10:
            raise ValueError()
    except (ValueError, KeyError, TypeError):
        raise HTTPException(400, "Invalid submission") from None
    with database() as connection:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute("SELECT tasks,submitted,result,answer_hash,created FROM runs WHERE id=?", (run_id,)).fetchone()
        if row is None or row[4] < time.time() - 86400:
            raise HTTPException(404, "Unknown or expired run")
        private = {item["id"]: Task.from_dict(item) for item in json.loads(row[0])}
        try:
            digest = submission_hash(answers, private)
        except ValueError:
            raise HTTPException(400, "Submit exactly one valid answer per task") from None
        if row[3] and row[3] != digest:
            raise HTTPException(409, "Run already bound to a different submission")
        if row[1]:
            if row[2]:
                return json.loads(row[2])
            raise HTTPException(409, "Submission already in progress")
        try:
            if set(item["id"] for item in answers) != set(private) or len(answers) != len(private):
                raise ValueError()
            for item in answers:
                if set(item) != {"id", "answer"} or not isinstance(item["answer"], (dict, str)):
                    raise ValueError()
        except (ValueError, TypeError, KeyError):
            raise HTTPException(400, "Submit exactly one answer per task") from None
        connection.execute("UPDATE runs SET submitted=1,answer_hash=? WHERE id=?", (digest, run_id))
    results = []
    try:
        prompt = Path("rules/judge.md").read_text()
        for item in answers:
            text = json.dumps(item["answer"]) if isinstance(item["answer"], dict) else item["answer"]
            result = await grade_answer(private[item["id"]], text, SandboxJudge(), prompt=prompt)
            results.append(result)
    except Exception:
        with database() as connection:
            connection.execute("UPDATE runs SET submitted=0 WHERE id=?", (run_id,))
        raise HTTPException(503, "Scoring unavailable; retry later") from None
    result = {"run_id": run_id, "ruleset_version": RULESET_VERSION, "rubric_version": RUBRIC_VERSION,
              "mean_reward": sum(item["score"] for item in results) / len(results), "per_task": results}
    with database() as connection:
        connection.execute("UPDATE runs SET result=? WHERE id=?", (json.dumps(result), run_id))
    return result


@app.get("/runs/{run_id}")
def get_run(run_id: str):
    with database() as connection:
        row = connection.execute("SELECT result,created FROM runs WHERE id=?", (run_id,)).fetchone()
    if row is None or row[0] is None or row[1] < time.time() - 86400:
        raise HTTPException(404, "Unknown or unfinished run")
    return json.loads(row[0])


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request=request, name="home.html", context=load_home())


@app.get("/report", response_class=HTMLResponse)
def report(request: Request):
    return templates.TemplateResponse(request=request, name="document.html", context={"title": "Report", **load_report()})


@app.get("/rules", response_class=HTMLResponse)
def rules(request: Request):
    return templates.TemplateResponse(request=request, name="document.html", context={"title": "Rules", **load_rules()})


@app.get("/run/{run_id}", response_class=HTMLResponse)
def training_run(request: Request, run_id: str):
    try:
        context = load_run_page(run_id)
    except ValueError:
        raise HTTPException(404, "Unknown run") from None
    return templates.TemplateResponse(request=request, name="run.html", context=context)


@app.get("/transcript/{run}/{task_id}", response_class=HTMLResponse)
def transcript(request: Request, run: str, task_id: str):
    try:
        context = load_transcript(run, task_id)
    except ValueError:
        raise HTTPException(404, "Unknown result") from None
    return templates.TemplateResponse(request=request, name="transcript.html", context=context)


@app.get("/results", include_in_schema=False)
@app.get("/local", include_in_schema=False)
@app.get("/sandbox", include_in_schema=False)
@app.get("/status", include_in_schema=False)
@app.get("/explore", include_in_schema=False)
@app.get("/view/{rest:path}", include_in_schema=False)
def old_page(request: Request):
    anchor = {"/results": "leaderboard", "/local": "training", "/sandbox": "sandbox", "/status": "training"}
    return RedirectResponse("/#" + anchor.get(request.url.path, "failures"), status_code=302)
