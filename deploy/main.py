import json
import secrets
import time
from types import SimpleNamespace
from urllib.parse import urlparse, parse_qs

from workers import WorkerEntrypoint, Response, fetch

from luma_gym import RULESET_VERSION
from luma_gym.documents import CaseDocuments
from luma_gym.judge import grade_answer
from luma_gym.scoring import RUBRIC_VERSION
from luma_gym.cases import generate_cases
from luma_gym.submissions import submission_hash
from luma_gym.tasks import Task

from rule_text import JUDGE, RULES


def respond(data, status=200):
    return Response(json.dumps(data), status=status, headers={"Content-Type": "application/json"})


class WorkerJudge:
    def __init__(self, env):
        self.env = env

    async def get_native_response(self, prompt, model, sampling_args, tools=None, **kwargs):
        reservation = 0.03
        row = await self.env.DB.prepare(
            "UPDATE budget SET reserved=reserved+? WHERE id=1 AND spent+reserved+?<=ceiling RETURNING id"
        ).bind(reservation, reservation).first()
        if row is None:
            raise ValueError("Hosted judge budget unavailable")
        payload = {"model": model, "messages": prompt, "max_tokens": 8192, **sampling_args}
        extra = payload.pop("extra_body", {})
        payload.update(extra)
        response = await fetch("https://openrouter.ai/api/v1/chat/completions", method="POST",
                               headers={"Authorization": "Bearer " + self.env.OPENROUTER_API_KEY,
                                        "Content-Type": "application/json"}, body=json.dumps(payload))
        if response.status != 200:
            raise ValueError("Judge provider unavailable")
        data = json.loads(await response.text())
        cost = data.get("usage", {}).get("cost")
        if type(cost) not in (int, float) or cost < 0 or cost > reservation:
            raise ValueError("Unknown judge accounting")
        await self.env.DB.prepare("UPDATE budget SET spent=spent+?,reserved=reserved-? WHERE id=1").bind(cost, reservation).run()
        content = data["choices"][0]["message"]["content"]
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class Default(WorkerEntrypoint):
    async def fetch(self, request):
        try:
            return await self.handle(request)
        except Exception as error:
            print(json.dumps({"error": type(error).__name__}))
            return respond({"error": "Scoring service unavailable; retry later"}, 503)

    async def handle(self, request):
        url = urlparse(request.url)
        query = parse_qs(url.query)
        if url.path == "/health":
            return respond({"status": "ok", "ruleset_version": RULESET_VERSION, "rubric_version": RUBRIC_VERSION})
        caller = request.headers.get("CF-Connecting-IP") or "unknown"
        minute = int(time.time() // 60)
        count = await self.env.DB.prepare(
            "INSERT INTO callers(id,minute,calls) VALUES(?,?,1) ON CONFLICT(id) DO UPDATE SET calls=CASE WHEN minute=? THEN calls+1 ELSE 1 END,minute=? RETURNING calls"
        ).bind(caller, minute, minute, minute).first()
        if count.calls > 12:
            return respond({"error": "Rate limit reached"}, 429)
        if url.path == "/tasks" and request.method == "GET":
            try:
                n = int(query.get("n", ["3"])[0])
            except ValueError:
                return respond({"error": "Invalid task count"}, 400)
            tier = query.get("tier", ["all"])[0]
            if not 1 <= n <= 10 or tier not in ("all", "1", "2", "3"):
                return respond({"error": "Invalid task count or tier"}, 400)
            seed = secrets.randbelow(2**31)
            tiers = (1, 2, 3) if tier == "all" else (int(tier),)
            candidates = generate_cases(seed, calculations=n, traps=n, retrievals=n)
            tasks = [task for task in candidates if task.difficulty in tiers][:n]
            if len(tasks) != n:
                return respond({"error": "Not enough distinct tasks"}, 400)
            run_id = secrets.token_urlsafe(24)
            await self.env.DB.prepare("DELETE FROM runs WHERE created<?").bind(int(time.time()) - 86400).run()
            await self.env.DB.prepare("INSERT INTO runs(id,tasks,created) VALUES(?,?,?)").bind(
                run_id, json.dumps([task.to_dict() for task in tasks]), int(time.time()),
            ).run()
            public = []
            for task in tasks:
                case = CaseDocuments(task)
                public.append({"id": task.id, "prompt": task.prompt, "documents": case.documents,
                               "reference_tables": case.tables})
            return respond({"run_id": run_id, "ruleset_version": RULESET_VERSION, "rubric_version": RUBRIC_VERSION, "seed": seed,
                            "tier_mix": {str(t): sum(task.difficulty == t for task in tasks) for t in (1, 2, 3)},
                            "rules": RULES, "tasks": public})
        if url.path == "/submit" and request.method == "POST":
            size = request.headers.get("Content-Length")
            if size is None or int(size) > 100000:
                return respond({"error": "Supply Content-Length below 100000"}, 413)
            try:
                data = json.loads(await request.text())
            except ValueError:
                return respond({"error": "Invalid JSON"}, 400)
            if not isinstance(data, dict):
                return respond({"error": "Submission must be an object"}, 400)
            run_id = data.get("run_id")
            if not isinstance(run_id, str):
                return respond({"error": "Invalid run ID"}, 400)
            row = await self.env.DB.prepare("SELECT tasks,result,answer_hash,created FROM runs WHERE id=?").bind(run_id).first()
            if row is None or row.created < time.time() - 86400:
                return respond({"error": "Unknown run"}, 404)
            tasks = {item["id"]: Task.from_dict(item) for item in json.loads(row.tasks)}
            answers = data.get("answers")
            try:
                digest = submission_hash(answers, tasks)
            except ValueError:
                return respond({"error": "Submit exactly one valid answer per task"}, 400)
            if row.answer_hash and row.answer_hash != digest:
                return respond({"error": "Run already bound to a different submission"}, 409)
            if row.result:
                return respond(json.loads(row.result))
            if not isinstance(answers, list) or any(not isinstance(item, dict) for item in answers):
                return respond({"error": "Invalid answers"}, 400)
            if len(answers) != len(tasks) or {item.get("id") for item in answers} != set(tasks):
                return respond({"error": "Submit exactly one answer per task"}, 400)
            if any(set(item) != {"id", "answer"} or not isinstance(item["answer"], (str, dict)) for item in answers):
                return respond({"error": "Invalid answer fields"}, 400)
            claimed = await self.env.DB.prepare("UPDATE runs SET processing=1,answer_hash=? WHERE id=? AND processing=0 AND result IS NULL AND (answer_hash IS NULL OR answer_hash=?) RETURNING id").bind(digest, run_id, digest).first()
            if claimed is None:
                return respond({"error": "Submission in progress"}, 409)
            results = []
            try:
                for item in answers:
                    text = json.dumps(item["answer"]) if isinstance(item["answer"], dict) else item["answer"]
                    results.append(await grade_answer(tasks[item["id"]], text, WorkerJudge(self.env),
                                                      prompt=JUDGE))
            except Exception:
                await self.env.DB.prepare("UPDATE runs SET processing=0 WHERE id=?").bind(run_id).run()
                raise
            result = {"run_id": run_id, "ruleset_version": RULESET_VERSION, "rubric_version": RUBRIC_VERSION,
                      "mean_reward": sum(item["score"] for item in results) / len(results), "per_task": results}
            await self.env.DB.prepare("UPDATE runs SET result=?,processing=0 WHERE id=?").bind(json.dumps(result), run_id).run()
            return respond(result)
        if url.path.startswith("/runs/"):
            row = await self.env.DB.prepare("SELECT result,created FROM runs WHERE id=?").bind(url.path.split("/")[-1]).first()
            return respond(json.loads(row.result)) if row and row.result and row.created >= time.time() - 86400 else respond({"error": "Unknown or unfinished run"}, 404)
        return respond({"name": "Luma Payroll Gym", "fetch": "/tasks?n=3", "submit": "/submit",
                        "dashboard": "/"})
