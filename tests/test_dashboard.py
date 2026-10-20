import unittest

from fastapi.testclient import TestClient

from luma_gym.dashboard import load_home, load_training, training_comparisons
from luma_gym.server import app


class TrainingComparisonsTests(unittest.TestCase):
    def test_submission_pages_and_totals(self):
        client = TestClient(app)
        self.assertEqual(client.get("/report").status_code, 404)
        self.assertEqual(client.get("/recording-notes").status_code, 404)
        home = client.get("/").text
        self.assertNotIn('href="/report"', home)
        self.assertNotIn('Historical code-fence', home)
        self.assertIn('>SD</th>', home)
        self.assertEqual(round(load_home()["gym"]["gpu_hours"], 1), 71.3)

    def test_controlled_and_focused_membership(self):
        runs, _ = load_training()
        groups = {group["id"]: group for group in training_comparisons(runs)}
        controlled = groups["controlled"]["runs"]
        self.assertEqual({run["beta"] for run in controlled}, {0.001, 0.04, 10, 100})
        self.assertEqual({run["done"] for run in controlled}, {60})
        for key in ("seed", "lr", "calc_prompts", "trap_prompts", "taskset_sha256", "demos_sha256"):
            self.assertEqual(len({str(run["config"][key]) for run in controlled}), 1, key)
        focused = groups["focused"]["runs"]
        self.assertEqual({run["done"] for run in focused}, {40, 60})
        self.assertEqual({run["beta"] for run in focused}, {0.04})
        self.assertTrue(all(not run["current"] for run in groups["earlier"]["runs"]))

    def test_curve_endpoints_and_page(self):
        runs, _ = load_training()
        groups = {group["id"]: group for group in training_comparisons(runs)}
        reward = groups["focused"]["charts"][0]
        self.assertEqual({path["last_step"] for path in reward["paths"]}, {40, 60})
        response = TestClient(app).get("/")
        self.assertEqual(response.status_code, 200)
        for key in ("controlled", "focused", "earlier"):
            self.assertEqual(response.text.count(f'id="training-{key}"'), 1)
        self.assertNotIn("data-curve-mode", response.text)
        self.assertEqual(response.text.count("Only beta changes in these four 60-step runs."), 1)
        self.assertIn("Fresh unseen-task confirmation", response.text)
        self.assertNotIn("Source: reports/", response.text)
        for group in groups.values():
            section = response.text.split(f'id="training-{group["id"]}"', 1)[1]
            self.assertLess(section.index('class="curves-grid"'), section.index(group["note"]))
