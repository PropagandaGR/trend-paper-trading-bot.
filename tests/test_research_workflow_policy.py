import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ResearchWorkflowPersistencePolicyTests(unittest.TestCase):
    def test_workflow_never_pushes_oos_ledger_or_research_artifacts_to_main(self):
        workflow = (ROOT / ".github" / "workflows" / "autonomous-research.yml").read_text()
        self.assertNotIn("git push origin HEAD:main", workflow)
        self.assertNotIn("git push origin HEAD:refs/heads/main", workflow)
        self.assertNotIn("git pull --rebase origin main", workflow)
        self.assertNotIn("Persist one-way OOS reservation even if research validation fails", workflow)
        self.assertIn("Restore authoritative OOS ledger from dedicated branch", workflow)
        self.assertIn("OOS_LEDGER_BRANCH: oos-ledger", workflow)
        self.assertIn("actions/upload-artifact@v4", workflow)

    def test_workflow_does_not_cancel_runs_after_reservation(self):
        workflow = (ROOT / ".github" / "workflows" / "autonomous-research.yml").read_text()
        self.assertIn("cancel-in-progress: false", workflow)

    def test_python_persistence_targets_dedicated_branch_and_uses_non_force_push(self):
        source = (ROOT / "research_engine.py").read_text()
        self.assertIn('branch == "main"', source)
        self.assertIn('f"HEAD:refs/heads/{branch}"', source)
        self.assertIn("normal fast-forward push", source)
        self.assertNotIn('f"HEAD:refs/heads/main"', source)


if __name__ == "__main__":
    unittest.main()
