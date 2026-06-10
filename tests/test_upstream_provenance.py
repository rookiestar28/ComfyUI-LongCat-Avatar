from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
PROVENANCE_PATH = REPO_ROOT / "LongCat_Video" / "UPSTREAM_VERSION.md"


class UpstreamProvenanceTests(unittest.TestCase):
    def test_provenance_file_declares_required_upstream_identity(self):
        text = PROVENANCE_PATH.read_text(encoding="utf-8")

        self.assertIn("https://github.com/meituan-longcat/LongCat-Video", text)
        self.assertIn("6b3f4b8582a8bc3f20f795735f5383716c4ba794", text)  # pragma: allowlist secret
        self.assertIn("https://meigen-ai.github.io/LongCat-Video-Avatar-1.5-Page/", text)
        self.assertIn("LongCat_Video/longcat_video/", text)

    def test_provenance_file_documents_boundaries_and_legacy_cleanup(self):
        text = PROVENANCE_PATH.read_text(encoding="utf-8")

        self.assertIn("Embedded Source Boundary", text)
        self.assertIn("Local Patch Categories", text)
        self.assertIn("Legacy Cleanup Policy", text)
        self.assertIn("stale debris", text)
        self.assertIn("supported Avatar 1.5 contract", text)

    def test_provenance_file_stays_public_safe(self):
        text = PROVENANCE_PATH.read_text(encoding="utf-8")

        forbidden_fragments = [
            ".planning/",
            ".planning\\",
            "reference/",
            "reference\\",
            "b:\\",
            "C:\\Users\\",
            "token",
            "secret",
        ]
        for fragment in forbidden_fragments:
            self.assertNotIn(fragment, text)


if __name__ == "__main__":
    unittest.main()
