"""Validate the pinned model catalog used by macOS model discovery."""
from __future__ import annotations

import re
import unittest

from tools.model_catalog import CONTEXTS, FAMILIES, HF_REVISIONS, MODELS


class ModelCatalogTests(unittest.TestCase):
    def test_every_catalog_revision_is_an_immutable_hub_commit(self):
        self.assertTrue(HF_REVISIONS)
        for repository, revision in HF_REVISIONS.items():
            with self.subTest(repository=repository):
                self.assertRegex(revision, re.compile(r"^[0-9a-f]{40}$"))

    def test_each_family_download_url_uses_its_pinned_revision(self):
        for name, family in FAMILIES.items():
            with self.subTest(family=name):
                for key in ("hf", "mmproj_hf"):
                    if key not in family:
                        self.assertFalse(family.get("vision", True))
                        continue
                    url = family[key]
                    self.assertIn("/resolve/", url)
                    self.assertNotIn("/resolve/main/", url)
                    self.assertIn(url.split("/resolve/", 1)[1][:40], HF_REVISIONS.values())

    def test_model_choices_and_contexts_are_unique(self):
        self.assertIn("Q2_0", MODELS)
        self.assertIn("IQ2_XS", MODELS)
        self.assertEqual(len(CONTEXTS), len(set(CONTEXTS)))
        self.assertEqual(CONTEXTS, sorted(CONTEXTS))


if __name__ == "__main__":
    unittest.main()
