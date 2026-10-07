import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools import strata_mcp


class MacPathValidation(unittest.TestCase):
    def test_path_case_follows_filesystem_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            original = parent / "StrataApp"
            original.mkdir()
            alias = parent / "strataapp"

            with mock.patch.object(strata_mcp.sys, "platform", "darwin"):
                if os.path.samefile(original, alias):
                    self.assertEqual(strata_mcp.norm(original), strata_mcp.norm(alias))
                else:
                    alias.mkdir()
                    self.assertNotEqual(strata_mcp.norm(original), strata_mcp.norm(alias))
                    (alias / "Documents").mkdir()
                    tools = strata_mcp.Tools(strata_mcp.Strata(original))
                    with self.assertRaisesRegex(strata_mcp.ToolError, "new data folder must be named"):
                        tools.check_data_dir(str(alias / "Documents"))


if __name__ == "__main__":
    unittest.main()
