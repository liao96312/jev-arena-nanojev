import tempfile
import unittest
from pathlib import Path

from arena.campaign_save import CampaignSave, load_campaign, save_campaign
from arena.entities import PlayerLoadout


class CampaignSaveTests(unittest.TestCase):
    def test_loadout_round_trips_and_bad_save_falls_back(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "campaign.json"
            expected = CampaignSave(4, 123, PlayerLoadout(True, True, 7, 11))
            save_campaign(path, expected)
            self.assertEqual(load_campaign(path), expected)
            path.write_text("broken", encoding="utf-8")
            self.assertEqual(load_campaign(path), CampaignSave())


if __name__ == "__main__":
    unittest.main()
