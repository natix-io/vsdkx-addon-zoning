import unittest

from vsdkx.addon.zoning.processor import ZoneProcessor
from vsdkx.addon.tracking.trackableobject import TrackableObject


class TestAddon(unittest.TestCase):
    ADDON_OBJECT = None

    def test_constructor(self):
        addon_config = {
            "remove_areas": [],
            "zones": [],
            "class_names": [] 
        }

        model_config = {
            "filter_class_ids": []
        }

        ADDON_OBJECT = ZoneProcessor(addon_config, {}, model_config, {})

    #def test_pre_process(self):
