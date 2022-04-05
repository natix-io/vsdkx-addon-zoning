import unittest
import numpy

from vsdkx.core.structs import AddonObject, Inference
from vsdkx.addon.zoning.processor import ZoneProcessor
from vsdkx.addon.tracking.trackableobject import TrackableObject


class TestAddon(unittest.TestCase):
    ADDON_OBJECT = None

    def test_constructor(self):
        addon_config = {
            "remove_areas": [[]],
            "zones": [[]],
            "class_names": [] 
        }

        model_config = {
            "filter_class_ids": []
        }

        TestAddon.ADDON_OBJECT = ZoneProcessor(addon_config, {}, model_config, {})

    def test_pre_process(self):
        
        frame = numpy.zeros((640, 640, 3))
        inference = Inference()
        shared = {}

        addon_object = AddonObject(frame=frame, inference=inference, shared=shared)

        TestAddon.ADDON_OBJECT._remove_areas = [
            [[13, 32], 
             [35, 200],
             [200, 200],
             [200, 16]]
             #[frame.shape[0], 0],
             #[frame.shape[0], frame.shape[1]],
             #[0, frame.shape[1]],
             #[0, 0]]
        ]
        print(addon_object.frame.shape)
        print(TestAddon.ADDON_OBJECT._remove_areas)
        pre_processed_addon_object = TestAddon.ADDON_OBJECT.pre_process(addon_object)
