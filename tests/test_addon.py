import cv2
import numpy as np
import unittest

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

        TestAddon.ADDON_OBJECT = ZoneProcessor(addon_config, {}, model_config,
                                               {})

    def test_pre_process(self):
        frame = (np.random.rand(640, 640, 3) * 100).astype('uint8')
        inference = Inference()
        shared = {}

        addon_object = AddonObject(frame=frame, inference=inference,
                                   shared=shared)

        TestAddon.ADDON_OBJECT._remove_areas = [
            [[frame.shape[0], 0],
             [frame.shape[0], frame.shape[1]],
             [0, frame.shape[1]],
             [0, 0]]
        ]
        frame_mean = np.mean(frame)

        pre_processed_addon_object = TestAddon.ADDON_OBJECT.pre_process(
            addon_object)

        self.assertIsInstance(
            pre_processed_addon_object.frame, np.ndarray
        )
        self.assertNotEqual(
            frame_mean, np.mean(pre_processed_addon_object.frame)
        )

        self.assertEqual(
            pre_processed_addon_object.inference, Inference()
        )

        self.assertEqual(
            pre_processed_addon_object.shared, {}
        )
