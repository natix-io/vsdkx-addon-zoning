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
            "class_names": ["Person"]
        }

        model_config = {
            "filter_class_ids": [0]
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

    def test_post_process_one_bbox(self):    
        frame = np.zeros((640, 480, 3)).astype('uint8')
        inference = Inference()
        shared = {}

        addon_object = AddonObject(frame=frame, inference=inference,
                                   shared=shared)        
        
        TestAddon.ADDON_OBJECT._remove_areas = [[]]
        TestAddon.ADDON_OBJECT._zones = [
                [[100, 100],
                 [200, 50],
                 [400, 100],
                 [400, 400],
                 [100, 400],
                 [100, 100]]
            ]
        
        bb_1 = np.array([120, 150, 170, 200])
        c_1 = [145, 175]
        
        boxes = [bb_1]
        
        trackable_object_1 = TrackableObject(0, c_1, bb_1)
        trackable_object_1.centroids.extend([c_1, c_1])

        shared = {
            "trackable_objects": {
                "0": trackable_object_1,
            }
        }
        
        addon_object.inference.boxes = boxes
        addon_object.inference.classes = np.array([0])
        addon_object.shared = shared

        post_processed_addon_object = TestAddon.ADDON_OBJECT.post_process(addon_object)
        
        self.assertEqual(
            post_processed_addon_object.inference.extra["zoning"]["zone_0"]["Person_count"], 
            1
        )

        self.assertEqual(
            post_processed_addon_object.inference.extra["zoning"]["zone_0"]["Person"], 
            [0]
        )

        self.assertEqual(
            post_processed_addon_object.inference.extra["zoning"]["rest"]["Person_count"], 
            0
        )

        self.assertEqual(
            post_processed_addon_object.inference.extra["zoning"]["rest"]["Person"], 
            []
        )

    def test_post_process_two_bbox(self):
        frame = np.zeros((640, 480, 3)).astype('uint8')
        inference = Inference()
        shared = {}

        addon_object = AddonObject(frame=frame, inference=inference,
                                   shared=shared)        
        
        TestAddon.ADDON_OBJECT._remove_areas = [[]]
        TestAddon.ADDON_OBJECT._zones = [
                [[100, 100],
                 [200, 50],
                 [400, 100],
                 [400, 400],
                 [100, 400],
                 [100, 100]]
            ]
        
        bb_1 = np.array([120, 150, 170, 200])
        c_1 = [145, 175]
        
        trackable_object_1 = TrackableObject(0, c_1, bb_1)
        trackable_object_1.centroids.extend([c_1, c_1])
       
        bb_2 = np.array([0, 0, 50 , 45 ])
        c_2 = [25, 22.5]
        
        trackable_object_2 = TrackableObject(1, c_2, bb_2)
        trackable_object_2.centroids.extend([c_2, c_2])
 
        boxes = [bb_1, bb_2]

        shared = {
            "trackable_objects": {
                "0": trackable_object_1,
                "1": trackable_object_2
            }
        }

        addon_object.inference.boxes = boxes
        addon_object.inference.classes = np.array([0, 0])
        addon_object.shared = shared

        post_processed_addon_object = TestAddon.ADDON_OBJECT.post_process(addon_object)

        self.assertEqual(
            post_processed_addon_object.inference.extra["zoning"]["zone_0"]["Person_count"], 
            1
        )

        self.assertEqual(
            post_processed_addon_object.inference.extra["zoning"]["zone_0"]["Person"], 
            [0]
        )

        self.assertEqual(
            post_processed_addon_object.inference.extra["zoning"]["rest"]["Person_count"], 
            1
        )

        self.assertEqual(
            post_processed_addon_object.inference.extra["zoning"]["rest"]["Person"], 
            [1]
        )

