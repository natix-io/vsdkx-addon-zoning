import cv2
import numpy as np
from shapely.geometry import Polygon, Point
from vsdkx.core.interfaces import Addon, AddonObject


class ZoneProcessor(Addon):
    """
    - Processes static zones by:
        1. Excluding zones from the field of view (restrictive FOV)
        2. Reporting the amount of detected events per zone
    """

    def __init__(self, addon_config: dict, model_settings: dict,
                 model_config: dict, drawing_config: dict):
        super().__init__(addon_config, model_settings, model_config,
                         drawing_config)
        self._remove_areas = addon_config.get("remove_areas", [])
        self._zones = addon_config.get("zones")
        self._class_names = addon_config.get('class_names', ['Person'])
        self._class_ids = model_config.get("filter_class_ids", [])
        self._blur_kernel = (53, 53)
        self._cv_sigma_x = 30

        assert len(self._zones) > 0 or len(self._remove_areas) > 0, \
            "Incorrect ZoneProcessor set up. Please make sure to " \
            "define the coordinates for zones and remove_areas in the " \
            "system.yaml configuration."

    def pre_process(self, addon_object: AddonObject) -> AddonObject:
        """
        Blurs the selected zones from the image

        Args:
            addon_object (AddonObject): addon object containing information
            about frame and/or other addons shared data

        Returns:
            (AddonObject): addon object has updated information for frame,
            inference, result and/or shared information:
        """
        for area in self._remove_areas:
            roi_corners = np.array(
                [area],
                dtype=np.int32)
            blurred_image = cv2.GaussianBlur(addon_object.frame, self._blur_kernel, self._cv_sigma_x)
            mask = np.zeros(addon_object.frame.shape, dtype=np.uint8)
            channel_count = addon_object.frame.shape[2]
            ignore_mask_color = (255,) * channel_count
            cv2.fillPoly(mask, roi_corners, ignore_mask_color)
            mask_inverse = np.ones(mask.shape).astype(np.uint8) * 255 - mask
            addon_object.frame = cv2.bitwise_and(
                blurred_image, mask
            ) + cv2.bitwise_and(
                addon_object.frame, mask_inverse
            )

        return addon_object

    def _create_dict(self):
        """
        Inits an empty dictionary with the class names

        Returns:
            (dict): Class names dictionary
        """
        obj_class_dict_sample = {}
        for obj_class in self._class_names:
            obj_class_dict_sample[obj_class] = []

        return obj_class_dict_sample

    def post_process(self, addon_object: AddonObject) -> AddonObject:
        """
        Counts the amount of predicted events per zone

        Args:
            addon_object (AddonObject): addon object containing information
            about inference, frame, other addons shared data

        Returns:
            (AddonObject): addon object has updated information for inference
            result and/or shared information
        """
        inference = addon_object.inference

        rest_zone_str = 'rest'
        rest_zone_dict = self._create_dict()

        zone_count = {}

        # Init box_zone array to register the zone ID each box is assigned to
        box_zones = np.zeros(len(inference.boxes))
        # Calculate all box areas once
        boxes_poly = ZoneProcessor._bounding_box_to_polygon(inference.boxes)
        zones_poly = ZoneProcessor._zones_to_polygons(self._zones)

        # Iterate through all zones
        for i, zone in enumerate(zones_poly):
            zone_id = f'zone_{i}'
            zone_count[zone_id] = 0

            # Create object class dictionary to store the counted
            # objects per class per zone
            obj_class_dict = self._create_dict()
            # Create dictionaries to store the objects per
            # class that entered/exited a zone
            enter_count = self._create_dict()
            exit_count = self._create_dict()
            trackable_objects = addon_object.shared.get("trackable_objects",
                                                        {})
            # Iterate through all boxes
            for j, (poly_box, bbox) in enumerate(
                    zip(boxes_poly, inference.boxes)):  
                to = ZoneProcessor._get_trackable_object(trackable_objects, bbox)

                # Checking if a trackable object was found for that
                # bounding box, and we have at least the centroids of
                # the two last frames
                if type(to) is not type(None) \
                        and len(to.centroids) > 2:
                    # Get the object centroids from the previous
                    # and current frames
                    prev_centroid = to.centroids[-2]
                    current_centroid = to.centroids[-1]
                    prev_centroid = Point(prev_centroid[0],
                                          prev_centroid[1])
                    current_centroid = Point(current_centroid[0],
                                             current_centroid[1])
                    prev_in = prev_centroid.within(zone)
                    current_in = current_centroid.within(zone)

                    # Get class idx and class name
                    idx = self._class_ids.index(
                        int(addon_object.inference.classes[j])
                    )
                    class_name = self._class_names[idx]

                    if not prev_in and current_in:
                        # Update the class count in enter_count
                        # (Object has entered the zone)

                        obj_class_dict[class_name].append(to.object_id)
                        enter_count[class_name].append(to.object_id)
                    elif prev_in and not current_in:
                        exit_count[class_name].append(to.object_id)

                    elif prev_in and current_in:
                        # Object exists in the zone
                        obj_class_dict[class_name].append(to.object_id)
                    else:
                        # Otherwise, assign the box to the rest zone
                        rest_zone_dict[class_name].append(to.object_id)

            # Assign the object class that entered/exited the zone
            obj_class_dict.update({'objects_entered': enter_count})
            obj_class_dict.update({'objects_exited': exit_count})

            zone_count[f'zone_{i}'] = obj_class_dict

        zone_count[rest_zone_str] = rest_zone_dict
        zone_count = ZoneProcessor._count_object_ids(zone_count)

        inference.extra["zoning"] = zone_count
        addon_object.inference = inference

        return addon_object

    @staticmethod
    def _count_object_ids(zone_dict: dict) -> dict:
        """
        Recursive function iterates over zones dictionary and adds count number
        for every list object value or calls itself for every dictionary value.

        Args:
            zone_dict (dict): zones dictionary with object class name to object
            ids mapping.

        Returns:
            (dict): updated zones dictionary with object class name to object
            count mapping values added.
        """
        zone_dict_copy = zone_dict.copy()

        for key, value in zone_dict.items():
            if isinstance(value, list):
                zone_dict_copy[key + '_count'] = len(value)
            elif isinstance(value, dict):
                zone_dict_copy[key + '_ids'] = \
                    ZoneProcessor._count_object_ids(value)

        return zone_dict_copy

    @staticmethod
    def _get_trackable_object(trackable_objects, bounding_box):
        """
        Filters trackable objects by their bounding boxes

        Args:
            trackable_objects (dict): Dictionary with Trackable Objects
            bounding_box (np.array): Array with bounding box coordinates

        Returns:
            (TrackableObject): Trackable object item
        """
        trackable_object = None

        for to_id, to_obj in trackable_objects.items():
            if np.array_equal(bounding_box.astype(int),
                              to_obj.bounding_box.astype(int)):
                trackable_object = to_obj
        return trackable_object

    @staticmethod
    def _bounding_box_to_polygon(boxes):
        """
        Converts bounding boxes from xmin, ymin, xmax,ymax, to a polygon object

        Args:
            boxes (List): List with bounding boxes

        Returns:
            polygons (List): List with Polygon objects
        """
        polygons = []
        for box in boxes:
            polygon = Polygon([(box[0], box[1]),
                               (box[2], box[1]),
                               (box[2], box[3]),
                               (box[0], box[3]),
                               (box[0], box[1])])
            polygons.append(polygon)

        return polygons

    @staticmethod
    def _zones_to_polygons(zones):
        """
        Converts a list of polygon zone points into a list of polygon objects

        Args:
            zones (List): List with polygon zone points

        Returns:
            polygons (List): List with Polygon objects
        """

        polygons = []
        for zone in zones:
            polygon = Polygon(zone)
            polygons.append(polygon)

        return polygons
