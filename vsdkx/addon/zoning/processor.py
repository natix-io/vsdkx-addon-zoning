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
        self._class_names = ['Person']
        self._class_ids = model_config.get("filter_class_ids", [])
        self._DNC_id = 500

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
        for i in range(len(self._remove_areas)):
            xmin = self._remove_areas[i][0]
            ymin = self._remove_areas[i][1]
            xmax = self._remove_areas[i][2]
            ymax = self._remove_areas[i][3]
            addon_object.frame[ymin:ymax, xmin:xmax] = cv2.blur(
                addon_object.frame[ymin:ymax, xmin:xmax],
                (30, 30)
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
            obj_class_dict_sample[obj_class] = 0

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

        zone_count = {}

        # Init box_zone array to register the zone ID each box is assigned to
        box_zones = np.zeros(len(inference.boxes))
        # Calculate all box areas once
        boxes_poly = self._bounding_box_to_polygon(inference.boxes)
        zones_poly = self._zones_to_polygons(self._zones)

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
                to = self._get_trackable_object(trackable_objects, bbox)
                # Checking if a trackable object was found for that
                # bounding box
                if type(to) is not type(None):
                    # Ensures that we have at least the centroids
                    # of the two last frames
                    if len(to.centroids) > 2:
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

                            obj_class_dict[class_name] += 1
                            enter_count[class_name] += 1
                        elif prev_in and not current_in:
                            exit_count[class_name] += 1

                        elif prev_in and current_in:
                            # Object exists in the zone
                            obj_class_dict[class_name] += 1
                        else:
                            # Otherwise, assign the box to the DNC zone
                            box_zones[j] = self._DNC_id  # don't care zone

            # Assign the object class that entered/exited the zone
            obj_class_dict.update({'objects_entered': enter_count})
            obj_class_dict.update({'objects_exited': exit_count})

            zone_count[f'zone_{i}'] = obj_class_dict

        # Get the total count of objects in the DNC zone
        rest_zone_dict = self._create_dict()

        for i, zone_id in enumerate(box_zones):
            if zone_id == self._DNC_id:
                idx = self._class_ids.index(int(inference.classes[i][0]))
                class_name = self._class_names[idx]
                rest_zone_dict[class_name] += 1

        zone_count[rest_zone_str] = rest_zone_dict
        inference.extra["zoning"] = zone_count
        addon_object.inference = inference

        return addon_object

    def _get_trackable_object(self, trackable_objects, bounding_box):
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

    def _bounding_box_to_polygon(self, boxes):
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

    def _zones_to_polygons(self, zones):
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
