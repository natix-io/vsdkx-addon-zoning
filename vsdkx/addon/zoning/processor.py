import cv2
import numpy as np
from shapely.geometry import Polygon, Point
from vsdkx.core.interfaces import Addon
from numpy import ndarray
from vsdkx.core.structs import Inference


class ZoneProcessor(Addon):
    """
    - Processes static zones by:
        1. Excluding zones from the field of view (restrictive FOV)
        2. Reporting the amount of detected events per zone
    """

    def __init__(self, addon_config: dict, model_settings: dict,
                 model_config: dict, drawing_config: dict):
        """
        Args:
            remove_areas (list): List with areas to remove
            zones (list): List with configured zones of
            [xmin, ymin, xmax, ymax] format
            iou_thresh (float): IOU threshold
            class_names (list): List with class names
            class_ids (array): Array with class IDs
        """
        super().__init__(addon_config, model_settings, model_config,
                         drawing_config)
        self._remove_areas = addon_config.get("remove_areas", [])
        self._zones = addon_config.get("zones")
        self._iou_thresh = addon_config.get("iou_thresh")
        self._class_names = ['Person']
        self._class_ids = model_config.get("filter_class_ids", [])
        self._DNC_id = 500

        assert len(self._zones) > 0 or len(self._remove_areas) > 0, \
            "Incorrect ZoneProcessor set up. Please make sure to " \
            "define the coordinates for zones and remove_areas in the " \
            "system.yaml configuration."

        assert len(self._zones) == len(self._iou_thresh), \
            "Incorrect Zone configuration. Please make sure to define " \
            "a IOU threshold 'zone_iou_thresh' for every pre-configured " \
            "zone in the system.yaml configuration."

    def pre_process(self, image: ndarray) -> ndarray:
        """
        Blurs the selected zones from the image

        Args:
            image (np.array): Image array

        Returns:
            image (np.array): Transformed image
        """
        for i in range(len(self._remove_areas)):
            xmin = self._remove_areas[i][0]
            ymin = self._remove_areas[i][1]
            xmax = self._remove_areas[i][2]
            ymax = self._remove_areas[i][3]
            image[ymin:ymax, xmin:xmax] = cv2.blur(image[ymin:ymax,
                                                   xmin:xmax],
                                                   (30, 30))
        return image

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

    def post_process(self, inference: Inference) -> Inference:
        """
        Counts the amount of predicted events per zone

        Args:
             inference (Inference): the result of the ai

        Returns:
            zone_count (dict): Dictionary with events counts per zone
        """
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
            trackable_objects = inference.extra.get("trackable_object", {})
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
                        prev_centroid = to.centroids[-2]
                        current_centroid = to.centroids[-1]
                        prev_centroid = Point(prev_centroid[0],
                                              prev_centroid[1])
                        current_centroid = Point(current_centroid[0],
                                                 current_centroid[1])
                        prev_in = prev_centroid.within(zone)
                        current_in = current_centroid.within(zone)

                        if not prev_in and current_in:
                            # Update the class count in tracker
                            idx = self._class_ids.index(
                                int(inference.classes[j]))
                            class_name = self._class_names[idx]
                            obj_class_dict[class_name] += 1
                            enter_count[class_name] += 1
                        elif prev_in and not current_in:
                            # Update the class count in tracker
                            idx = self._class_ids.index(
                                int(inference.classes[j]))
                            class_name = self._class_names[idx]
                            obj_class_dict[class_name] -= 1
                            exit_count[class_name] += 1

                # Process the boxes that have not been assigned to a zone yet
                if box_zones[j] == 0 or box_zones[j] == self._DNC_id:
                    # Get the IoU
                    iou = self._get_iou(zone, poly_box)
                    # If the IoU is higher and equal than the IoU threshold
                    # register the box to the current zone ID and increase
                    # the object's counter (by class name) in the dictionary
                    if iou >= self._iou_thresh[i]:
                        box_zones[j] = i
                        idx = self._class_ids.index(int(inference.classes[j][0]))
                        class_name = self._class_names[idx]
                        obj_class_dict[class_name] += 1
                    else:
                        # Otherwise, assign the box to the DNC zone
                        box_zones[j] = self._DNC_id  # don't care zone
                else:
                    print(f'Box assigned to zone {box_zones[j]}, moving on..')

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
        return inference

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

    def _get_iou(self, zone, box):
        """
        Calculates the IOU between two objects

        Args:
            zone (Polygon): Zone Polygon
            box (Polygon): Box Polygon

        Returns:
            iou (float): IOU of the two objects
        """

        # Normally the IoU formula is calculated by:
        # I = intersection area / zone area + box area - intersection area
        # However, when calculating the IoU between two boxes where one of them
        # is larger by a greater scale, the IoU results to a very low score.
        # The same problem was observed when comparing small object bounding
        # boxes intersecting with a large box (zone). As a workaround,
        # we modified the IoU formula to only consider the intersection
        # between the zone and the box, with respect to the box area, which
        # results to a normal IoU score.

        polygon_intersection = zone.intersection(box).area
        polygon_union = box.union(box).area
        # polygon_union = box.union(zone).area # Original formula
        iou = polygon_intersection / polygon_union

        return iou
