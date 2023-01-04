
## Zoning
This module processes static zones by executing two different operations:

1. Excludes pre-configured zones from the field of view. By excluding zones from the field of view, prevents these areas from being used during inference. This results in event detection on the field of view that has remained intact.
2. Counts the detected events in the pre-configured zones, by zone ID and class name.

### Addon Config
```yaml
remove_areas: [ [ 0, 0, 736, 163 ]], # Array with zone(s) for exclusion coordinates
zones: [[[190, 250], [450, 250], [450, 420], [225, 450], [225, 350], [190, 350], [190, 250]],
                    [[480, 190], [950, 190], [1250, 380], [780, 430], [480, 190]]], # Array with zone(s) coordinates
iou_thresh: [0.85, 0.90], # List with IOU thresholds for IoU score per pre-configured zone
```

where:
- `zones`: Array of pre-configured zone coordinates (`xmin, ymin, xmax, ymax`) in polygon format:
   - **Example**: A square in polygon format `[(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax), (xmin, ymin)]`
- `iou_thresh`: Intersection over union for each zone
- `remove_areas`: Array of pre-configured area coordinates in the `xmin, ymin, xmax, ymax` format

It also require these properties inside model config dictionary:

```yaml
filter_class_ids: [0] # Array with event IDs that correspond to the class names
class_names: ['Person'] # Array with event class names
```

where: 
- `class_names`: Array with event class names
- `class_ids`: Array with event IDs that correspond to the class names
