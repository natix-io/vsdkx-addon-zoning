
## Zoning
This module processes static zones by executing two different operations:

1. During `pre_process` it excludes pre-configured zones from the field of view. By excluding zones from the field of view, prevents these areas from being used during inference. This results in event detection on the field of view that has remained intact. 
2. In the `post_process` it counts the detected events in the pre-configured zones, by zone ID and class name.

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

It also requires the following properties that can be passed from the model config dictionary. These properties are essential for the operations executed during the `post_process` operation of this add-on:

```yaml
filter_class_ids: [0] # Array with event IDs that correspond to the class names
class_names: ['Person'] # Array with event class names
```

where: 
- `class_names`: Array with event class names
- `class_ids`: Array with event IDs that correspond to the class names

## Debug

- Object initialization:
  ```python
  from vsdkx.addon.zoning.processor import ZoneProcessor
  
  add_on_config = {
    'filter_class_ids': [0], 
    'remove_areas': [], 'zones': [[[480, 279], [580, 270], [1258, 697], [332, 710], [396, 405], [480, 279]]], 
    'class': 'vsdkx.addon.zoning.processor.ZoneProcessor', 
    'class_names': ['Person'], 'iou_thresh': (0.85,)
    }
    
  model_config = {
    'classes_len': 1, 
    'filter_class_ids': [0], 
    'input_shape': [640, 640], 
    'model_path': 'vsdkx/weights/ppl_detection_retrain_training_2.pt'
    }
    
  model_settings = {
    'conf_thresh': 0.5, 
    'device': 'cpu', 
    'iou_thresh': 0.4
    }
    
  zone_processor = ZoneProcessor(add_on_config, model_settings, model_config)
  ```

- `pre_process()` to blur the preselected zones:
  ```python
  addon_object = AddonObject(
    frame=np.array(RGB image), #Required RGB image in numpy format
    inference=None, #Usually it's none at this point since inference is populated at the `post_process` step
    shared={} #Usually it's {} at this point since shared is populated at the `post_process` step
    )
    
  zone_processor.pre_process(addon_object)
  
  # returns the addon_object, however at this step it is unchanged 
  ```
  
  This step of ZoneProcessor still returns the `addon_object` without making any changes to it. The logic dictates for the `addon_object` to be returned regardless, to allow extendibility of this method in the future.
 
 - `post_process()` to blur the preselected zones:
  ```python
  addon_object = AddonObject(
    frame=np.array(RGB image), #Required RGB image in numpy format
    inference=dict{boxes=[], classes=[], scores=[], extra={tracked_objects=int}}, #Contains the results of the inference and the amount of tracked_objects
    shared={tracked_objects=dict{}, trackable_objects_history=dict{}} #Contains the shared dictionaries generated on this frame by the ObjectTracker
    )
    
  zone_processor.post_process(addon_object)
  
  returns addon_object
  ```
  
  This step applies changes to the `addon_object.inference.extra`, which should look like this:
  ```python
  
  inference=dict{
    boxes=[],
    classes=[],
    scores=[],
    extra={
      tracked_objects=int,
      zoning={'zone_0': {'Person': [], 'Person_count': 0, 'objects_entered': {'Person': [], 'Person_count': 0}, 'objects_exited': {'Person': [], 'Person_count': 0}}, 'rest': {'Person': [], 'Person_count': 0}}
      }
    }, 
  
  ```
  
