### Addon Config
```yaml
remove_areas: [ [ 0, 0, 736, 163 ]], # Array with zone(s) for exclusion coordinates
zones: [[[190, 250], [450, 250], [450, 420], [225, 450], [225, 350], [190, 350], [190, 250]],
                    [[480, 190], [950, 190], [1250, 380], [780, 430], [480, 190]]], # Array with zone(s) coordinates
iou_thresh: [0.85, 0.90], # List with IOU thresholds for IoU score per pre-configured zone
```
It also require these properties inside model config dictionary:
```yaml
filter_class_ids:
```