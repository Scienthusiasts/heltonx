from .fcos_assigner import FCOSAssigner
from .yolov5_assigner import YOLOv5Assigner
from .hungarian_assigner import HungarianAssigner
from .max_iou_assigner import MaxIoUAssigner

__all__ = ["FCOSAssigner", "YOLOv5Assigner", "HungarianAssigner", "MaxIoUAssigner"]