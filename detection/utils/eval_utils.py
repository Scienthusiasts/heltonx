# coding=utf-8
import os
import json
import torch
from torch import nn
from tqdm import tqdm
from detection.utils.metrics import *
from detection.utils.utils import OpenCVDrawBox, map_boxes_to_origin_size
from heltonx.utils.register import EVALPIPELINES
from heltonx.utils.utils import to_device








@EVALPIPELINES.register
class DetectionEvalPipeline():
    '''一个epoch的评估(基于验证集)
    '''
    def __init__(self, ):
        # 图像均值 标准差
        self.mean = np.array([0.485, 0.456, 0.406]) 
        self.std = np.array([[0.229, 0.224, 0.225]]) 

    def __call__(self, runner, model=None):
        # 直接从传入的类中获取参数(避免每个任务的特殊化):
        device = runner.device
        if model==None:
            model = runner.model
        model.eval()

        valid_dataloader = runner.valid_dataloader
        json_path = runner.valid_dataset.ann_json_path
        inv_map = runner.valid_dataset.inv_map
        log_dir = runner.log_dir
        anns_dict = []

        with torch.no_grad():
            for batch_datas in tqdm(valid_dataloader):
                '''推理一张图'''
                # 确保 batch_datas 的所有数据已经在 self.device 上(batch_datas的组织形式是list)
                batch_datas = to_device(batch_datas, device, non_blocking=True)
                imgs, imgs_ids, raw_size = batch_datas[0], batch_datas[3], batch_datas[4]
                # boxes, box_scores, box_classes = model.infer(imgs, vis_heatmap=True, save_vis_path='./det_res.jpg')
                boxes, box_scores, box_classes = model.infer(imgs)
                
                # 将box坐标(对应有黑边的图, 且默认padding成正方形)映射回无黑边的原始图像
                if (len(boxes) > 0):
                    boxes = map_boxes_to_origin_size(boxes, raw_size[0], imgs.shape[2])
                '''生成符合COCO格式的推理结果'''
                # 将预测结果转化为COCO格式下'annotations'下的一个字段(注意COCO的box格式是xywh)
                for box, score, cat_id in zip(boxes, box_scores, box_classes):
                    # 如果像COCO数据集一样categories id没有按顺序来，则还得映射回去
                    if inv_map!=None: cat_id = inv_map[cat_id]
                    anns_dict.append(
                        {
                            "image_id":int(imgs_ids[0]),
                            "category_id":int(cat_id),
                            "bbox":[
                                float(box[0]),
                                float(box[1]),
                                float(box[2] - box[0]),
                                float(box[3] - box[1])
                            ],
                            "score":float(score),
                        }
                    )

        # 将anns_dict保存为json文件(eval_tmp)
        pred_json_path = os.path.join(log_dir, 'pred_tmp.json')
        with open(pred_json_path, 'w') as json_file:
            json.dump(anns_dict, json_file)
        # 计算 mAP, ap_50
        m_ap, ap_50 = evalCOCOmAP(json_path, pred_json_path)
        evaluations = dict(
            val_map=m_ap, 
            val_ap50 = ap_50
        )
        # 后续保存best_ckpt以val_flag_metric为参考
        flag_metric_name = "val_map"
        return evaluations, flag_metric_name