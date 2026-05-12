import torch
import torch.nn as nn
from detection.utils.fcos_utils import *
from heltonx.utils.register import MODELS
from heltonx.utils.utils import multi_apply




@MODELS.register
class YOLOv5Assigner(nn.Module):
    """FCOS正负样本分配策略
    """
    def __init__(self, img_size, anchors, anchors_mask, threshold=4):
        """
            Args:
                img_size:      网络接受的输入图像的大小[640, 640]
                anchors:       [[w0, h0], ..., [w8, h8]]
                anchors_mask:  [[0,1,2], [3,4,5], [6,7,8]]
                threshold:     anchor作为正样本时和GT的长宽比在 (1/threshold~threshol
        """
        super(YOLOv5Assigner, self).__init__()
        self.img_size = img_size
        self.anchors = anchors
        self.anchors_mask = anchors_mask
        self.threshold = threshold



    def assgin_single(self, gt_bboxes, gt_labels, bbox_attrs):
        ''' - YOLOv5正负样本分配策略(根据gt与anchors的长宽比例, 是基于anchor_base的静态分配方法).
            - 该函数仅针对一张图片的分配, 因为在dataset里调用, 所以没实现batch的.
            - YOLOv5计算损失时只对正样本计算分类和回归损失, 唯一用到负样本的地方只有obj_map即这一部分.

            # Args
                - targets:         [num_gts, 5]
                - input_shape:     网络接受的输入图像的大小[640, 640]
                - anchors:         [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]]
                - anchors_mask:    [[0,1,2], [3,4,5], [6,7,8]]
                - bbox_attrs:      5 + num_cls
                - threshold:       anchor作为正样本时和GT的长宽比在 (1/threshold~threshold) 之间

            # Returns
                - output_targets:  分配结果 [[3(每个尺度3个anchor), w, h, 5+cls_num],[...],[...]] 
        '''
        '''初始化'''
        # s代表特征图相对原图的下采样率
        s={0:8, 1:16, 2:32}
        num_layers  = len(self.anchors_mask)
        img_size = np.array(self.img_size, dtype='int32')
        # grid_shapes是每一层特征图的尺寸  [[80,80],[40,40],[20,20]]
        grid_shapes = [img_size // s[l] for l in range(num_layers)] 
        # 初始化 output_targets = [[...],[...],[...]] 
        # 为每个特征层的每个锚框预留空间存储预测目标 output_targets[i].shape = (3, 80, 80, 5+cls_num), (3, 40, 40, 5+cls_num), (3, 20, 20, 5+cls_num) 5:(x0, y0, x1, y1, )
        output_targets = [np.zeros((len(self.anchors_mask[l]), grid_shapes[l][0], grid_shapes[l][1], bbox_attrs), dtype='float32') for l in range(num_layers)] 
        # 初始化 box_best_ratio = [[...],[...],[...]] 
        # 存储每个网格最佳锚框的匹配比例 box_best_ratio[i].shape = (3, 80, 80), (3, 40, 40), (3, 20, 20)
        best_ratio_scores = [np.zeros((len(self.anchors_mask[l]), grid_shapes[l][0], grid_shapes[l][1]), dtype='float32') for l in range(num_layers)] 
        # 没有GT, 则返回全0分配(全是负样本)
        if len(gt_bboxes) == 0: return output_targets

        '''正负样本分配'''
        '''遍历每个特征层进行正负样本的分配(这样一个gt在不同特征层可能都会有匹配上的anchors)'''
        for l in range(num_layers):
            # 特征层的高和宽
            feat_h, feat_w = grid_shapes[l]
            # 将anchor调整到当前特征层的尺度
            lvl_anchors = np.array(self.anchors) / s[l] 
            scaled_targets = np.zeros((gt_bboxes.shape[0], 5)) # [num_gts, 5]
            # 对每个真实框计算其在当前特征层的位置(基于特征图尺寸的cxcywh)
            scaled_targets[:, [0,2]] = gt_bboxes[:, [0,2]] * feat_w
            scaled_targets[:, [1,3]] = gt_bboxes[:, [1,3]] * feat_h
            scaled_targets[:, 4]     = gt_labels
            '''计算GT和anchor两两之间的长宽比'''
            # np.expand_dims(a, 1) 在a的第1维度添加一个新的维度: a = [bs, w, h] -> [bs, 1, w, h]
            # ratios_of_gt_anchors代表每一个真实框和每一个先验框的宽与宽的比值、高与高的比值  [num_gts, 1, 2] [1, 9, 2] -> [num_gts, 9, 2]
            ratios_of_gt_anchors = np.expand_dims(scaled_targets[:, 2:4], 1) / np.expand_dims(lvl_anchors, 0)
            # 合并比率信息[num_true_box, 9, 4](取倒数是因为0.25~4的范围都行)
            aspect_ratios = np.concatenate([ratios_of_gt_anchors, 1 / ratios_of_gt_anchors], axis = -1)
            # 在宽比值、高比值这2个比值中，取最极端的一个比值，作为GT框和anchor的比值 [num_true_box, 9]
            max_ratios = np.max(aspect_ratios, axis = -1)
            
            '''对每个GT匹配对应的anchors'''
            for gt_id, ratio in enumerate(max_ratios):
                # 判断是否符合正样本条件
                valid_anchors = ratio < self.threshold
                # 强制保留最佳匹配锚框为正样本
                valid_anchors[np.argmin(ratio)] = True
                '''对每个GT一一检查当前层每个尺寸的anchor是否匹配, 是则作为正样本 k是不同尺寸的anchor, k=3'''
                for k, mask in enumerate(self.anchors_mask[l]):
                    # 如果不符合正样本条件，跳过当前锚框组
                    if not valid_anchors[mask]:
                        continue
                    # 获得真实框属于哪个网格点(会取整)
                    # 计算当前GT点的左上角x0对应网格的x索引，使用 clamp 防止越界
                    i = int(np.floor(np.clip(scaled_targets[gt_id, 0], 0, feat_w - 1)))
                    # 计算当前GT点的左上角y0对应网格的y索引，使用 clamp 防止越界
                    j = int(np.floor(np.clip(scaled_targets[gt_id, 1], 0, feat_h - 1)))
                    # 获取当前+附近的网格点偏移(这一步为了增加正样本数量)
                    offsets = self.get_near_points(scaled_targets[gt_id, 0], scaled_targets[gt_id, 1], i, j)
                    '''对每个正样本, 计算在特征图上的位置与gt value并记录到output_targets中'''
                    for offset in offsets:
                        # 计算当前GT点的左上角x0对应网格的x索引(包括相邻网格点)
                        local_i = i + offset[0]
                        # 计算当前GT点的左上角y0对应网格的y索引(包括相邻网格点)
                        local_j = j + offset[1]
                        # 检查网格点是否超出特征图范围
                        if local_i >= feat_w or local_i < 0 or local_j >= feat_h or local_j < 0:
                            continue
                        # 如果该网格点已经有更佳匹配的锚框, 则跳过
                        if best_ratio_scores[l][k, local_j, local_i] != 0:
                            if best_ratio_scores[l][k, local_j, local_i] > ratio[mask]:
                                output_targets[l][k, local_j, local_i, :] = 0
                            else:
                                continue
                        
                        '''将正样本的信息写入output_targets'''
                        # 真实框的种类
                        c = int(scaled_targets[gt_id, 4])
                        # cx, cy, w, h
                        output_targets[l][k, local_j, local_i, :4] = scaled_targets[gt_id, :4]
                        # obj_map=1, 表示对应位置存在目标
                        output_targets[l][k, local_j, local_i, 4] = 1     
                        # 对应类别置为1, 其余为0 
                        output_targets[l][k, local_j, local_i, c + 5] = 1  
                        # 记录当前网格点的最佳匹配IOU
                        best_ratio_scores[l][k, local_j, local_i] = ratio[mask]
                        
        return output_targets




    def get_near_points(self, x, y, i, j):
        '''获得正样本周围的相邻网格点,也作为正样本（与官方一致，最多5个点）
           官方逻辑: 当GT中心靠近网格边界时，同时扩展到两个相邻方向，
           最多产生5个正样本网格点: [0,0], [1,0], [0,1], [-1,0], [0,-1]
        '''
        sub_x = x - i
        sub_y = y - j
        # 始终包含中心点
        offsets = [[0, 0]]
        # 当GT中心靠近网格右边界(x > 0.5)时，扩展到右侧网格
        if sub_x > 0.5:
            offsets.append([1, 0])
        # 当GT中心靠近网格左边界(x < 0.5)时，扩展到左侧网格
        else:
            offsets.append([-1, 0])
        # 当GT中心靠近网格下边界(y > 0.5)时，扩展到下方网格
        if sub_y > 0.5:
            offsets.append([0, 1])
        # 当GT中心靠近网格上边界(y < 0.5)时，扩展到上方网格
        else:
            offsets.append([0, -1])
        return offsets


