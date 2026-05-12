import torch
import torch.nn as nn
from detection.utils.fcos_utils import *
from heltonx.utils.register import MODELS
from heltonx.utils.utils import multi_apply




@MODELS.register
class FCOSAssigner(nn.Module):
    """FCOS正负样本分配策略
    """
    def __init__(self, img_size, strides=[8, 16, 32, 64, 128], limit_ranges=[[-1,64],[64,128],[128,256],[256,512],[512,999999]], sample_radiu_ratio=1.5):
        """
            Args:
                img_size:            网络输入的图像尺寸 比如:[640, 640]
                strides:             多尺度特征相对于原图的下采样率
                limit_ranges:        每个尺度负责预测的gt框的尺寸范围
                sample_radiu_ratio:  stride * sample_radiu_ratio半径内的样本为正样本
        """
        super(FCOSAssigner, self).__init__()
        self.img_size = img_size
        self.strides=strides
        self.limit_ranges=limit_ranges
        self.sample_radiu_ratio=sample_radiu_ratio



    def forward_single(self, gt_boxes, gt_labels, stride, limit_range):
        """单层特征层的正负样本分配
        """
        feat_w = self.img_size[0] // stride
        feat_h = self.img_size[1] // stride
        h_mul_w = feat_w * feat_h
        bs = gt_boxes.shape[0]

        '''获得网格'''
        # grids.shape [w*h, 2]
        grids  = self.get_grids_single(feat_w, feat_h, stride).type_as(gt_boxes)
        x, y   = grids[:, 0], grids[:, 1]
        
        '''计算两两ltrb偏移量, 以及相应指标, 用于后续筛选'''
        # 求真实框的左上角和右下角相比于特征点的偏移情况, 得到每个anchor-point相比于每个gt的偏移量(两两计算)
        # [bs, h*w, gt_nums] = [1, h*w, 1] - [bs, 1, gt_nums]
        left_off    = x[None, :, None] - gt_boxes[...,0][:, None, :]
        top_off     = y[None, :, None] - gt_boxes[...,1][:, None, :]
        right_off   = gt_boxes[..., 2][:, None, :] - x[None, :, None]
        bottom_off  = gt_boxes[..., 3][:, None, :] - y[None, :, None]
        # [bs, h*w, gt_nums, 4]
        ltrb_off    = torch.stack([left_off, top_off, right_off, bottom_off],dim=-1)
        # 求每个框的面积 [bs, h*w, gt_nums]
        areas       = (ltrb_off[...,0] + ltrb_off[...,2]) * (ltrb_off[...,1] + ltrb_off[...,3])
        # 计算偏移量中的最小/最大值[bs, h*w, gt_nums]
        off_min     = torch.min(ltrb_off, dim=-1)[0]
        off_max     = torch.max(ltrb_off, dim=-1)[0]

        '''由于上面是计算了每个anchor-point相比于每个gt的两两偏移量, 因此会有很多冗余, 下面进行过滤'''
        # 1.mask_in_gtboxes筛选那些落在真实框内的特征点
        mask_in_gtboxes = off_min > 0
        # 2.mask_in_level筛选哪些gt适合在当前特征层进行检测
        # 修复边界条件检查，使用 >= 确保正确处理边界情况
        mask_in_level = (off_max >= limit_range[0]) & (off_max <= limit_range[1])
        # 在radiu半径圆内的grid作为正样本
        radiu       = stride * self.sample_radiu_ratio
        # 计算gt中心点与grid中心点两两距离
        # [1,h*w,1] - [bs, 1, gt_nums] --> [bs,h * w, gt_nums]
        # 计算GT的中心点坐标x, y
        gt_center_x = (gt_boxes[...,0] + gt_boxes[...,2]) / 2
        gt_center_y = (gt_boxes[...,1] + gt_boxes[...,3]) / 2
        # 计算grid中心点与gt中心点四个方向的垂直距离, 取最大的作为实际距离(为啥不用欧式距离?)
        c_left_off   = x[None, :, None] - gt_center_x[:, None, :]
        c_top_off    = y[None, :, None] - gt_center_y[:, None, :]
        c_right_off  = gt_center_x[:, None, :] - x[None, :, None]
        c_bottom_off = gt_center_y[:, None, :] - y[None, :, None]
        c_ltrb_off   = torch.stack([c_left_off, c_top_off, c_right_off, c_bottom_off],dim=-1)
        c_off_max    = torch.max(c_ltrb_off,dim=-1)[0]
        # 3.正样本与GT的中心点距离小于radiu
        mask_center = c_off_max < radiu
        # 联合考虑条件1.2.3, 筛选出正样本, 得到pos_mask为bool型
        pos_mask = mask_in_gtboxes & mask_in_level & mask_center
        # 将所有不是正样本的特征点，面积设成max [bs, h*w, gt_nums](其实就是标记为负样本)
        areas[~pos_mask] = 99999999
        # 选取特征点对应面积最小的框对应的索引 [bs, h*w, gt_nums] -> [bs, h*w] (其实就是筛选出每个grid匹配的GT框)
        areas_min_idx = torch.min(areas, dim = -1)[1]
        # 通过索引生成配对mask(每个grid匹配哪个gt) [bs, h*w, max_box_nums]
        match_mask = torch.zeros_like(areas, dtype=torch.bool).scatter_(-1, areas_min_idx.unsqueeze(dim=-1), 1)

        '''为每个grid分配最佳gt, 得到reg_targets, cls_targets, cls_targets'''
        # 筛选reg_targets [bs, h*w, max_box_nums, 4] -> [bs*h*w, 4] -> [bs, h*w, 4]
        reg_targets = ltrb_off[match_mask].reshape(bs, -1, 4)
        # 筛选cls_targets 
        # 将gt_labels[:, None, :]和areas广播为相同的形状 [bs, 1, max_box_nums] -> [bs, h*w, max_box_nums](在第二维度广播)
        _gt_labels, _  = torch.broadcast_tensors(gt_labels[:, None, :], areas.long())
        # 根据match_mask取出对应的正样本 [bs, h*w, max_box_nums] -> [bs, h*w] -> [bs, h*w, 1]
        cls_targets = _gt_labels[match_mask].reshape(bs, -1, 1)
        # 根据reg_targets生成对应grid的centerness value [bs, h*w]
        left_right_min = torch.min(reg_targets[..., 0], reg_targets[..., 2])
        left_right_max = torch.max(reg_targets[..., 0], reg_targets[..., 2])
        top_bottom_min = torch.min(reg_targets[..., 1], reg_targets[..., 3])
        top_bottom_max = torch.max(reg_targets[..., 1], reg_targets[..., 3])
        # 计算centerncss [bs, h*w, 1]
        cnt_targets= ((left_right_min * top_bottom_min) / (left_right_max * top_bottom_max + 1e-10)).sqrt().unsqueeze(dim=-1)
        # 排查形状是否正确
        assert reg_targets.shape == (bs,h_mul_w,4)
        assert cls_targets.shape == (bs,h_mul_w,1)
        assert cnt_targets.shape == (bs,h_mul_w,1)

        '''正负样本筛选'''
        # 那些任意一个gt都没配对上的样本为负样本, 否则为正样本 [bs, h*w, max_box_nums] -> [bs, h*w]
        pos_mask = pos_mask.long().sum(dim=-1) >= 1
        assert pos_mask.shape == (bs, h_mul_w)
        # 负样本对应位置的gt全设为-1
        cls_targets[~pos_mask] = -1
        cnt_targets[~pos_mask] = -1
        reg_targets[~pos_mask] = -1

        return cls_targets, cnt_targets, reg_targets



    def forward(self, gt_boxes, gt_labels):
        '''FCOS正负样本分配
            Args:
                gt_boxes:   gt框坐标  [[num_bboxes, 4=(x, y, w, h)], ..., [...]]
                gt_labels:  类别gt    [[num_bboxes], ..., [...]]

            Returns:
        '''
        # 把gt进行padding, 方便通过矩阵运算处理 [bs, max_num_bboxes, 4=(x, y, w, h)] [bs, max_num_bboxes]
        gt_boxes, gt_labels, _ = self.pad_gt_and_labels(gt_boxes, gt_labels)
        # x,y,w,h -> x0,y0,x1,y1
        gt_boxes[..., 2:] += gt_boxes[..., :2]

        '''并行处理每个特征层'''
        cls_targets_all, cnt_targets_all, reg_targets_all = multi_apply(
            self.forward_single,
            # 每层共享相同gt
            [gt_boxes] * len(self.strides),      
            [gt_labels] * len(self.strides),
            self.strides,
            self.limit_ranges
        )
        # 合并不同层的结果(dense GT)
        cls_targets_all = torch.cat(cls_targets_all, dim=1).reshape(-1, 1)
        cnt_targets_all = torch.cat(cnt_targets_all, dim=1).reshape(-1, 1)
        reg_targets_all = torch.cat(reg_targets_all, dim=1).reshape(-1, 4)

        return cls_targets_all, cnt_targets_all, reg_targets_all




    def get_grids_single(self, w, h, stride):
        """生成某个尺度的网格中心点坐标
            Args:
                w:      当前尺度特征图的w
                h:      当前尺度特征图的h
                stride: 当前尺度特征图相对原图的下采样率
        """
        shifts_x = torch.arange(0, w * stride, stride, dtype=torch.float32)
        shifts_y = torch.arange(0, h * stride, stride, dtype=torch.float32)

        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')

        shift_x = torch.reshape(shift_x, [-1])
        shift_y = torch.reshape(shift_y, [-1])
        grid    = torch.stack([shift_x, shift_y], -1) + stride // 2

        return grid



    def pad_gt_and_labels(self, gt_boxes_list, labels_list, pad_value=-1):
        """
        将 batch 中的 gt_boxes 和 labels padding 到相同长度 (max_box_nums)
        方便后续 batch-level 并行计算。
            Args:
                gt_boxes_list: list[Tensor], 每个元素形状为 [num_boxes, 4]
                labels_list:   list[Tensor], 每个元素形状为 [num_boxes]
                pad_value:     int, padding 的填充值(默认 -1)

            Returns:
                gt_boxes: Tensor, [bs, max_box_nums, 4]
                labels:   Tensor, [bs, max_box_nums]
                mask:     BoolTensor, [bs, max_box_nums]  表示哪些位置是真实 box(True)
        """
        bs = len(gt_boxes_list)
        assert bs == len(labels_list), "gt_boxes_list 与 labels_list 长度不一致"

        # 当前 batch 中最大 GT 数
        max_box_nums = max([b.shape[0] for b in gt_boxes_list])

        padded_boxes = []
        padded_labels = []
        masks = []

        for i in range(bs):
            n = gt_boxes_list[i].shape[0]
            # 初始化 padding
            pad_boxes = torch.full((max_box_nums, 4), pad_value, dtype=gt_boxes_list[i].dtype, device=gt_boxes_list[i].device)
            pad_cls   = torch.full((max_box_nums,), pad_value, dtype=labels_list[i].dtype, device=labels_list[i].device)
            mask      = torch.zeros((max_box_nums,), dtype=torch.bool, device=gt_boxes_list[i].device)

            # 填充前 n 个为真实值
            pad_boxes[:n] = gt_boxes_list[i]
            pad_cls[:n]   = labels_list[i]
            mask[:n]      = True

            padded_boxes.append(pad_boxes)
            padded_labels.append(pad_cls)
            masks.append(mask)

        # 拼接成 batch tensor
        gt_boxes = torch.stack(padded_boxes, dim=0)    # [bs, max_box_nums, 4]
        gt_labels  = torch.stack(padded_labels, dim=0)  # [bs, max_box_nums]
        mask     = torch.stack(masks, dim=0)           # [bs, max_box_nums]

        return gt_boxes, gt_labels, mask