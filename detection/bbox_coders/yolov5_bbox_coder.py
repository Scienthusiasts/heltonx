import numpy as np
import torch
import torch.nn as nn
from heltonx.utils.register import MODELS


@MODELS.register
class YOLOv5BBoxCoder(nn.Module):
    """YOLOv5 bbox编解码操作.
       encode: 将gt编码成能直接和网络预测结果计算损失的特征(由assigner完成)
       decode: 对网络预测的结果解码成真实的基于原图尺寸的结果

    Args:
        img_size (List[int]): 输入图像尺寸，如 [640, 640]
        anchors (List[List[int]]): 先验框列表，如 [[10, 13], [16, 30], ...]
        anchors_mask (List[List[int]]): 先验框掩码，如 [[0,1,2], [3,4,5], [6,7,8]]
        nc (int): 类别数量
    """
    def __init__(self, img_size, anchors, anchors_mask, nc):
        super().__init__()
        self.img_size = img_size
        self.anchors = anchors
        self.anchors_mask = anchors_mask
        self.nc = nc

    def decode(self, inputs):
        """将预测结果解码为真实结果(推理时使用)

        将多尺度特征图中的 offset 作用到 anchor 上,
        解码得到归一化的 cxcywh 格式预测结果.

        Args:
            inputs (List[Tensor]): 多尺度特征图列表,
                如 [bs, 3*(5+nc), H_i, W_i]

        Returns:
            List[Tensor]: 各尺度解码后的预测结果,
                每个 Tensor 形状为 [bs, num_anchors, 5+nc],
                坐标格式为归一化的 cxcywh
        """
        # 各尺度固定下采样率, 不依赖img_size, 避免非正方形图像时stride计算错误
        strides = [8, 16, 32]
        outputs = []
        for i, input in enumerate(inputs):
            batch_size = input.size(0)
            input_height = input.size(2)
            input_width = input.size(3)
            stride = strides[i]
            # scaled_anchors是anchors相对于特征图的尺寸
            scaled_anchors = [(a_w / stride, a_h / stride) for a_w, a_h in self.anchors]
            # 将预测结果reshape [bs, 3*(5+nc), w, h] -> [bs, 3, w, h, 5+nc]
            prediction = input.view(batch_size, len(self.anchors_mask[i]),
                                    5 + self.nc, input_height, input_width).permute(0, 1, 3, 4, 2).contiguous()
            # 得到offset(通过sigmoid将offset的范围限制在(0,1)之间)
            # anchors的中心位置调整(平移因子)
            dx = torch.sigmoid(prediction[..., 0])
            dy = torch.sigmoid(prediction[..., 1])
            # anchors的宽高调整参数(缩放因子)
            dw = torch.sigmoid(prediction[..., 2])
            dh = torch.sigmoid(prediction[..., 3])
            # obj置信度
            conf = torch.sigmoid(prediction[..., 4])
            # cls置信度
            pred_cls = torch.sigmoid(prediction[..., 5:])
            # 将offset作用到anchor上进行解码得到最终预测结果(基于特征图尺寸的cxcywh)
            pred_boxes = self.reg2box(batch_size, i, dx, dy, dh, dw, scaled_anchors, input_height, input_width)
            # 将输出结果转换成归一化坐标norm(cxcywh)
            _scale = torch.Tensor([input_width, input_height, input_width, input_height]).type_as(dx)
            output = torch.cat((pred_boxes.view(batch_size, -1, 4) / _scale,
                                conf.view(batch_size, -1, 1),
                                pred_cls.view(batch_size, -1, self.nc)), -1)
            outputs.append(output.data)
        return outputs

    def reg2box(self, bs, l, dx, dy, dh, dw, scaled_anchors, in_h, in_w):
        """将预测的offset作用到anchor上进行解码(训练和推理均使用)

        YOLOv5的核心解码操作: 对网格的左上角点和宽高做微调.
        - dx: 将范围调整成(-0.5, 1.5), anchor中心点x可调整范围为相邻一个网格单位
        - dy: 将范围调整成(-0.5, 1.5), anchor中心点y可调整范围为相邻一个网格单位
        - dw: 将范围调整成(0, 4), anchor宽可调整范围为其本身的4倍以内
        - dh: 将范围调整成(0, 4), anchor高可调整范围为其本身的4倍以内

        Args:
            bs (int): batch size
            l (int): 当前层索引 (0: P3, 1: P4, 2: P5)
            dx (Tensor): 预测的x偏移量, 形状 [bs, 3, h, w]
            dy (Tensor): 预测的y偏移量, 形状 [bs, 3, h, w]
            dh (Tensor): 预测的h缩放因子, 形状 [bs, 3, h, w]
            dw (Tensor): 预测的w缩放因子, 形状 [bs, 3, h, w]
            scaled_anchors (List[Tuple[float, float]]): 相对于特征图尺寸的anchor列表
            in_h (int): 特征图高度
            in_w (int): 特征图宽度

        Returns:
            pred_boxes (Tensor): 解码后的预测框, 形状 [bs, 3, h, w, 4], 格式为 cxcywh
        """
        # 生成网格, 先验框中心, 网格左上角(特征图尺寸下的绝对坐标)
        grid_x = torch.linspace(0, in_w - 1, in_w).repeat(in_h, 1).repeat(
            int(bs * len(self.anchors_mask[l])), 1, 1).view(dx.shape).type_as(dx)
        grid_y = torch.linspace(0, in_h - 1, in_h).repeat(in_w, 1).t().repeat(
            int(bs * len(self.anchors_mask[l])), 1, 1).view(dy.shape).type_as(dx)

        # 按照网格格式生成先验框的宽高(特征图尺寸下的绝对坐标)
        scaled_anchors_l = np.array(scaled_anchors)[self.anchors_mask[l]]
        anchor_w = torch.Tensor(scaled_anchors_l).index_select(1, torch.LongTensor([0])).type_as(dx)
        anchor_h = torch.Tensor(scaled_anchors_l).index_select(1, torch.LongTensor([1])).type_as(dx)
        anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(dw.shape)
        anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(dh.shape)

        # 将offset作用到anchor上进行解码
        # NOTE: 不要写dx.data, 否则没有梯度, 导致训练时box_loss不收敛
        dx = dx * 2. - 0.5
        dy = dy * 2. - 0.5
        dw = (dw * 2) ** 2
        dh = (dh * 2) ** 2
        pred_boxes = torch.zeros((bs, len(self.anchors_mask[l]), in_h, in_w, 4), device=dx.device)
        pred_boxes[..., 0] = grid_x + dx
        pred_boxes[..., 1] = grid_y + dy
        pred_boxes[..., 2] = anchor_w * dw
        pred_boxes[..., 3] = anchor_h * dh
        return pred_boxes
