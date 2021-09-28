#coding:utf-8
# visual anchors

import cv2
import numpy as np
import torch
import os,glob
import os.path as osp
import codecs
from tqdm import tqdm

# pre_anchors=[[23,28, 68,57, 46,113],
#     [143,44, 135,78, 98,127],
#     [236,112, 148,209, 243,282]]

# pre_anchors=[[30.,10.,  19.,66.,  71.,23.],  # P3/8
#    [125.,30.,  30.,135.,  212.,39.],  # P4/16
#   [45.,248.,  321.,72.,  69.,351.] ] # P5/32
pre_anchors = [
    [9., 4., 3., 9., 5., 11.],
    [10., 5., 18., 8., 19., 9.],
    [21., 10., 25., 11., 37., 15.],
    [39., 17., 49., 19., 65., 22.]
]
stride = [
    4.,
    8.,
    16.,
    32.
]
anchor_t = 3.0  # anchor-multiple threshold

na = len(pre_anchors[0]) // 2  # number of anchors
nl = len(pre_anchors)  # # number of detection layers
pre_anchors = torch.tensor(pre_anchors).view(nl, na, 2)


def longsideformat2cvminAreaRect(x_c, y_c, longside, shortside, theta_longside):
    '''
    trans longside format(x_c, y_c, longside, shortside, θ) to minAreaRect(x_c, y_c, width, height, θ)
    两者区别为:
            当opencv表示法中width为最长边时（包括正方形的情况），则两种表示方法一致
            当opencv表示法中width不为最长边 ，则最长边表示法的角度要在opencv的Θ基础上-90度
    @param x_c: center_x
    @param y_c: center_y
    @param longside: 最长边
    @param shortside: 最短边
    @param theta_longside: 最长边和x轴逆时针旋转的夹角，逆时针方向角度为负 [-180, 0)
    @return: ((x_c, y_c),(width, height),Θ)
            x_c: center_x
            y_c: center_y
            width: x轴逆时针旋转碰到的第一条边最长边
            height: 与width不同的边
            theta: x轴逆时针旋转与width的夹角，由于原点位于图像的左上角，逆时针旋转角度为负 [-90, 0)
    '''
    if (theta_longside >= -180 and theta_longside < -90):  # width is not the longest side
        width = shortside
        height = longside
        theta = theta_longside + 90
    else:
        width = longside
        height =shortside
        theta = theta_longside

    if theta < -90 or theta >= 0:
        print('当前θ=%.1f，超出opencv的θ定义范围[-90, 0)' % theta)

    return ((x_c, y_c), (width, height), theta)


def txt2targets(txt_path):
    txt_file = codecs.open(txt_path, encoding='utf-8')
    imgid = [0]
    targets = []
    for line in txt_file:
        # import pdb;pdb.set_trace()
        line = line.strip()
        row_list = line.split()
        target = imgid + [float(x) for x in row_list[:]]
        targets.append(target)
    txt_file.close()
    targets = torch.tensor(targets)
    # import pdb;pdb.set_trace()
    return targets


def match_anchors(targets,imgsz):
    """
        @ target: gt target [[index, cls, x, y, l, s, theta], ...]
        @ imgsz:
    """

    nt = len(targets)  # number of target
    # init feature map in different stride size
    p = [torch.ones(imgsz // int(down_sample), imgsz // int(down_sample)) for down_sample in stride]
    tcls, tbox, indices, anch = [], [], [], []
    gain = torch.ones(8, device=targets.device)  # normalized to gridspace gain
    # ai.shape = (3, nt) 生成anchor索引  anchor index; ai[0]全等于0. ai[1]全等于1. ai[2]全等于2.用于表示当前gtbox和当前层哪个anchor匹配
    ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
    targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # copy target by $na times and cat to the right of each row
    g = 0.5  # bias 网格中心偏移
    # 附近的四个网格 off.shape = (5, 2)
    off = torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

    for i in range(nl):
        anchors = pre_anchors[i]/stride[i]
        gain[2:6] = torch.tensor(p[i].shape)[[1, 0, 1, 0]]  # xyxy gain 把p[i]wh维度的数据赋给gain
        num_h, num_w = p[i].shape[0: 2]

        # Match targets to anchors
        t = targets * gain  # 将labels的归一化的xywh从基于0~1映射到基于特征图的xywh 即变成featuremap尺度
        if nt:
            # Matches
            """
                GT的wh与anchor的wh做匹配，筛选掉比值大于hyp['anchor_t']的(这应该是yolov5的创新点)targets，从而更好的回归(与新的边框回归方式有关)
                若gt_wh/anhor_wh 或 anhor_wh太大/gt_wh 超出hyp['anchor_t']，则说明当前target与所选anchor形状匹配度不高，该物体宽高过于极端，不应强制回归，将该处的labels信息删除，在该层预测中认为是背景
    
                由于yolov3回归wh采用的是out=exp(in)，这很危险，因为out=exp(in)可能会无穷大，就会导致失控的梯度，不稳定，NaN损失并最终完全失去训练；
                (当然原yolov3采用的是将targets进行反算来求in与网络输出的结果，就问题不大，但采用iou loss，就需要将网络输出算成out来进行loss求解，所以会面临这个问题)；
                所以作者采用新的wh回归方式:
                (wh.sigmoid() * 2) ** 2 * anchors[i], 原来yolov3为anchors[i] * exp(wh)
                将标签框与anchor的倍数控制在0~4之间；
                hyp.scratch.yaml中的超参数anchor_t=4，所以也是通过此参数来判定anchors与标签框契合度；
            """
            r = t[:, :, 4:6] / anchors[:, None]  # wh ratio  获取gt bbox与anchor的wh比值
            j = torch.max(r, 1. / r).max(2)[0] < anchor_t  # compare with anchor_t
            # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
            t = t[j]  # filter 筛选出与anchor匹配的targets;

            # Offsets
            # 得到筛选后的GT的中心点坐标xy-featuremap尺度(相对于左上角的), 其shape为(M, [x_featuremap, y_featuremap])
            gxy = t[:, 2:4]  # grid xy
            # 得到筛选后的GT的中心点相对于右下角的坐标, 其shape为(M, 2)
            # gain = [1, 1, w, h, w, h, 1, 1]
            gxi = gain[[2, 3]] - gxy  # inverse
            """
                把相对于各个网格左上角x<g=0.5,y<0.5和相对于右下角的x<0.5,y<0.5的框提取出来；
                也就是j,k,l,m，在选取gij(也就是标签框分配给的网格的时候)对这四个部分的框都做一个偏移(减去上面的off),也就是下面的gij = (gxy - offsets).long()操作；
                再将这四个部分的框与原始的gxy拼接在一起，总共就是五个部分；
                也就是说：①将每个网格按照2x2分成四个部分，每个部分的框不仅采用当前网格的anchor进行回归，也采用该部分相邻的两个网格的anchor进行回归；
                原yolov3就仅仅采用当前网格的anchor进行回归；
                估计是用来缓解网格效应，但由于v5没发论文，所以也只是推测，yolov4也有相关解决网格效应的措施，是通过对sigmoid输出乘以一个大于1的系数；
                这也与yolov5新的边框回归公式相关；
                由于①，所以中心点回归也从yolov3的0~1的范围变成-0.5~1.5的范围；
                所以中心点回归的公式变为：
                xy.sigmoid() * 2. - 0.5 + cx
            """
            j, k = ((gxy % 1. < g) & (gxy > 1.)).T  # 判断筛选后的GT中心坐标是否相对于各个网格的左上角偏移<0.5 同时 判断 是否不处于最左上角的网格中 （xy两个维度）
            l, m = ((gxi % 1. < g) & (gxi > 1.)).T  # 判断筛选后的GT中心坐标是否相对于各个网格的右下角偏移<0.5 同时 判断 是否不处于最右下角的网格中 （xy两个维度）
            j = torch.stack((torch.ones_like(j), j, k, l, m)) # shape(5, M) 其中元素为True或False
            # 由于预设的off为5 先将t在第一个维度重复5次 shape(5, M, 8),现在选出最近的3个(包括 0，0 自己)
            t = t.repeat((5, 1, 1))[j]  # 得到经过第二次筛选的框(3*M, 8)
            offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]  # 选出最近的三个网格 offsets.shape=(3*M, 2)
        else:
            t = targets[0]
            offsets = 0

        # Define
        b, c = t[:, :2].long().T  # image, class
        gxy = t[:, 2:4]  # grid xy 不考虑offset时负责预测的网格坐标 xy_featuremap 即feature尺度上的gt真实xy
        gwh = t[:, 4:6]  # grid wh
        g_theta = t[:, 6].unsqueeze(1)
        gij = (gxy - offsets).long()  # featuremap上的gt真实xy坐标减去偏移量再取整  即计算当前label落在哪个网格坐标上
        gi, gj = gij.T  # grid xy indices 将x轴坐标信息压入gi 将y轴坐标索引信息压入gj 负责预测网格具体的整数坐标 比如 23， 2

        # Append
        a = t[:, 7].long()  # anchor indices
        # import pdb;pdb.set_trace()
        indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
        tbox.append(torch.cat((gxy - gij, gwh, g_theta), 1))  # box
        anch.append(anchors[a])  # anchors
        tcls.append(c)  # class
    # import pdb;pdb.set_trace()
    return tcls, tbox, indices, anch


def draw_matched_h_anchor(img, indices, anch):
    # get different layer
    for nl in range(len(indices)):
        cur_indice = indices[nl]
        cur_anch = anch[nl]
        # import pdb;pdb.set_trace()
        # get every matched anchor
        # get anchor index
        grid_x, grid_y = cur_indice[3].view(-1, 1), cur_indice[2].view(-1, 1)
        match_gridxy = torch.cat((grid_x, grid_y), 1).float()
        # cat anchor x,y center and anchor w,h
        match_anchor = stride[nl] * torch.cat((match_gridxy, cur_anch), 1)

        match_anchor = match_anchor.numpy().astype(np.int32)
        # import pdb;pdb.set_trace()
        color = [
            (255, 0, 0), (255, 0, 50), (255, 0, 100),
            (0, 0, 255), (50, 0, 255), (100, 0, 255),
            (50, 255, 0), (0, 255, 0), (100, 255, 0),
        ]
        for i in range(len(match_anchor)):
            x, y, w, h = match_anchor[i]
            x1, y1, x2, y2 = x - w // 2, y - h // 2, x + w // 2, y + h // 2
            # import pdb;pdb.set_trace()
            cv2.rectangle(img, (x1, y1), (x2, y2), color[i], 1, 1)
            cv2.putText(img, str(i), (x1, y1), 0, 0.5, [220, 220, 220], thickness=1, lineType=16)


def draw_target_regresion_box(img, indices, tbox):
    '''
    verify regresion target is true
    '''
    for nl in range(len(indices)):
        cur_indice = indices[nl]
        cur_tbox = tbox[nl]
        if len(cur_tbox) < 1:
            continue
        match_tbox = cur_tbox.clone()
        # import pdb;pdb.set_trace()
        # get every matched anchor
        # get anchor index
        grid_x, grid_y = cur_indice[3].view(-1, 1), cur_indice[2].view(-1, 1)
        match_gridxy = torch.cat((grid_x, grid_y), 1).float()
        # x,y
        # import pdb;pdb.set_trace()
        match_tbox[:, 0:2] = match_gridxy + cur_tbox[:, 0:2]
        # match_tbox[:, 4] *= 90.

        # cat anchor x,y center and anchor w,h
        match_tbox[:, 0:4] = stride[nl] * match_tbox[:, 0:4]

        match_tbox = match_tbox.numpy().astype(np.int32)
        # import pdb;pdb.set_trace()
        for i in range(len(match_tbox)):
            x, y, w, h, theta = match_tbox[i]
            rect = longsideformat2cvminAreaRect(x, y, w, h, (theta - 179.9))
            # import pdb;pdb.set_trace()
            bbox = cv2.boxPoints(rect).reshape((-1, 1, 2)).astype(np.int32)
            put_text = 'pos: {:.1f} {:.1f} {:.1f} {:.1f}'.format(x, y, w, h)
            cv2.putText(img, put_text, (50, 100 + nl * 50), 1, cv2.FONT_HERSHEY_PLAIN, (255, 255, 0), 1)
            cv2.putText(img, str(theta), (400 + 50 * i, 100 + nl * 50), 1, cv2.FONT_HERSHEY_PLAIN, (255, 255, 0), 1)
            # import pdb;pdb.set_trace()
            cv2.polylines(img, [bbox], True, (255, 255, 0), 2, 2)


def draw_targets(img,targets):
    '''
    draw targets box
    '''
    show_targets = targets.clone()
    show_targets = show_targets.cpu().numpy()
    imgsz = img.shape[0]
    for i in range(len(show_targets)):
        x, y, w, h, theta = show_targets[i][2:]
        x *= imgsz
        y *= imgsz
        w *= imgsz
        h *= imgsz
        # theta*=90
        # import pdb;pdb.set_trace()

        rect = longsideformat2cvminAreaRect(x, y, w, h, (theta-179.9))

        put_text = 'pos: {:.1f} {:.1f} {:.1f} {:.1f} {:.2f}'.format(x, y, w, h, theta)
        cv2.putText(img, put_text, (50, 50), 1, cv2.FONT_HERSHEY_PLAIN, (0, 255, 0), 1)
        bbox = cv2.boxPoints(rect).reshape((-1, 1, 2)).astype(np.int32)

        cv2.polylines(img, [bbox], True, (0, 255, 0), thickness=3)


def vis_matched_anchor(targets, img):
    imgsz=img.shape[0]

    tcls, tbox, indices, anch = match_anchors(targets, imgsz)
    # import pdb;pdb.set_trace()

    draw_matched_h_anchor(img, indices, anch)
    cv2.imshow("", img)
    cv2.waitKey()
    draw_targets(img, targets)
    draw_target_regresion_box(img, indices, tbox)

    return img


def main(image_dir, label_dir, vis_dir):
    imglist = glob.glob(osp.join(image_dir, '*.png'))
    # import pdb;pdb.set_trace()
    for imgpath in tqdm(imglist):
        basename, suffix = osp.splitext(osp.basename(imgpath))
        annotpath = osp.join(label_dir, '{}.txt'.format(basename))
        vis_name = osp.join(vis_dir, '{}.jpg'.format(basename))

        img = cv2.imread(imgpath)

        targets = txt2targets(annotpath)

        # matched_anchors=match_anchors(targets,img.shape[0])
        show_img = vis_matched_anchor(targets, img)

        cv2.imwrite(vis_name, show_img)
        # import pdb;pdb.set_trace()
# main()
imgdir = '../DOTA_demo_view/images/train'
annot_dir = '../DOTA_demo_view/labels/train'

vis_dir = '../DOTA_demo_view/vis_anchor_dir/train/'

if not osp.exists(vis_dir):
    os.makedirs(vis_dir)
main(imgdir, annot_dir, vis_dir)
