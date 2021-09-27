# --------------------------------------------------------
# 在线评估
# --------------------------------------------------------

"""
    To use the code, users should to config detpath, annopath and imagesetfile
    detpath is the path for 15 result files, for the format, you can refer to "http://captain.whu.edu.cn/DOTAweb/tasks.html"
    search for PATH_TO_BE_CONFIGURED to config the paths
    Note, the evaluation is on the large scale images
"""
import csv
import xml.etree.ElementTree as ET
import os
# import cPickle
import numpy as np
import matplotlib.pyplot as plt
from utils import polyiou
from functools import partial
import pdb
from utils.evaluation_utils import mergebypoly, evaluation_trans, image2txt, draw_DOTA_image
from utils.general import colorstr, plotChart, plotChart_interest


def parse_gt(filename):
    """

    :param filename: ground truth file to parse
    :return: all instances in a picture
    """
    objects = []
    with  open(filename, 'r') as f:
        while True:
            line = f.readline()
            if line:
                splitlines = line.strip().split(' ')
                object_struct = {}
                if len(splitlines) < 9:
                    continue
                object_struct['name'] = splitlines[8]

                # if (len(splitlines) == 9):
                #     object_struct['difficult'] = 0
                # elif (len(splitlines) == 10):
                #     object_struct['difficult'] = int(splitlines[9])
                object_struct['difficult'] = 0
                object_struct['bbox'] = [float(splitlines[0]),
                                         float(splitlines[1]),
                                         float(splitlines[2]),
                                         float(splitlines[3]),
                                         float(splitlines[4]),
                                         float(splitlines[5]),
                                         float(splitlines[6]),
                                         float(splitlines[7])]
                objects.append(object_struct)
            else:
                break
    return objects


def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(detpath,
             annopath,  # DOTA_demo_view/row_DOTA_labels/
             imagesetfile,  # DOTA_demo_view/detection/result_txt/imgnamefile.txt
             classname,
             # cachedir,
             ovthresh=0.5,
             use_07_metric=False):
    """
    Top level function that does the PASCAL VOC evaluation.
    @param detpath: Path to detections
           detpath.format(classname) should produce the detection results file.
    @param annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    @param imagesetfile: Text file containing the list of images, one image per line.
    @param classname: Category name (duh)
    cachedir: Directory for caching the annotations
    @param [ovthresh]: Overlap threshold (default = 0.5)
    @param [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    @return: rec, prec, ap
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # first load gt
    # if not os.path.isdir(cachedir):
    #   os.mkdir(cachedir)
    # cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]
    # print('imagenames: ', imagenames)
    # if not os.path.isfile(cachefile):
    # load annots
    recs = {}
    for i, imagename in enumerate(imagenames):
        # print('parse_files name: ', annopath.format(imagename))
        recs[imagename] = parse_gt(annopath.format(imagename))  # parse row labels

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets from Task1* files
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])

    # print('check confidence: ', confidence)

    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)

    # print('check sorted_scores: ', sorted_scores)
    # print('check sorted_ind: ', sorted_ind)

    ## note the usage only in numpy not for list
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]
    # print('check imge_ids: ', image_ids)
    # print('imge_ids len:', len(image_ids))
    # go down dets and mark TPs and FPs
    iouv = np.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
    nd = len(image_ids)
    tp = np.zeros((10, nd))
    fp = np.zeros((10, nd))
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        ## compute det bb with each BBGT

        if BBGT.size > 0:
            # compute overlaps
            # intersection

            # 1. calculate the overlaps between hbbs, if the iou between hbbs are 0, the iou between obbs are 0, too.
            # pdb.set_trace()
            BBGT_xmin = np.min(BBGT[:, 0::2], axis=1)
            BBGT_ymin = np.min(BBGT[:, 1::2], axis=1)
            BBGT_xmax = np.max(BBGT[:, 0::2], axis=1)
            BBGT_ymax = np.max(BBGT[:, 1::2], axis=1)
            bb_xmin = np.min(bb[0::2])
            bb_ymin = np.min(bb[1::2])
            bb_xmax = np.max(bb[0::2])
            bb_ymax = np.max(bb[1::2])

            ixmin = np.maximum(BBGT_xmin, bb_xmin)
            iymin = np.maximum(BBGT_ymin, bb_ymin)
            ixmax = np.minimum(BBGT_xmax, bb_xmax)
            iymax = np.minimum(BBGT_ymax, bb_ymax)
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb_xmax - bb_xmin + 1.) * (bb_ymax - bb_ymin + 1.) +
                   (BBGT_xmax - BBGT_xmin + 1.) *
                   (BBGT_ymax - BBGT_ymin + 1.) - inters)

            overlaps = inters / uni

            BBGT_keep_mask = overlaps > 0
            BBGT_keep = BBGT[BBGT_keep_mask, :]
            BBGT_keep_index = np.where(overlaps > 0)[0]

            # pdb.set_trace()
            def calcoverlaps(BBGT_keep, bb):
                overlaps = []
                for index, GT in enumerate(BBGT_keep):
                    overlap = polyiou.iou_poly(polyiou.VectorDouble(BBGT_keep[index]), polyiou.VectorDouble(bb))
                    overlaps.append(overlap)
                return overlaps

            if len(BBGT_keep) > 0:
                overlaps = calcoverlaps(BBGT_keep, bb)

                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)
                # pdb.set_trace()
                jmax = BBGT_keep_index[jmax]
        flag = 0  # R['det'][jmax]
        for i, ov in enumerate(iouv):
            if ovmax > ov:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[i][d] = 1.
                        flag = 1
                    else:
                        fp[i][d] = 1.
            else:
                fp[i][d] = 1.
        if flag == 1:
            R['det'][jmax] = 1

    # compute precision recall

    # print('check fp:', fp)
    # print('check tp', tp)

    # print('npos num:', npos)
    fp = np.cumsum(fp, axis=1)
    tp = np.cumsum(tp, axis=1)

    rec = np.zeros(tp.shape)
    rec[:, 0] = tp[:, 0] / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = np.zeros(tp.shape)
    # eps是很小的非负数，即0和负数全变为eps
    prec[:, 0] = tp[:, 0] / np.maximum(tp[:, 0] + fp[:, 0], np.finfo(np.float64).eps)
    ap = np.zeros(10)
    for i, (rec, prec) in enumerate(rec, prec):
        ap[i] = voc_ap(rec, prec, use_07_metric)

    print(ap)
    return rec, prec, ap


def val(detectionPath, rawImagePath, rawLabelPath, resultPath):
    """
    评估程序
    @param detectionPath: detect.py函数的结果存放输出路径
    @param rawImagePath: # val DOTA原图数据集图像路径
    @param rawLabelPath: 'r/.../{:s}.txt'  原始val测试集的DOTAlabels路径
    @param resultPath: AP结果输出路径 (Path)
    # @param classnames: 测试集中的目标种类
    """
    # For DOTA-v1.5
    # classnames = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle',
    #               'large-vehicle', 'ship', 'tennis-court', 'basketball-court', 'storage-tank',
    #               'soccer-ball-field', 'roundabout', 'harbor',
    #               'swimming-pool', 'helicopter', 'container-crane']
    # For interest
    classnames = ['small-vehicle', 'ship']

    result_before_merge_path = str(detectionPath + '/result_txt/result_before_merge')
    result_merged_path = str(detectionPath + '/result_txt/result_merged')
    result_classname_path = str(detectionPath + '/result_txt/result_classname')
    imageset_name_file_path = str(detectionPath + '/result_txt')

    # merge检测结果
    # 将源路径中所有的txt目标信息,经nms后存入目标路径中的同名txt
    mergebypoly(
        result_before_merge_path,
        result_merged_path
    )

    # 按照类别分类检测结果
    # 将result_merged_path文件夹中的所有txt中的目标提取出来,按照目标类别分别存入 Task1_类别名.txt中
    evaluation_trans(
        result_merged_path,
        result_classname_path
    )

    # 生成校验数据集名称文件
    # 将srcpath文件夹下的所有子文件名称打印到namefile.txt中
    image2txt(
        rawImagePath,  # val原图数据集路径
        imageset_name_file_path
    )

    detpath = str(result_classname_path + '/Task1_{:s}.txt')  # 'r/.../Task1_{:s}.txt'  存放各类别结果文件txt的路径
    rawLabelPath = rawLabelPath
    imagesetfile = str(imageset_name_file_path + '/imgnamefile.txt')  # 'r/.../imgnamefile.txt'  测试集图片名称txt

    # detpath = r'PATH_TO_BE_CONFIGURED/Task1_{:s}.txt'
    # annopath = r'PATH_TO_BE_CONFIGURED/{:s}.txt' # change the directory to the path of val/labelTxt, if you want to do evaluation on the valset
    # imagesetfile = r'PATH_TO_BE_CONFIGURED/valset.txt'

    # classaps = []
    epoch = len(open(resultPath / 'classAP.csv', 'r').readlines())
    allClassAp = [epoch]
    all_class_ap50 = 0
    all_class_ap = 0
    skippedClassCount = 0
    print(('%20s' * 2) % ('Class', 'AP'))
    for classname in classnames:
        detfile = detpath.format(classname)
        if not (os.path.exists(detfile)):  # 如果这个类别不存在
            skippedClassCount += 1
            allClassAp.append(0.0)
            continue

        # 得到[ap50, ap55, ap60, ...]
        rec, prec, ap = voc_eval(detpath,
                                 rawLabelPath,
                                 imagesetfile,
                                 classname,
                                 ovthresh=0.5,
                                 use_07_metric=True)
        ap50, ap = ap[0], ap.mean()
        # 累加所有ap
        all_class_ap50 += ap50
        all_class_ap += ap

        # print('rec: ', rec, 'prec: ', prec, 'ap: ', ap)
        print('%20s%20g' % (classname, ap[0]))
        allClassAp.append(ap[0])
        # classaps.append(ap)

        # umcomment to show p-r curve of each category
        # plt.figure(figsize=(8,4))
        # plt.xlabel('recall')
        # plt.ylabel('precision')
        # plt.plot(rec, prec)
    # plt.show()
    nc = len(classnames) - skippedClassCount
    if skippedClassCount != len(classnames):
        map50 = all_class_ap50 / nc
        map = all_class_ap / nc
    else:
        map50 = 0
        map = 0
    print(colorstr('          mAP@0.5:'), map50)
    print(colorstr('     mAP@0.5:0.95:'), map)
    allClassAp.insert(1, map50)

    with open(resultPath / 'classAP.csv', 'a', encoding='utf-8', newline="") as f_ap:
        csv_ap = csv.writer(f_ap)
        csv_ap.writerow(allClassAp)

    plotChart_interest("classAP", str(resultPath))
    # classaps = 100 * np.array(classaps)
    # print('classaps: ', classaps)


if __name__ == '__main__':
    '''
    计算AP的流程:
    1.detect.py检测一个文件夹的所有图片并把检测结果按照图片原始来源存入 原始图片名称.txt中:   (rbox2txt函数)
        txt中的内容格式:  目标所属图片名称_分割id 置信度 poly classname
    2.ResultMerge.py将所有 原始图片名称.txt 进行merge和nms,并将结果存入到另一个文件夹的 原始图片名称.txt中:
        txt中的内容格式:  目标所属图片名称 置信度 poly classname
    3.写一个evaluation_trans.py将上个文件夹中的所有txt中的目标提取出来,按照目标类别分别存入 Task1_类别名.txt中:
        txt中的内容格式:  目标所属原始图片名称 置信度 poly
    4.写一个imgname2txt.py 将测试集的所有图片名称打印到namefile.txt中
    '''

    val(
        detectionPath='/home/test/Persons/hukaixuan/yolov5_DOTA_OBB/DOTA_demo_view/detection',
        rawImagePath=r'/home/test/Persons/hukaixuan/yolov5_DOTA_OBB/DOTA_demo_view/row_images',
        rawLabelPath=r'/home/test/Persons/hukaixuan/yolov5_DOTA_OBB/DOTA_demo_view/row_DOTA_labels/{:s}.txt'
    )
