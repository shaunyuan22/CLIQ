import itertools
from terminaltables import AsciiTable
from mmcv.utils import print_log
import torch
import glob
import mmcv
import numpy as np
import re
import copy
import os
import os.path as osp
from .builder import DATASETS
from .tinydota import TinyDOTADataset
from .tinydota_eval import TinyDOTAEval
from .drawer.userinput import UserinputDrawer
from mmcv.parallel import DataContainer as DC
from collections import Sequence
from mmrotate.core import poly2obb_np
from mmcv.ops import batched_nms
import multiprocessing
from functools import partial
from mmcv.ops.nms import nms_rotated

def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    """
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not mmcv.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError('type {} cannot be converted to tensor.'.format(
            type(data)))

@DATASETS.register_module()
class TinyDOTADatasetUserInput(TinyDOTADataset):
    def __init__(self, *args, **kwargs):
        ori_ann_file = kwargs.pop('ori_ann_file')
        noc = kwargs.pop('noc', False) 
        max_user_input = kwargs.pop('max_user_input')
        max_gt_num_perimg = kwargs.pop('max_gt_num_perimg')
        self.max_gt_num_perimg = max_gt_num_perimg
        super(TINYDOTADataset_UserInput, self).__init__(*args, **kwargs)
        self.drawer = UserinputDrawer(classes=self.CLASSES, noc=noc, max_user_input=max_user_input)
        if self.test_mode:
            
            self.test_in_ori_imgs = True
            self.ori_ann_file = ori_ann_file


    def prepare_train_img(self, idx):
        try:
            data = super().prepare_train_img(idx)
            data = self.drawer.prepare_user_input(data, idx)
        except:
            data = None
        return data

    def _prepare_test_img(self, idx):
        data = super().prepare_test_img(idx)
        assert len(data['img']) == 1, 'The input image should be 1.'

        ann = self.get_ann_info(idx)
        gt_bboxes = ann['bboxes']
        gt_labels = ann['labels']

        data['gt_labels'] = DC(to_tensor(gt_labels))
        data['gt_bboxes'] = DC(to_tensor(gt_bboxes))

        if self.drawer.noc:
            data = self.drawer.prepare_noc(data, idx)
        else:
            data = self.drawer.prepare_user_input(data, idx)

        data.pop('gt_labels')
        data.pop('gt_bboxes')
        return data

    def prepare_test_img(self, idx):
        # try:
            data = self._prepare_test_img(idx)
        # except:
            # print(idx)
            # data = None
            return data


    def _filter_imgs(self):
        """Filter images without ground truths."""
        valid_inds = []
        for i, data_info in enumerate(self.data_infos):
            if (not self.filter_empty_gt or data_info['ann']['labels'].size > 0):
                    if data_info['ann']['labels'].size < self.max_gt_num_perimg:
                        valid_inds.append(i)
        return valid_inds

    def load_ori_annotations(self, ann_folder, test_in_ori_imgs=True):
        cls_map = {c: i
                   for i, c in enumerate(self.CLASSES)
                   }  # in mmdet v2.0 label is 0-based
        ann_files = glob.glob(ann_folder + '/*.txt')
        data_infos = []
        if not ann_files:  # test phase
            ann_files = glob.glob(ann_folder + '/*.png')
            for ann_file in ann_files:
                data_info = {}
                img_id = osp.split(ann_file)[1][:-4]
                img_name = img_id + '.png'
                data_info['filename'] = img_name
                data_info['ann'] = {}
                data_info['ann']['bboxes'] = []
                data_info['ann']['labels'] = []
                data_infos.append(data_info)
        else:
            for ann_file in ann_files:
                data_info = {}
                img_id = osp.split(ann_file)[1][:-4]
                img_name = img_id + '.png'
                data_info['filename'] = img_name
                data_info['ann'] = {}
                gt_bboxes = []
                gt_labels = []
                gt_polygons = []
                gt_bboxes_ignore = []
                gt_labels_ignore = []
                gt_polygons_ignore = []

                if os.path.getsize(ann_file) == 0 and self.filter_empty_gt:
                    continue

                with open(ann_file) as f:
                    s = f.readlines()
                    line1 = s[0] 
                    line2 = s[1]
                    if 'imagesource' in line1 and 'gsd' in line2: 
                        s = s[2:]
                    for si in s:
                        if si.split()[-2] in self.CLASSES:
                            bbox_info = si.split()
                            poly = np.array(bbox_info[:8], dtype=np.float32)
                            try:
                                x, y, w, h, a = poly2obb_np(poly, self.version)
                            except:  # noqa: E722
                                continue
                            cls_name = bbox_info[8]
                            difficulty = int(bbox_info[9])
                            label = cls_map[cls_name]
                            if difficulty > self.difficulty:
                                pass
                            else:
                                gt_bboxes.append([x, y, w, h, a])
                                gt_labels.append(label)
                                gt_polygons.append(poly)

                if gt_bboxes:
                    data_info['ann']['bboxes'] = np.array(
                        gt_bboxes, dtype=np.float32)
                    data_info['ann']['labels'] = np.array(
                        gt_labels, dtype=np.int64)
                    data_info['ann']['polygons'] = np.array(
                        gt_polygons, dtype=np.float32)
                else:
                    data_info['ann']['bboxes'] = np.zeros((0, 5),
                                                          dtype=np.float32)
                    data_info['ann']['labels'] = np.array([], dtype=np.int64)
                    data_info['ann']['polygons'] = np.zeros((0, 8),
                                                            dtype=np.float32)

                if gt_polygons_ignore:
                    data_info['ann']['bboxes_ignore'] = np.array(
                        gt_bboxes_ignore, dtype=np.float32)
                    data_info['ann']['labels_ignore'] = np.array(
                        gt_labels_ignore, dtype=np.int64)
                    data_info['ann']['polygons_ignore'] = np.array(
                        gt_polygons_ignore, dtype=np.float32)
                else:
                    data_info['ann']['bboxes_ignore'] = np.zeros(
                        (0, 5), dtype=np.float32)
                    data_info['ann']['labels_ignore'] = np.array(
                        [], dtype=np.int64)
                    data_info['ann']['polygons_ignore'] = np.zeros(
                        (0, 8), dtype=np.float32)

                data_infos.append(data_info)

        return data_infos


    def merge_det_classes(self, results):
        nameboxdict = {}
        for i, splitline in enumerate(self.data_infos):
            # splitline = splitline.split(' ')
            subname = splitline['filename']
            splitname = subname.split('__')
            oriname = splitname[0]
            pattern1 = re.compile(r'__\d+___\d+')
            x_y = re.findall(pattern1, subname)
            x_y_2 = re.findall(r'\d+', x_y[0])
            x, y = int(x_y_2[0]), int(x_y_2[1])

            pattern2 = re.compile(r'__([\d+\.]+)__\d+___')
            rate = re.findall(pattern2, subname)[0]

            pre_bbox = results[i]
            # ori_img_bbox = self.translate(pre_bbox, x, y)

            if (oriname not in nameboxdict):
                nameboxdict[oriname] = [[] for c in range(len(self.CLASSES))]
            translated = copy.deepcopy(pre_bbox)
            for j, cls_bboxs in enumerate(pre_bbox):
                if cls_bboxs.size == 0:
                    continue
                translated[j][..., :2] = cls_bboxs[..., :2] + \
                                        np.array([x, y], dtype=np.float32)
                nameboxdict[oriname][j].append(translated[j])

        new_results = []
        for i in range(len(self.ori_data_infos)):
            oriname = self.ori_data_infos[i]['filename'].split('.')[0]
            new_result = []
            for i_c_det in nameboxdict[oriname]:
                if len(i_c_det) == 0:
                    continue
                cat_det = np.vstack(i_c_det) 
                new_result.append(cat_det)
            new_results.append(new_result)
        nms_new_results = self.nms_ori_img_det(new_results)
        return nms_new_results
    
    def nms_ori_img_det(self, new_results, iou_threshold=0.1, num_processes=4):
        nms_new_results = []
        for new_result in new_results:
            nms_new_results.append(self.nms2img_classes(new_result, iou_threshold=iou_threshold))
        # with multiprocessing.Pool(processes=4) as pool:
        #         nms2all_classes_fn = partial(self.nms2img_classes, 
        #                                      iou_threshold=iou_threshold)
        #         nms_new_results = pool.map(nms2all_classes_fn, new_results)

        return nms_new_results
    

    def nms2img_classes(self, det, iou_threshold):
        new_det = []
        for cls_idx, boxes in enumerate(det):
            if boxes.size == 0:
                continue

            bboxes = torch.from_numpy(boxes[:, :5]).to(torch.float32).contiguous().cuda()
            scores = torch.from_numpy(boxes[:, 5]).to(torch.float32).contiguous().cuda()
            results, _ = nms_rotated(bboxes, scores, iou_threshold)
        
            new_boxes = results.cpu().numpy()
            new_det.append(new_boxes)

        return new_det
    

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 metric_items=None,
                 scale_ranges=None,
                 nproc=8):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. It must be a float
                when evaluating mAP, and can be a list when evaluating recall.
                Default: 0.5.
            scale_ranges (list[tuple] | None): Scale ranges for evaluating mAP.
                Default: None.
            nproc (int): Processes used for computing TP and FP.
                Default: 4.
        """
        iou_thr = np.linspace(.5, 0.5, int(np.round((0.5 - .5) / .05)) + 1, endpoint=True)
        nproc = min(nproc, os.cpu_count())
        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
        
        if self.test_in_ori_imgs:
            
            self.ori_data_infos = self.load_ori_annotations(self.ori_ann_file, 
                                                        test_in_ori_imgs=True)
            annotations = [self.ori_data_infos[i]['ann']
                           for i in range(len(self.ori_data_infos))]
            results = self.merge_det_classes(results)
        else:
            annotations = [self.get_ann_info(i) for i in range(len(self))]
        


        eval_results = {}
        TDEval = TinyDOTAEval(annotations, results, numCats=len(self.CLASSES), nproc=nproc)
        TDEval.iouThrs = iou_thr

        # mapping of cocoEval.stats
        tinydota_metric_names = {
            'AP': 0,
            'AP_eS': 1,
            'AP_rS': 2,
            'AP_gS': 3,
            'AP_Normal': 4,
            'AR': 5,
            'AR_eS': 6,
            'AR_rS': 7,
            'AR_gS': 8,
            'AR_Normal': 9
        }
        TDEval.evaluate()
        TDEval.accumulate()
        TDEval.summarize()

        classwise = True
        if classwise:  # Compute per-category AP
            # Compute per-category AP
            # from https://github.com/facebookresearch/detectron2/
            precisions = TDEval.eval['precision']
            # precision: (iou, recall, cls, area range, max dets)
            assert len(self.cat_ids) == precisions.shape[2]

            results_per_category = []
            for catId, catName in self.cat_ids.items():
                # area range index 0: all area ranges
                # max dets index -1: typically 20000 per image
                precision = precisions[:, :, catId, 0, -1]
                precision = precision[precision > -1]
                if precision.size:
                    ap = np.mean(precision)
                else:
                    ap = float('nan')
                results_per_category.append(
                    (f'{catName}', f'{float(ap):0.5f}'))

            num_columns = min(6, len(results_per_category) * 2)
            results_flatten = list(
                itertools.chain(*results_per_category))
            headers = ['category', 'AP'] * (num_columns // 2)
            results_2d = itertools.zip_longest(*[
                results_flatten[i::num_columns]
                for i in range(num_columns)
            ])
            table_data = [headers]
            table_data += [result for result in results_2d]
            table = AsciiTable(table_data)
            print_log('\n' + table.table, logger=logger)

        # TODO: proposal evaluation
        if metric_items is None:
            metric_items = [
                'AP', 'AP_eS',
                'AP_rS', 'AP_gS', 'AP_Normal'
            ]

        for metric_item in metric_items:
            key = f'{metric}_{metric_item}'
            val = float(
                f'{TDEval.stats[tinydota_metric_names[metric_item]]:.5f}'
            )
            eval_results[key] = val
        ap = TDEval.stats[:5]
        eval_results[f'{metric}_mAP_copypaste'] = (
            f'{ap[0]:.5f} {ap[1]:.5f} {ap[2]:.5f} '
            f'{ap[3]:.5f} {ap[4]:.5f}'
        )

        return eval_results