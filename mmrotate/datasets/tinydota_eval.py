__author__ = 'tsungyi'

import copy
import datetime
import time
from collections import defaultdict

from multiprocessing import Pool
from functools import partial

import numpy as np
import torch

from mmcv.ops import box_iou_rotated
# import numba

class TinyDOTAEval:
    def __init__(self, annotations=None, results=None, numCats=9, iouType='mAP', nproc=10):
        '''
        Initialize CocoEval using coco APIs for gt and dt
        :param sodaaGt: coco object with ground truth annotations
        :param sodaaDt: coco object with detection results
        :return: None
        '''
        if not iouType:
            print('iouType not specified. use default iouType: mAP')
        self.annotations = annotations  # ground truth
        self.results = results  # detections
        self.numCats = numCats
        self.nproc = nproc
        self.evalImgs = defaultdict(
            list)  # per-image per-category evaluation results [KxAxI] elements
        self.eval = {}  # accumulated evaluation results
        self._gts = defaultdict(list)  # gt for evaluation
        self._dts = defaultdict(list)  # dt for evaluation
        self.params = SODAAParams(iouType=iouType)  # parameters
        self._paramsEval = {}  # parameters for evaluation
        self.stats = []  # result summarization
        self.ious = {}  # ious between all gts and dts
        # TODO: get ids
        if self.annotations is not None:
            self._getImgAndCatIds()

    def _getImgAndCatIds(self):
        self.params.imgIds = [i for i, _ in enumerate(self.annotations)]
        self.params.catIds = [i for i in range(self.numCats)]

    def _prepare(self):
        '''
        Prepare ._gts and ._dts for evaluation based on params
        :return: None
        '''
        p = self.params
        if p.useCats:
            # TODO: we do not specify the area sue to no-area split so far.
            gts = list()
            insId = 0
            for i, imgAnn in enumerate(self.annotations):
                for j in range(len(imgAnn['labels'])):
                    gt = dict(
                        bbox = imgAnn['bboxes'][j],
                        area = imgAnn['bboxes'][j][2] * imgAnn['bboxes'][j][3],
                        category_id = imgAnn['labels'][j],
                        image_id = i,
                        id = insId,
                        ignore = 0  # no ignore
                    )
                    gts.append(gt)
                    insId += 1

            dts = list()
            insId = 0
            for i, imgRes in enumerate(self.results):
                for j, catRes in enumerate(imgRes):
                    if len(catRes) == 0:
                        continue
                    bboxes, scores = catRes[:, :5], catRes[:, -1]
                    for k in range(len(scores)):
                        dt = dict(
                            image_id = i,
                            bbox = bboxes[k],
                            score = scores[k],
                            category_id = j,
                            id = insId,
                            area = bboxes[k][2] * bboxes[k][3]
                        )
                        dts.append(dt)
                        insId += 1
        else:
            # TODO: add class-agnostic evaluation codes
            pass

        self._gts = defaultdict(list)  # gt for evaluation
        self._dts = defaultdict(list)  # dt for evaluation
        for gt in gts:
            self._gts[gt['image_id'], gt['category_id']].append(gt)
        for dt in dts:
            self._dts[dt['image_id'], dt['category_id']].append(dt)
        self.evalImgs = defaultdict(
            list)  # per-image per-category evaluation results
        self.eval = {}  # accumulated evaluation results

    def evaluate(self):
        '''
        Run per image evaluation on given images and store results
         (a list of dict) in self.evalImgs
        :return: None
        '''
        # tic = time.time()
        p = self.params
        print('Evaluate annotation type *{}*'.format(p.iouType))
        p.imgIds = list(np.unique(p.imgIds))
        if p.useCats:
            p.catIds = list(np.unique(p.catIds))
        p.maxDets = sorted(p.maxDets)
        self.params = p

        self._prepare()
        # loop through images, area range, max detection number
        catIds = p.catIds if p.useCats else [-1]

        if p.iouType == 'mAP':
            computeIoU = self.computeIoU
        else:
            raise Exception('unknown iouType for iou computation')
        print('Calculating IoUs...')
        tic = time.time()

        self.ious = {(imgId, catId): computeIoU(imgId, catId)
                     for imgId in p.imgIds for catId in catIds}
        toc = time.time()
        print('IoU calculation Done (t={:0.2f}s).'.format(toc - tic))

        print('Running per image evaluation...')
        tic = time.time()
        evaluateImg = self.evaluateImgPartial if self.nproc else self.evaluateImg
        maxDet = p.maxDets[-1]
        evaluateImgFunc = partial(evaluateImg)
        inteLst = [[imgId, catId, areaRng, maxDet] for catId in catIds for areaRng in p.areaRng for imgId in p.imgIds]
        imgIdLst, catIdLst, areaRngLst, maxDetLst = [], [], [], []
        for lst in inteLst:
            imgIdLst.append(lst[0])
            catIdLst.append(lst[1])
            areaRngLst.append(lst[2])
            maxDetLst.append(lst[3])

        if self.nproc > 1:
            pool = Pool(self.nproc)
            contents = pool.map(evaluateImgFunc,
                                   zip(imgIdLst, catIdLst, areaRngLst, maxDetLst))
            pool.close()
        else:
            contents = [evaluateImg(imgId, catId, areaRng, maxDet) for catId in catIds
                        for areaRng in p.areaRng for imgId in p.imgIds]
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc - tic))
        self.evalImgs = [c for c in contents]
        self._paramsEval = copy.deepcopy(self.params)

    def computeIoUPartial(self, args):
        imgId, catId = args
        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
        if len(gt) == 0 and len(dt) == 0:
            return []
        # sort dt highest score first
        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt = dt[0:p.maxDets[-1]]

        if p.iouType == 'mAP':
            g = [g['bbox'] for g in gt]
            d = [d['bbox'] for d in dt]
        else:
            raise Exception('unknown iouType for iou computation')

        # compute iou between each dt and gt region (rotated rectangle)
        ious = box_iou_rotated(
            torch.from_numpy(np.array(d)).float(),
            torch.from_numpy(np.array(g)).float()).numpy()
        return ious

    def computeIoU(self, imgId, catId):
        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
        if len(gt) == 0 and len(dt) == 0:
            return []
        # sort dt highest score first
        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt = dt[0:p.maxDets[-1]]

        if p.iouType == 'mAP':
            g = [g['bbox'] for g in gt]
            d = [d['bbox'] for d in dt]
        else:
            raise Exception('unknown iouType for iou computation')

        # compute iou between each dt and gt region (rotated rectangle)
        if len(gt)==0:
            ious = torch.empty(len(dt), 0, dtype=torch.float32)
        else:
            d = torch.from_numpy(np.array(d)).float()
            g = torch.from_numpy(np.array(g)).float()
            ious = box_iou_rotated(d, g)
        ious = ious.numpy()
        return ious

    
    def evaluateImgPartial(self, args):
        '''
        perform evaluation for single category and image
        :return: dict (single image results)
        '''
        imgId, catId, aRng, maxDet = args
        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
        if len(gt) == 0 and len(dt) == 0:
            return None

        # TODO: all to 0
        for g in gt:
            if g['ignore'] or (g['area'] < aRng[0] or g['area'] > aRng[1]):
                g['_ignore'] = 1
            else:
                g['_ignore'] = 0

        # sort dt highest score first, sort gt ignore last
        gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in dtind[0:maxDet]]
        iscrowd = [0 for o in gt]   # [int(o['iscrowd']) for o in gt]
        # load computed ious
        ious = self.ious[imgId, catId][:, gtind] if len(
            self.ious[imgId, catId]) > 0 else self.ious[imgId, catId]

        T = len(p.iouThrs)
        G = len(gt)
        D = len(dt)
        gtm = np.zeros((T, G))
        dtm = np.zeros((T, D))
        gtIg = np.array([g['_ignore'] for g in gt])
        dtIg = np.zeros((T, D))
        if not len(ious) == 0:
            for tind, t in enumerate(p.iouThrs):
                for dind, d in enumerate(dt):
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t, 1 - 1e-10])
                    m = -1
                    for gind, g in enumerate(gt):
                        # if this gt already matched, and not a crowd, continue
                        if gtm[tind, gind] > 0 and not iscrowd[gind]:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        if m > -1 and gtIg[m] == 0 and gtIg[gind] == 1:
                            break
                        # continue to next gt unless better match made
                        if ious[dind, gind] < iou:
                            continue
                        # if match successful and best so far, store
                        # appropriately
                        iou = ious[dind, gind]
                        m = gind
                    # if match made store id of match for both dt and gt
                    if m == -1:
                        continue
                    dtIg[tind, dind] = gtIg[m]
                    dtm[tind, dind] = gt[m]['id']
                    gtm[tind, m] = d['id']
        # set unmatched detections outside of area range to ignore
        a = np.array([d['area'] < aRng[0] or d['area'] > aRng[1]
                      for d in dt]).reshape((1, len(dt)))
        dtIg = np.logical_or(dtIg, np.logical_and(dtm == 0, np.repeat(a, T,
                                                                      0)))
        # store results for given image and category
        return {
            'image_id': imgId,
            'category_id': catId,
            'aRng': aRng,
            'maxDet': maxDet,
            'dtIds': [d['id'] for d in dt],
            'gtIds': [g['id'] for g in gt],
            'dtMatches': dtm,
            'gtMatches': gtm,
            'dtScores': [d['score'] for d in dt],
            'gtIgnore': gtIg,
            'dtIgnore': dtIg,
        }

    def evaluateImg(self, imgId, catId, aRng, maxDet):
        '''
        perform evaluation for single category and image
        :return: dict (single image results)
        '''
        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
        if len(gt) == 0 and len(dt) == 0:
            return None

        # TODO: all to 0
        for g in gt:
            if g['ignore'] or (g['area'] < aRng[0] or g['area'] > aRng[1]):
                g['_ignore'] = 1
            else:
                g['_ignore'] = 0

        # sort dt highest score first, sort gt ignore last
        gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in dtind[0:maxDet]]
        iscrowd = [0 for o in gt]   # [int(o['iscrowd']) for o in gt]
        # load computed ious
        ious = self.ious[imgId, catId][:, gtind] if len(
            self.ious[imgId, catId]) > 0 else self.ious[imgId, catId]

        T = len(p.iouThrs)
        G = len(gt)
        D = len(dt)
        gtm = np.zeros((T, G))
        dtm = np.zeros((T, D))
        gtIg = np.array([g['_ignore'] for g in gt])
        dtIg = np.zeros((T, D))
        if not len(ious) == 0:
            for tind, t in enumerate(p.iouThrs):
                for dind, d in enumerate(dt):
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t, 1 - 1e-10])
                    m = -1
                    for gind, g in enumerate(gt):
                        # if this gt already matched, and not a crowd, continue
                        if gtm[tind, gind] > 0 and not iscrowd[gind]:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        if m > -1 and gtIg[m] == 0 and gtIg[gind] == 1:
                            break
                        # continue to next gt unless better match made
                        if ious[dind, gind] < iou:
                            continue
                        # if match successful and best so far, store
                        # appropriately
                        iou = ious[dind, gind]
                        m = gind
                    # if match made store id of match for both dt and gt
                    if m == -1:
                        continue
                    dtIg[tind, dind] = gtIg[m]
                    dtm[tind, dind] = gt[m]['id']
                    gtm[tind, m] = d['id']
        # set unmatched detections outside of area range to ignore
        a = np.array([d['area'] < aRng[0] or d['area'] > aRng[1]
                      for d in dt]).reshape((1, len(dt)))
        dtIg = np.logical_or(dtIg, np.logical_and(dtm == 0, np.repeat(a, T,
                                                                      0)))

        # store results for given image and category
        return {
            'image_id': imgId,
            'category_id': catId,
            'aRng': aRng,
            'maxDet': maxDet,
            'dtIds': [d['id'] for d in dt],
            'gtIds': [g['id'] for g in gt],
            'dtMatches': dtm,
            'gtMatches': gtm,
            'dtScores': [d['score'] for d in dt],
            'gtIgnore': gtIg,
            'dtIgnore': dtIg,
        }

    def accumulate(self, p=None):
        '''
        Accumulate per image evaluation results and store the result in
        self.eval

        :param p: input params for evaluation
        :return: None
        '''
        print('Accumulating evaluation results...')
        tic = time.time()
        if not self.evalImgs:
            print('Please run evaluate() first')
        # allows input customized parameters
        if p is None:
            p = self.params
        p.catIds = p.catIds if p.useCats == 1 else [-1]
        T = len(p.iouThrs)
        R = len(p.recThrs)
        K = len(p.catIds) if p.useCats else 1
        A = len(p.areaRng)
        M = len(p.maxDets)
        precision = -np.ones(
            (T, R, K, A, M))  # -1 for the precision of absent categories
        recall = -np.ones((T, K, A, M))
        scores = -np.ones((T, R, K, A, M))

        # create dictionary for future indexing
        _pe = self._paramsEval
        catIds = _pe.catIds if _pe.useCats else [-1]
        setK = set(catIds)
        setA = set(map(tuple, _pe.areaRng))
        setM = set(_pe.maxDets)
        setI = set(_pe.imgIds)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(p.catIds) if k in setK]
        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        a_list = [
            n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng))
            if a in setA
        ]
        i_list = [n for n, i in enumerate(p.imgIds) if i in setI]
        I0 = len(_pe.imgIds)
        A0 = len(_pe.areaRng)
        # retrieve E at each category, area range, and max number of detections
        for k, k0 in enumerate(k_list):
            Nk = k0 * A0 * I0
            for a, a0 in enumerate(a_list):
                Na = a0 * I0
                for m, maxDet in enumerate(m_list):
                    E = [self.evalImgs[Nk + Na + i] for i in i_list]
                    E = [e for e in E if e is not None]
                    if len(E) == 0:
                        continue
                    dtScores = np.concatenate(
                        [e['dtScores'][0:maxDet] for e in E])

                    # different sorting method generates slightly different
                    # results. mergesort is used to be consistent as Matlab
                    # implementation.
                    inds = np.argsort(-dtScores, kind='mergesort')
                    dtScoresSorted = dtScores[inds]

                    dtm = np.concatenate(
                        [e['dtMatches'][:, 0:maxDet] for e in E], axis=1)[:,
                                                                          inds]
                    dtIg = np.concatenate(
                        [e['dtIgnore'][:, 0:maxDet] for e in E], axis=1)[:,
                                                                         inds]
                    gtIg = np.concatenate([e['gtIgnore'] for e in E])
                    npig = np.count_nonzero(gtIg == 0)
                    if npig == 0:
                        continue
                    tps = np.logical_and(dtm, np.logical_not(dtIg))
                    fps = np.logical_and(np.logical_not(dtm),
                                         np.logical_not(dtIg))

                    tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float64)
                    fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float64)
                    for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                        tp = np.array(tp)
                        fp = np.array(fp)
                        nd = len(tp)
                        rc = tp / npig
                        pr = tp / (fp + tp + np.spacing(1))
                        q = np.zeros((R, ))
                        ss = np.zeros((R, ))

                        if nd:
                            recall[t, k, a, m] = rc[-1]
                        else:
                            recall[t, k, a, m] = 0

                        # numpy is slow without cython optimization for
                        # accessing elements use python array gets significant
                        # speed improvement
                        pr = pr.tolist()
                        q = q.tolist()

                        for i in range(nd - 1, 0, -1):
                            if pr[i] > pr[i - 1]:
                                pr[i - 1] = pr[i]

                        inds = np.searchsorted(rc, p.recThrs, side='left')
                        try:
                            for ri, pi in enumerate(inds):
                                q[ri] = pr[pi]
                                ss[ri] = dtScoresSorted[pi]
                        except:  # noqa: E722
                            pass
                        precision[t, :, k, a, m] = np.array(q)
                        scores[t, :, k, a, m] = np.array(ss)
        self.eval = {
            'params': p,
            'counts': [T, R, K, A, M],
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'precision': precision,
            'recall': recall,
            'scores': scores,
        }
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc - tic))

    def summarize(self):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter
        setting
        '''
        def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100):
            p = self.params
            iStr = '{:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'  # noqa: E501
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap == 1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [
                i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng
            ]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, :, aind, mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, aind, mind]
            if len(s[s > -1]) == 0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s > -1])
            print(
                iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets,
                            mean_s))
            return mean_s

        def _summarizeDets():
            stats = np.zeros((10, ))
            # AP metric

            stats[0] = _summarize(1,
                                  iouThr=.50,
                                  areaRng='Small',
                                  maxDets=self.params.maxDets[0])
            stats[1] = _summarize(1,
                                  iouThr=.50,
                                  areaRng='eS',
                                  maxDets=self.params.maxDets[0])
            stats[2] = _summarize(1,
                                  iouThr=.50,
                                  areaRng='rS',
                                  maxDets=self.params.maxDets[0])
            stats[3] = _summarize(1,
                                  iouThr=.50,
                                  areaRng='gS',
                                  maxDets=self.params.maxDets[0])
            stats[4] = _summarize(1,
                                  iouThr=.50,
                                  areaRng='Normal',
                                  maxDets=self.params.maxDets[0])

            # AR metric
            stats[5] = _summarize(0,
                                   areaRng='Small',
                                   maxDets=self.params.maxDets[0])
            stats[6] = _summarize(0,
                                   areaRng='eS',
                                   maxDets=self.params.maxDets[0])
            stats[7] = _summarize(0,
                                   areaRng='rS',
                                   maxDets=self.params.maxDets[0])
            stats[8] = _summarize(0,
                                   areaRng='gS',
                                   maxDets=self.params.maxDets[0])
            stats[9] = _summarize(0,
                                   areaRng='Normal',
                                   maxDets=self.params.maxDets[0])
            return stats


        if not self.eval:
            raise Exception('Please run accumulate() first')
        iouType = self.params.iouType
        if iouType == 'mAP':
            summarize = _summarizeDets
        else:
            raise Exception('unknown iouType for iou computation')
        self.stats = summarize()

    def __str__(self):
        self.summarize()


class SODAAParams:
    '''
    Params for coco evaluation api
    '''
    def __init__(self, iouType='mAP'):
        if iouType == 'mAP':
            self.setDetParams()
        else:
            raise Exception('iouType not supported')
        self.iouType = iouType

    def setDetParams(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly
        # larger than the true value
        self.iouThrs = np.linspace(.5,
                                   0.95,
                                   int(np.round((0.95 - .5) / .05)) + 1,
                                   endpoint=True)
        self.recThrs = np.linspace(.0,
                                   1.00,
                                   int(np.round((1.00 - .0) / .01)) + 1,
                                   endpoint=True)
        # TODO: ensure
        self.maxDets = [20000]
        # self.areaRng = [[0 ** 2, 32 ** 2], [0 ** 2, 12 ** 2], [12 ** 2, 20 ** 2],
        #                 [20 ** 2, 32 ** 2], [32 ** 2, 40 * 50],[0 ** 2, 10000 ** 2]]
        # self.areaRngLbl = ['Small', 'eS', 'rS', 'gS', 'Normal', 'All']
        self.areaRng = [[0 ** 2, 32 ** 2], [0 ** 2, 12 ** 2], [12 ** 2, 20 ** 2],
                        [20 ** 2, 32 ** 2], [32 ** 2, 40 * 50]]
        self.areaRngLbl = ['Small', 'eS', 'rS', 'gS', 'Normal']      
        self.useCats = 1