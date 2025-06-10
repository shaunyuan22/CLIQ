import argparse
import os
import os.path as osp
import shutil
import tempfile
import json
import numpy as np
import warnings
import os, sys
sys.path.insert(0, os.getcwd())
import time
import pandas as pd
import mmcv
from mmcv import Config, DictAction
import torch
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet.datasets import build_dataloader, replace_ImageToTensor
from mmdet.models import build_detector
from mmdet.apis import multi_gpu_test, single_gpu_test

from mmrotate.datasets import build_dataset
from mmrotate.models import build_detector
from mmrotate.utils import compat_cfg, setup_multi_processes
from mmrotate.core import results2json, coco_eval, OBBDetComp4

from DOTA_devkit.ResultMerge_multi_process import py_cpu_nms_poly_fast
from utils_noc import read_gt, voc_eval, mergebase_parallel, mergebase_parallel_cell


classnames = None

def parse_args():
    parser = argparse.ArgumentParser(description='MMRotate test detector')
    parser.add_argument('config',
                                               help='test config file path')
    parser.add_argument('checkpoint', 
                                           help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        default='mAP',
        type=str,
        nargs='+',
        choices=['proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints'],
        help='eval types')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument('--log_dir', help='log the inference speed')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--exp', type=int)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    exp = args.exp

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg = compat_cfg(cfg)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1)

    if args.launcher == 'none':
        distributed = False
        if len(cfg.gpu_ids) > 1:
            warnings.warn(
                f'We treat {cfg.gpu_ids} as gpu-ids, and reset to '
                f'{cfg.gpu_ids[0:1]} as gpu-ids to avoid potential error in '
                'non-distribute testing time.')
            cfg.gpu_ids = cfg.gpu_ids[0:1]
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    if "UserInput" in cfg.data.test.type:
        cfg.data.test['noc'] = True

    test_dataloader_default_args = dict(
            samples_per_gpu=1, workers_per_gpu=2, dist=distributed, shuffle=False)
    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get('test_dataloader', {})
    }


    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        if 'samples_per_gpu' in cfg.data.test:
            warnings.warn('`samples_per_gpu` in `test` field of '
                          'data will be deprecated, you should'
                          ' move it to `test_dataloader` field')
            test_dataloader_default_args['samples_per_gpu'] = \
                cfg.data.test.pop('samples_per_gpu')
        if test_dataloader_default_args['samples_per_gpu'] > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
            if 'samples_per_gpu' in ds_cfg:
                warnings.warn('`samples_per_gpu` in `test` field of '
                              'data will be deprecated, you should'
                              ' move it to `test_dataloader` field')
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        test_dataloader_default_args['samples_per_gpu'] = samples_per_gpu
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # build the model and load checkpoint
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    data_path = osp.dirname(args.checkpoint)   
    if args.out is None:
        args.out = osp.join(data_path, 'out.pkl')

    # result_dic = {}
    result_list_AP = []
    result_list_es = []
    result_list_rs = []
    result_list_gs = []
    result_list_Normal = []


    init_point = 1
    max_points = 21
    for point in range(init_point, max_points):
        dataset.drawer.iter = point
        print(f'{point} iteration')
        if not distributed:
            model = MMDataParallel(model, device_ids=[0])
            outputs = single_gpu_test(model, data_loader, args.show, args.log_dir)
        else:
            model = MMDistributedDataParallel(model.cuda(), device_ids=[torch.cuda.current_device()])
            outputs = multi_gpu_test(model, data_loader, args.tmpdir)

        rank, _ = get_dist_info()


        if rank == 0:
            print('\nwriting results to {}'.format(args.out))
            # mmcv.dump(outputs, args.out)
            eval_types = args.eval
            if eval_types:
                print('Starting evaluate {}'.format(eval_types))
                if eval_types == ['proposal_fast']:
                    result_file = args.out
                    coco_eval(result_file, eval_types, dataset.coco)
                else:
                    if not isinstance(outputs[0], dict):
                        # result_file = '.'.join(args.out.split('.')[:-1]) + f'_{point}.json'
                        # results2json(dataset, outputs, result_file)

                        eval_kwargs=dict(metric=args.eval)
                        metric = dataset.evaluate(outputs, **eval_kwargs)
                        print(metric)



                        if cfg._cfg_dict['dataset_type'] == 'TINYDOTADataset_UserInput':
                            map= 0
                            map = metric['mAP_AP']
                            result_list_AP.append(map)
                            result_list_es.append(metric['mAP_AP_eS'])
                            result_list_rs.append(metric['mAP_AP_rS'])
                            result_list_gs.append(metric['mAP_AP_gS'])
                            result_list_Normal.append(metric['mAP_AP_Normal'])


                        
                    else:
                        for name in outputs[0]:
                            print('\nEvaluating {}'.format(name))
                            outputs_ = [out[name] for out in outputs]
                            result_file = args.out + '.{}.json'.format(name)
                            results2json(dataset, outputs_, result_file)
                            coco_eval(result_file, eval_types, dataset)
    if len(result_list_AP) > 0:
        name = ['mAP', 'mAP_eS', 'mAP_rS', 'mAP_gS', 'mAP_Normal']
        data = list(zip(result_list_AP, result_list_es, result_list_rs, 
                        result_list_gs, result_list_Normal))  # 将列表数据逐项组合

        test_x = pd.DataFrame(columns=name, data=data)
        test_x.to_csv(osp.join(data_path, 'testcsv.csv'), encoding='gbk', index=False)

    
if __name__ == '__main__':
    main()
