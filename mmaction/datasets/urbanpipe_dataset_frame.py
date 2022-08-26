from cProfile import label
import copy
import os.path as osp
import json

import torch
import mmcv

from .base import BaseDataset
from .builder import DATASETS


@DATASETS.register_module()
class UrbanPipeFrame(BaseDataset):

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_prefix=None,
                 test_mode=False,
                 num_classes=17,
                 filename_tmpl='img_{:05}.jpg'):
        data_prefix = '/home/chenbeitao/data/code/mmlab/Video-Swin-Transformer/data/urbanpipe/urbanpipe_data/media/sdd/zhangxuan/eccv_data_raw_video/'
        super(UrbanPipeFrame, self).__init__(
            ann_file, 
            pipeline, 
            data_prefix=osp.realpath(data_prefix), 
            test_mode=test_mode,
            multi_class=True,
            start_index=0,
            num_classes=num_classes
        )

        self.filename_tmpl = filename_tmpl

        from mmaction.datasets.pipelines.loading import MixDecordDecode
        from mmaction.datasets.pipelines.loading import MixTimeDecordDecode
        for i, t in enumerate(self.pipeline.transforms):
            if type(t) == MixDecordDecode or  type(t) == MixTimeDecordDecode:
                self.pipeline.transforms[i].datasets = self


    def load_annotations(self):
        video_infos = []
        file = json.load(open(self.ann_file, 'r'))
        flag = ('test_video_name' in file)

        if flag:
            file = file['test_video_name']
            video_num_file = '/mnt/hdd1/chenbeitao/data/datasets/UrbanPipe-Track/video_frame_num.json'
            # video_num_file = '/mnt/hdd1/chenbeitao/data/datasets/UrbanPipe-Track/video_frame_train_num.json'
            video_num_dict = json.load(open(video_num_file, 'r'))
            clip_len = 32
            for video_name in file:
                single_frame_num = video_num_dict[video_name]
                segment_num = single_frame_num // clip_len
                if segment_num % clip_len != 0:
                    segment_num += 1
                for segment_id in range(segment_num):
                    video_infos.append(
                        dict(
                            filename = osp.join(self.data_prefix, video_name),
                            frame_dir = video_name,
                            total_frames = video_num_dict[video_name],
                            segment_id = segment_id,
                            label = [0]
                        )
                    )
        # total_video_num = file['total_video_num']
        # video_name_list = file['test_video_name']
        else:
            for video_name in file:
                filename = video_name
                video_infos.append(
                    dict(
                        filename=osp.join(self.data_prefix, video_name),
                        frame_dir=video_name,
                        total_frames=-1,
                        label=[0] if flag else file[video_name]
                    )
                )
        return video_infos
 
    def prepare_test_frames(self, idx):
        """Prepare the frames for testing given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        results['modality'] = self.modality
        results['start_index'] = self.start_index

        # prepare tensor in getitem
        # If HVU, type(results['label']) is dict
        if self.multi_class and isinstance(results['label'], list):
            onehot = torch.zeros(self.num_classes)
            onehot[results['label']] = 1.
            results['label'] = onehot

        return self.pipeline(results)


    def evaluate(self,
                 results,
                 metrics='top_k_accuracy',
                 topk=(1, 5),
                 logger=None):
        pass

