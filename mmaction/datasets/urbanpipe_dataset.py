import copy
import os.path as osp
import json

import mmcv

from .base import BaseDataset
from .builder import DATASETS


@DATASETS.register_module()
class UrbanPipe(BaseDataset):

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_prefix=None,
                 test_mode=False,
                 num_classes=17,
                 filename_tmpl='img_{:05}.jpg'):
        data_prefix = '/home/chenbeitao/data/code/mmlab/Video-Swin-Transformer/data/urbanpipe/urbanpipe_data/media/sdd/zhangxuan/eccv_data_raw_video/'
        super(UrbanPipe, self).__init__(
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
        for t in self.transforms:
            if type(t) == MixDecordDecode:
                t.datasets = self
                # results = copy.deepcopy(self.video_infos[idx])
                # results['modality'] = self.modality
                # results['start_index'] = self.start_index

                # # prepare tensor in getitem
                # # If HVU, type(results['label']) is dict
                # if self.multi_class and isinstance(results['label'], list):
                #     onehot = torch.zeros(self.num_classes)
                #     onehot[results['label']] = 1.
                #     results['label'] = onehot

                # return self.pipeline(results)

    def load_annotations(self):
        video_infos = []
        file = json.load(open(self.ann_file, 'r'))
        flag = ('test_video_name' in file)
        if flag:
            file = file['test_video_name']
        # total_video_num = file['total_video_num']
        # video_name_list = file['test_video_name']
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
        # with open(self.ann_file, 'r') as fin:
        #     for line in fin:
        #         if line.startswith("directory"):
        #             continue
        #         frame_dir, total_frames, label = line.split(',')
        #         if self.data_prefix is not None:
        #             frame_dir = osp.join(self.data_prefix, frame_dir)
        #         video_infos.append(
        #             dict(
        #                 frame_dir=frame_dir,
        #                 total_frames=int(total_frames),
        #                 label=int(label)))
        return video_infos

    # def prepare_train_frames(self, idx):
    #     results = copy.deepcopy(self.video_infos[idx])
    #     results['filename_tmpl'] = self.filename_tmpl
    #     return self.pipeline(results)

    # def prepare_test_frames(self, idx):
    #     results = copy.deepcopy(self.video_infos[idx])
    #     results['filename_tmpl'] = self.filename_tmpl
    #     return self.pipeline(results)

    def evaluate(self,
                 results,
                 metrics='top_k_accuracy',
                 topk=(1, 5),
                 logger=None):
        pass