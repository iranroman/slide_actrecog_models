import os
import torch
import torch.utils.data
import cv2
import numpy as np

from .build import DATASET_REGISTRY
import os
#from .epickitchens_record import EpicKitchensVideoRecord, timestamp_to_sec

#from . import transform as transform
#from . import utils as utils
#from .frame_loader import pack_frames_to_video_clip, pack_flow_frames_to_video_clip

@DATASET_REGISTRY.register()
class Milly(torch.utils.data.Dataset):

    def __init__(self, cfg, mode):

        assert mode in [
            "slide"
        ], "Split '{}' not supported for EPIC-KITCHENS".format(mode)
        self.data_dir = cfg.DATASET.LOCATION
        self.dataset_name = cfg.DATASET.NAME
        self.dataset_fps = cfg.DATASET.FPS
        self.model_length = cfg.MODEL.WIN_LENGTH # seconds
        self.model_hopsize = cfg.MODEL.HOP_SIZE # seconds
        if self.dataset_name == 'Milly':
            self.get_milly_items()

    def get_milly_items(self):
        milly_participants = os.listdir(self.data_dir)
        participant_videos = [os.listdir(os.path.join(self.data_dir,p,'rgb_frames')) for p in milly_participants]
        participant_videos = [item for sublist in participant_videos for item in sublist]
        
        # accumulate the number of windows that fit in a video
        self._video_records = {}
        irecord = 0
        for v in participant_videos:
            nframes = len(os.listdir(os.path.join(self.data_dir,v.split('_')[0],'rgb_frames',v)))
            vid_seconds = nframes//self.dataset_fps
            vid_nwins = int((vid_seconds - self.model_length/2)/self.model_hopsize)
            start_frame = 0
            stop_frame = 60
            for i in range(vid_nwins):
                self._video_records[irecord] = {
                        'vid_id':v,
                        'start_frame': start_frame,   
                        'stop_frame': stop_frame,    
                    }
                start_frame += self.model_hopsize*self.dataset_fps
                stop_frame += self.model_hopsize*self.dataset_fps
                irecord += 1
        

    def __getitem__(self, index):
        """
        Given the video index, return the list of frames and video
        index if the video can be fetched and decoded successfully.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            video_id (str): the video id where the frames came from 
        """

        frames = pack_frames_to_video_clip(self.cfg, self._video_records[index], temporal_sample_index)

        
        if self.cfg.MODEL.MODEL_NAME == 'SlowFast':
            # Perform color normalization.
            frames = frames.float()
            frames = frames / 255.0
            frames = frames - torch.tensor(self.cfg.DATA.MEAN)
            frames = frames / torch.tensor(self.cfg.DATA.STD)
            # T H W C -> C T H W.
            frames = frames.permute(3, 0, 1, 2)
            # Perform data augmentation.
            frames = self.spatial_sampling(
                frames,
                spatial_idx=spatial_sample_index,
                min_scale=min_scale,
                max_scale=max_scale,
                crop_size=crop_size,
            )

            label = self._video_records[index].label
            frames = utils.pack_pathway_output(self.cfg, frames)
            metadata = self._video_records[index].metadata
            return frames, label, index, metadata
        elif self.cfg.MODEL.MODEL_NAME == 'TSM':
            input_frames = []
            for i, frames in enumerate([frames_rgb, frames_flow]):
                scale = min_scale/frames.shape[1]
                frames = [
                        cv2.resize(
                            img_array.numpy(),
                            (0,0),
                            fx=scale,fy=scale,  # The input order for OpenCV is w, h.
                        )
                        for img_array in frames
                ]
                frames = np.concatenate(
                    [np.expand_dims(img_array, axis=0) for img_array in frames],
                    axis=0,
                )
                frames = torch.from_numpy(np.ascontiguousarray(frames))
                frames = torch.flip(frames,dims=[3]) if i ==0 else frames # from bgr to rgb
                frames = frames.float()
                frames = frames / 255.0
                frames = frames - torch.tensor(self.cfg.DATA.MEAN) if i==0 else frames - 0.5
                frames = frames / torch.tensor(self.cfg.DATA.STD) if i==0 else frames/0.226
                # T H W C -> C T H W.
                frames = frames.permute(3, 0, 1, 2)
                frames = self.spatial_sampling(
                    frames,
                    spatial_idx=spatial_sample_index,
                    min_scale=min_scale,
                    max_scale=max_scale,
                    crop_size=crop_size,
                )
                input_frames.append(frames)
            label = self._video_records[index].label
            metadata = self._video_records[index].metadata
            return input_frames, label, index, metadata
        elif self.cfg.MODEL.MODEL_NAME == 'Omnivore':
            scale = min_scale/frames.shape[1]
            frames = [
                    cv2.resize(
                        img_array.numpy(),
                        (0,0),
                        fx=scale,fy=scale,  # The input order for OpenCV is w, h.
                    )
                    for img_array in frames
            ]
            frames = np.concatenate(
                [np.expand_dims(img_array, axis=0) for img_array in frames],
                axis=0,
            )
            frames = torch.from_numpy(np.ascontiguousarray(frames))
            frames = torch.flip(frames,dims=[3]) # from bgr to rgb
            frames = frames.float()
            frames = frames / 255.0
            frames = frames - torch.tensor(self.cfg.DATA.MEAN)
            frames = frames / torch.tensor(self.cfg.DATA.STD)
            # T H W C -> C T H W.
            frames = frames.permute(3, 0, 1, 2)
            frames = self.spatial_sampling(
                frames,
                spatial_idx=spatial_sample_index,
                min_scale=min_scale,
                max_scale=max_scale,
                crop_size=crop_size,
            )
            label = self._video_records[index].label
            metadata = self._video_records[index].metadata
            return frames, label, index, metadata


    def __len__(self):
        return len(self._video_records)

    def spatial_sampling(
            self,
            frames,
            spatial_idx=-1,
            min_scale=256,
            max_scale=320,
            crop_size=224,
    ):
        """
        Perform spatial sampling on the given video frames. If spatial_idx is
        -1, perform random scale, random crop, and random flip on the given
        frames. If spatial_idx is 0, 1, or 2, perform spatial uniform sampling
        with the given spatial_idx.
        Args:
            frames (tensor): frames of images sampled from the video. The
                dimension is `num frames` x `height` x `width` x `channel`.
            spatial_idx (int): if -1, perform random spatial sampling. If 0, 1,
                or 2, perform left, center, right crop if width is larger than
                height, and perform top, center, buttom crop if height is larger
                than width.
            min_scale (int): the minimal size of scaling.
            max_scale (int): the maximal size of scaling.
            crop_size (int): the size of height and width used to crop the
                frames.
        Returns:
            frames (tensor): spatially sampled frames.
        """
        assert spatial_idx in [-1, 0, 1, 2]
        if spatial_idx == -1:
            frames, _ = transform.random_short_side_scale_jitter(
                frames, min_scale, max_scale
            )
            frames, _ = transform.random_crop(frames, crop_size)
            frames, _ = transform.horizontal_flip(0.5, frames)
        else:
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            #assert len({min_scale, max_scale, crop_size}) == 1
            #frames, _ = transform.random_short_side_scale_jitter(
            #    frames, min_scale, max_scale
            #)
            frames, _ = transform.uniform_crop(frames, crop_size, spatial_idx)
        return frames
