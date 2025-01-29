import argparse
import collections
import logging
import json
from pathlib import Path
from natsort import natsorted
from pycocotools.coco import COCO
from script_utils.common import common_setup
from tqdm import tqdm
from tao.utils import fs
from burstapi import BURSTDataset

def default_arg_parser():
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Input/Output parameters
    parser.add_argument('--annotations', type=Path, required=False,
                        default='/home/cc/tao/TAO-Amodal/annotations/train.json',
                        help='Path to TAO-Amodal annotation json.')
    parser.add_argument('--mask-annotations', type=Path, required=False,
                        default='/home/cc/tao/TAO-Amodal/BURST_annotations/train/all_classes_visibility.json',
                        help='The path to your BURST annotation json.')
    parser.add_argument('--output-dir', type=Path, required=False,
                        default='/home/cc/cache1',
                        help='Output folder where you want to save your visualization results.')
    parser.add_argument('--images-dir', type=Path, required=False,
                        default='/home/cc/tao/TAO-Amodal/frames',
                        help=('Path to TAO-Amodal/frames.'))

    # Other parameters
    parser.add_argument('--split', type=str, 
                        default='train',
                        help="Dataset split (e.g., train/val/test).")
    parser.add_argument('--video-name', type=str, nargs='*', default=None,
                        help='If specified, only the specified videos will be visualized.')

    args = parser.parse_args()
    return args

def process_annotations(coco, videos, args):
    annotations_dict = {}

    for video, labeled_frames in tqdm(videos.items(), desc="Processing Videos"):
        video_dict = {}

        # Ensure frames directory exists
        frames_dir = args.images_dir / video
        if not frames_dir.exists():
            logging.warning(f"Could not find images at {frames_dir}")
            continue

        frame_infos = {
            x['file_name'].split('/')[-1]: x
            for x in labeled_frames
        }
        frames = natsorted(fs.glob_ext(frames_dir, fs.IMG_EXTENSIONS))

        frame_annotations = {
            frame.rsplit('.', 1)[0]: coco.imgToAnns[info['id']]
            for frame, info in frame_infos.items()
        }

        cats = coco.cats.copy()
        for cat in cats.values():
            if cat['name'] == 'baby':
                cat['name'] = 'person'

        # Collect annotations for each frame
        for frame in frames:
            frame_name = Path(frame).stem
            if frame_name in frame_annotations:
                annotations = frame_annotations[frame_name]
                labels = [cats[ann['category_id']]['name'] for ann in annotations]
                video_dict[frame_name] = labels
            else:
                video_dict[frame_name] = []

        annotations_dict[video] = video_dict

    # Save as a single JSON file
    output_file = args.output_dir / f"{args.split}_annotations.json"
    with open(output_file, "w") as f:
        json.dump(annotations_dict, f, indent=4)

def main():
    '''
    * Process Arguments
    '''
    args = default_arg_parser()
    args.output_dir.mkdir(exist_ok=True, parents=True)
    common_setup(__file__, args.output_dir, args)

    '''
    * Create Dataset to load videos
    '''
    coco = COCO(args.annotations)

    '''
    * Process all videos in the dataset
    '''
    videos = collections.defaultdict(list)
    for image in coco.imgs.values():
        if 'video' in image:
            video = str(Path(image['video']).with_suffix(''))
        else:
            video = image['file_name'].split('/')[-2]
        videos[video].append(image)

    '''
    * Process annotations for each video and save in JSON format
    '''
    process_annotations(coco, videos, args)

if __name__ == "__main__":
    main()
