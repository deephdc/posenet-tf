import tensorflow as tf
import cv2
import time
import argparse
import os

from posenetclas.model import load_model
from posenetclas.utils import read_imgfile, draw_skel_and_kp
from posenetclas.decode_multi import decode_multiple_poses
from posenetclas.constants import PART_NAMES
import json
import base64
from posenetclas import paths



def posenet_image(timestamp):


    scale_factor=1.0
    model=101
    image_dir="/tmp/"+timestamp+"/"
    output_dir=os.path.join(paths.get_image_dir(),timestamp+"/")
    dictoutput= []

    with tf.Session() as sess:
        model_cfg, model_outputs = load_model(model, sess)
        output_stride = model_cfg['output_stride']

        if output_dir:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

        filenames = [
            f.path for f in os.scandir(image_dir) if f.is_file() and f.path.endswith(('.png', '.jpg','.jpeg'))]

        start = time.time()
        for f in filenames:
            input_image, draw_image, output_scale = read_imgfile(
                f, scale_factor=scale_factor, output_stride=output_stride)

            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
                model_outputs,
                feed_dict={'image:0': input_image}
            )

            pose_scores, keypoint_scores, keypoint_coords = decode_multiple_poses(
                heatmaps_result.squeeze(axis=0),
                offsets_result.squeeze(axis=0),
                displacement_fwd_result.squeeze(axis=0),
                displacement_bwd_result.squeeze(axis=0),
                output_stride=output_stride,
                max_pose_detections=10,
                min_pose_score=0.25)

            keypoint_coords *= output_scale

            if output_dir:
                draw_image = draw_skel_and_kp(
                    draw_image, pose_scores, keypoint_scores, keypoint_coords,
                    min_pose_score=0.25, min_part_score=0.25)
                output_image=os.path.join(output_dir, os.path.relpath(f, image_dir))
                cv2.imwrite(output_image, draw_image)
                
     

            if True:
                imgdict = {"output": output_image}
                print()
                print("Results for image: %s" % f)
                for pi in range(len(pose_scores)):
                    if pose_scores[pi] == 0.:
                        break
                    print('Pose #%d, score = %f' % (pi, pose_scores[pi]))
                    for ki, (s, c) in enumerate(zip(keypoint_scores[pi, :], keypoint_coords[pi, :, :])):
                        print('Keypoint %s, score = %f, coord = %s' % (PART_NAMES[ki], s, c))
                        imgdict[PART_NAMES[ki]]={ "coordinate_x": c[0], "coordinate_y": c[1], "score":s}
            
            dictoutput.append(imgdict)
    print('Average FPS:', len(filenames) / (time.time() - start))
    return dictoutput, output_image


