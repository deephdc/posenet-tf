import time
import os
import json
import base64

import tensorflow as tf
import cv2

from posenetclas.model import load_model
from posenetclas.utils import read_imgfile, draw_skel_and_kp
from posenetclas.decode_multi import decode_multiple_poses
from posenetclas.constants import PART_NAMES


loaded = False
sess, model_cfg, model_outputs = None, None, None


def load_predict_model():
    print('Loading model ...')
    global loaded, sess, model_cfg, model_outputs

    tf.reset_default_graph()
    sess = tf.Session()
    model_cfg, model_outputs = load_model(model_id=101, sess=sess)
    loaded = True


def predict_images(filepaths, output_dir=None):
    scale_factor = 1.0
    outputs = []

    if not loaded:
        load_predict_model()
    output_stride = model_cfg['output_stride']

    start = time.time()
    for f in filepaths:
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

        imgdict = {}

        if output_dir:
            draw_image = draw_skel_and_kp(
                draw_image, pose_scores, keypoint_scores, keypoint_coords,
                min_pose_score=0.25, min_part_score=0.25)
            fname = os.path.splitext(os.path.basename(f))[0]
            output_path = os.path.join(output_dir, fname + '.png')
            cv2.imwrite(output_path, draw_image)
            imgdict['img_path'] = output_path

        if True:
            # print("Results for image: %s" % f)
            for pi in range(len(pose_scores)):
                if pose_scores[pi] == 0.:
                    break
                # print('Pose #%d, score = %f' % (pi, pose_scores[pi]))
                for ki, (s, c) in enumerate(zip(keypoint_scores[pi, :], keypoint_coords[pi, :, :])):
                    # print('Keypoint %s, score = %f, coord = %s' % (PART_NAMES[ki], s, c))
                    imgdict[PART_NAMES[ki]] = {"coordinate_x": c[0], "coordinate_y": c[1], "score": s}

        outputs.append(imgdict)
    # print('Average FPS:', len(filepaths) / (time.time() - start))

    # Save json
    if output_dir:
        fpath = os.path.join(output_dir, 'output.json')
        with open(fpath, 'w') as fp:
            json.dump(outputs, fp)

    return outputs
