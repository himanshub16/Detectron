#!/usr/bin/env python2

# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Perform inference on a single image or all images with a certain extension
(e.g., .jpg) in a folder.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import os
import sys
import time

from caffe2.python import workspace

from core.config import assert_and_infer_cfg
from core.config import cfg
from core.config import merge_cfg_from_file
from utils.io import cache_url
from utils.timer import Timer
import core.test_engine as infer_engine
import datasets.dummy_datasets as dummy_datasets
import utils.c2 as c2_utils
import utils.logging
import utils.vis as vis_utils

logger = logging.getLogger(__name__)


c2_utils.import_detectron_ops()
# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='directory for visualization pdfs (default: /tmp/infer_simple)',
        default='/tmp/infer_simple',
        type=str
    )
    parser.add_argument(
        '--image-ext',
        dest='image_ext',
        help='image file name extension (default: jpg)',
        default='jpg',
        type=str
    )
    parser.add_argument(
        '--output-ext',
        dest='output_ext',
        help='extension of output file (default: png)',
        default='png',
        type=str
    )
    parser.add_argument(
        '--port',
        dest='port',
        help='port to run the flask server on',
        default=8000,
        type=int
    )
    parser.add_argument(
        '--visualize',
        dest='visualize',
        help='should detectron visualize bounding boxes',
        default=False,
        type=bool
    )
    parser.add_argument(
        'im_or_folder', help='image or folder of images', default=None
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def prepare_inference_engine(args):
    merge_cfg_from_file(args.cfg)
    cfg.NUM_GPUS = 1
    args.weights = cache_url(args.weights, cfg.DOWNLOAD_CACHE)
    assert_and_infer_cfg(cache_urls=False)
    model = infer_engine.initialize_model_from_cfg(args.weights)
    dummy_coco_dataset = dummy_datasets.get_coco_dataset()
    return model, dummy_coco_dataset


def infer_bbox_for_image(image_path, output_dir, model, dataset, visualize=False):
    outfile_name = os.path.splitext(image_path)[0]
    out_name = os.path.join(
        output_dir, '{}.{}'.format(os.path.basename(outfile_name), args.output_ext)
    )
    logger.info('Processing {} -> {}'.format(image_path, out_name))
    im = cv2.imread(image_path)
    timers = defaultdict(Timer)
    t = time.time()
    with c2_utils.NamedCudaScope(0):
        cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
            model, im, None, timers=timers
        )
    logger.info('Inference time: {:.3f}s'.format(time.time() - t))
    # for k, v in timers.items():
    #     logger.info(' | {}: {:.3f}s'.format(k, v.average_time))
    logger.info(' | bbox detection time: {:.3f}s'.format(timers['im_detect_bbox'].average_time))

    if visualize:
    	if not dataset:
             logger.error('Cannot visualize without dataset. Check code for errors')
        vis_utils.vis_one_image(
            im[:, :, ::-1],  # BGR -> RGB for visualization
            image_path,
            # args.output_dir,
            out_name,
            cls_boxes,
            cls_segms,
            cls_keyps,
            dataset=dataset,
            box_alpha=0.3,
            show_class=True,
            thresh=0.7,
            kp_thresh=2,
            ext=args.output_ext
        )
    # print(image_path, vis_utils.get_final_bounding_boxes(cls_boxes, thresh=0.7, get_class=True))
    bounding_boxes = vis_utils.get_final_bounding_boxes(cls_boxes, thresh=0.7, get_class=True, dataset=dataset)
    return bounding_boxes


def prepare_flask_app(model, dataset, visualize=True):
    from flask import Flask, request, jsonify, redirect
    from uuid import uuid4 as uuid
    app = Flask(__name__)
   
    @app.route('/getbbox', methods=['GET', 'POST'])
    def getbbox():
        if request.method == 'POST':
	    if 'file' not in request.files:
	        return redirect(request.url)
	    file = request.files['file']
            if file.filename == '':
	        return redirect(request.url)
	    if file:
	        target_file = '/tmp/'+uuid().hex+'.jpg'
	        output_dir = '/tmp/'
	        file.save(target_file)
		bboxes = infer_bbox_for_image(target_file, output_dir,
			model, dataset, visualize=True)
		print('detected', bboxes)
		return jsonify(list(bboxes))
	
	return '''
	<title>Upload new file </title>
	<h1>Upload new file</h1>
	<form method=post enctype=multipart/form-data>
	    <p><input type=file name=file>
	    <input type=submit value=Upload>
	</form>
	'''

    return app

if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    utils.logging.setup_logging(__name__)
    args = parse_args()
    # main(args)
    model, dataset = prepare_inference_engine(args)
    image_path = '/home/irm15006/Detectron/demo/pcd0162gray.jpg'
    #infer_bbox_for_image(image_path, args.output_dir, model, dataset, visualize=True)

    flask_app = prepare_flask_app(model, dataset, args.visualize)
    print('starting flask app on port', args.port)
    flask_app.run(host="0.0.0.0", port=args.port, debug=True)
