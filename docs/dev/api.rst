API
===

src/cv/classify.py
------------------

.. autoclass:: discam.cv.classify.Classifier
   :members:

.. autofunction:: discam.cv.classify.stddev_filter

src/cv/detect.py
----------------

.. autoclass:: discam.cv.detect.Detector
   :members:

src/cv/perspective.py
---------------------

.. autoclass:: discam.cv.perspective.ComputePersp
   :members:

.. autofunction:: discam.cv.perspective.vis_vanishing

src/cv/pipeline.py
------------------

.. autoclass:: discam.cv.pipeline.CVPipeline
   :members:

.. autofunction:: discam.cv.pipeline.vis_frame

.. autofunction:: discam.cv.pipeline.vis_locations

src/post/bounding_box.py
------------------------

.. autofunction:: discam.post.bounding_box.extract_box

.. autofunction:: discam.post.bounding_box.lerp_boxes

.. autoclass:: discam.post.bounding_box.SmoothEMA
   :members:

.. autofunction:: discam.post.bounding_box.ema_smooth_boxes

.. autofunction:: discam.post.bounding_box.resize_box

.. autofunction:: discam.post.bounding_box.moving_average

.. autofunction:: discam.post.bounding_box.compute_final_boxes

.. autofunction:: discam.post.bounding_box.vis_static_box

src/cv/run_pipe.py
------------------

.. autofunction:: discam.post.run_pipe.post_run_pipeline

src/cv/trim.py
--------------

.. autofunction:: discam.post.trim.find_plateaus

.. autofunction:: discam.post.trim.find_trim_sections

.. autofunction:: discam.post.trim.gen_timestamps

src/utils/constants.py
----------------------

.. automodule:: discam.utils.constants
   :members:

src/cv/logger.py
----------------

.. autofunction:: discam.utils.logger.init_logger

.. autofunction:: discam.utils.logger.add_scalar

.. autofunction:: discam.utils.logger.add_image

src/cv/video_rw.py
------------------

.. autofunction:: discam.utils.video_rw.FFmpegWriter

.. autofunction:: discam.utils.video_rw.post_write_video

.. autofunction:: discam.utils.video_rw.vis_output_video

src/field_mask_main.py
----------------------

.. automodule:: discam.field_mask_main
   :members:

src/live_main.py
----------------

.. automodule:: discam.live_main
   :members:

src/post_main.py
----------------

.. automodule:: discam.post_main
   :members:
