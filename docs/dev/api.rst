API
===

src/cv
------

.. autoclass:: discam.cv.classify.Classifier
   :members:

.. autofunction:: discam.cv.classify.stddev_filter

.. autoclass:: discam.cv.detect.Detector
   :members:

.. autoclass:: discam.cv.perspective.ComputePersp
   :members:

.. autofunction:: discam.cv.perspective.vis_vanishing

.. autoclass:: discam.cv.pipeline.Pipeline
   :members:

.. autofunction:: discam.cv.pipeline.post_run_pipeline

.. autofunction:: discam.cv.pipeline.vis_frame

.. autofunction:: discam.cv.pipeline.vis_locations

src/post
--------

.. autofunction:: discam.post.bounding_box.extract_box

.. autofunction:: discam.post.bounding_box.median_filter

.. autofunction:: discam.post.bounding_box.lerp_boxes

.. autoclass:: discam.post.bounding_box.SmoothEMA
   :members:

.. autofunction:: discam.post.bounding_box.ema_smooth_boxes

.. autofunction:: discam.post.bounding_box.moving_average

.. autofunction:: discam.post.bounding_box.resize_box

.. autofunction:: discam.post.bounding_box.compute_final_boxes

.. autofunction:: discam.post.bounding_box.vis_static_box

src/utils
---------

.. automodule:: discam.utils.constants
   :members:

.. autofunction:: discam.utils.field_mask.create_mask

.. autofunction:: discam.utils.field_mask.create_persp_scale

.. autofunction:: discam.utils.video_rw.FFmpegWriter

.. autofunction:: discam.utils.video_rw.post_write_video

.. autofunction:: discam.utils.video_rw.vis_output_video
