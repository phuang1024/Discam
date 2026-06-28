API
===

src/cv
------

.. autoclass:: cv.classify.Classifier
   :members:

.. autofunction:: cv.classify.stddev_filter

.. autoclass:: cv.detect.Detector
   :members:

.. autoclass:: cv.perspective.ComputePersp
   :members:

.. autofunction:: cv.perspective.vis_vanishing

.. autoclass:: cv.pipeline.Pipeline
   :members:

.. autofunction:: cv.pipeline.post_run_pipeline

.. autofunction:: cv.pipeline.vis_frame

.. autofunction:: cv.pipeline.vis_locations

src/post
--------

.. autofunction:: post.bounding_box.extract_box

.. autofunction:: post.bounding_box.median_filter

.. autofunction:: post.bounding_box.lerp_boxes

.. autoclass:: post.bounding_box.SmoothEMA
   :members:

.. autofunction:: post.bounding_box.ema_smooth_boxes

.. autofunction:: post.bounding_box.moving_average

.. autofunction:: post.bounding_box.resize_box

.. autofunction:: post.bounding_box.compute_final_boxes

.. autofunction:: post.bounding_box.vis_static_box

src/utils
---------

.. automodule:: utils.constants
   :members:

.. autofunction:: utils.field_mask.create_mask

.. autofunction:: utils.field_mask.create_persp_scale

.. autofunction:: utils.video_rw.FFmpegWriter

.. autofunction:: utils.video_rw.post_write_video

.. autofunction:: utils.video_rw.vis_output_video
