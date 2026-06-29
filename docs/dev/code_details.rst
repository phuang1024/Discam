Code details
============

Discam is written in Python.

Code design
-----------

TODO

Every component of Discam is designed to be *streamable*, or able to process frames on
the fly and return results at every iteration.
Deploying as an automatic PTZ camera requires such ability.

Therefore, most components in the pipeline are implemented as a ``class``, with an ``update``
method to process a single frame.

When running Discam in Post Processing, the pipeline is run sequentially through the video.

Pipeline
--------

The overall pipeline consists of a few steps.

.. list-table:: Pipeline steps
   :widths: 20 80
   :header-rows: 1

   * - Task
     - Description
   * - Input processing
     - Read input video, check resolution and FPS, etc..
   * - Detection
     - Detect person bounding boxes.
   * - Perspective
     - Camera location and perspective estimation.
   * - Classification
     - Active and inactive player classification.
   * - Output processing
     - Write output video, apply video cropping, command PTZ camera, etc..

Details can be found in their corresponding docs.

Configuration
-------------

