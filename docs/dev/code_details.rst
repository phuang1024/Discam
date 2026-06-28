Code details
============

Discam is written in Python.

Code design
-----------

TODO

Every component of Discam is designed to be *streamable*, or able to
process frames on the fly and return results at every iteration.
Deploying as an automatic PTZ camera requires such ability.

Therefore, most components in the pipeline are implemented as a ``class``,
with an ``update`` method to process a single frame.

When running Discam in post processing, the pipeline is run sequentially
through the video.

