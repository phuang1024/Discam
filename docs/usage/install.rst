Installation
============

Discam is written in Python.
It requires a (reasonably modern) Python environment and the dependencies.

Then, use various Discam utilities by running Python files.

.. code-block:: bash

   git clone https://github.com/phuang1024/Discam --depth=1
   cd Discam
   pip install -r requirements.txt
   cd src

   # E.g. to run post processing:
   python post_main.py /path/to/video.mp4
