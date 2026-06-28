Installation
============

Discam is written in Python.
Install a Python 3 environment, and the required packages.

Then, use various Discam utilities by running Python files.

In the future, Discam will be packaged in a Python module with executable entry points.

.. code-block:: bash

   git clone https://github.com/phuang1024/Discam --depth=1
   cd Discam
   pip install -r requirements.txt
   cd src

   # E.g. to run post processing:
   python post_main.py /path/to/video.mp4
