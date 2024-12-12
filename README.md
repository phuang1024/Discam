# Discam

AI motorized camera for Ultimate filming and analysis.

## Usage

Preprocess data:

```py
python preprocess_data.py /path/to/videos/ /path/to/output/
```

Train model:

```py
python train.py --data /path/to/data/
```

Live testing:

```py
python live_test.py --model /path/to/model.pt --video /path/to/video.mp4
```
