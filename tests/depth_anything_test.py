import time

from transformers import pipeline
from PIL import Image

pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
image = Image.open("../data/videos_short/NW_30s.jpg")
image = image.resize((960, 540))

time_start = time.time()
outputs = pipe(image)
#outputs["depth"].save("depth.jpg")
print("Infer time:", time.time() - time_start)

# Tensor (H, W) float32
depth = outputs["predicted_depth"]


import matplotlib.pyplot as plt
xs = []
ys = []
for y in range(0, depth.shape[0], 12):
    for x in range(0, depth.shape[1], 12):
        xs.append(y)
        ys.append(depth[y, x])

plt.scatter(xs, ys)
plt.show()
