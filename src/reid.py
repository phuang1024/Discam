"""
Wrapper around Clip ReID.
"""

from clip_reid.config import cfg
from clip_reid.model.make_model_clipreid import make_model


cfg.merge_from_file("reid_config.yml")
cfg.freeze()
print(cfg)

model = make_model(cfg, num_class=751, camera_num=6, view_num=1)
model.load_param("clip_reid.pt")
