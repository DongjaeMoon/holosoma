from pathlib import Path
import cv2
p=Path('/workspace/holosoma/logs/bilateral_queue/corrected_v2_20261029_0513/evaluation')
v=next((p/'videos/biHAM/model_0010000/left_0.6mps').glob('*.mp4'))
c=cv2.VideoCapture(str(v))
print(c.get(3),c.get(4),c.get(5),c.get(7),flush=True)
ok,f=c.read()
assert ok
cv2.imwrite(str(p/'composition_preview_source.png'),f)
