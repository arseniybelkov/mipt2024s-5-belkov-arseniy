# Documentation

```python
from ultralytics import YOLO
from src.model import predict, predict_single_image, predict_on_dir

model = YOLO('path/to/checkpoint/')
image = np.load("path/to/image")

# returns xyxyxyxy box
obb = predict(model, image)

# runs on image file
pred = predict_on_single_image(model, 'path/to/single/imagefile')
obb = pred[0].obb.xyxyxyxy.numpy()

# runs on the directory of images
pred = predict_on_dir(model, 'path/to/dir/with/images')

for img_path, res in pred.items():
    obb = res[0].obb.xyxyxyxy.numpy()
    ...
```