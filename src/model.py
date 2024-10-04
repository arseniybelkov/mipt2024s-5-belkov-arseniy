import argparse
from pathlib import Path

import numpy as np
from jboc import composed
from tqdm import tqdm
from deli import save_json
from ultralytics import YOLO


"""
In order to use the model, do the following.

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
"""


def main(args):
    model_path = Path(args.path).resolve()
    model = YOLO(model_path)
    print(f">>> Loaded model from {str(model_path.resolve())}", flush=True)
    model.to(args.device)
    print(f">>> Inference happens on {args.device} device", flush=True)
    
    if (args.dir is not None) and (args.image is not None):
        raise ValueError(f"Only one option of image location can be chosen "
                         f", you have {args.dir=}, {args.image=}")
        
    if args.image is not None:
        image_path = Path(args.image).resolve() 
        results = predict_single_image(model, args.image)
        _save_result(image_path, results)
        
    elif args.dir is not None:
        for (p, results) in predinct_on_dir(model, args.dir).items():
            _save_result(p, results)
    else:
        raise ValueError("Neither option of image location was chosen.")
    

def predict_single_image(model, image_path: str):
    """
    Runs model on image specified in `image_path`.
    Outputs `result` with all the necessary information about predictions. 
    
    Example:
    path = 'path/to/image.jpg'
    model = YOLO(...)
    results = predict_single_image(model, path)
    obb = results[0].obb
    """
    image_path = Path(image_path).resolve()
    results = model(image_path)
    return results
    

@composed(dict)
def predinct_on_dir(model, directory: str):
    """
    Runs model on images specified in `directory`.
    Outputs dict of (image_path, result) with all the necessary information about predictions. 
    
    Example:
    path = 'path/to/image_folder'
    model = YOLO(...)
    results = predict_single_image(model, path)
    obb = results[path_to_image][0].obb
    """
    for p in tqdm(Path(directory).iterdir()):
        if p.suffix.endswith(("png", "jpeg", "jpg")):
            yield p, predict_single_image(model, p)


def predict(model, image: np.ndarray) -> np.ndarray:
    """
    Runs model on np.ndarray.
    Outputs OBB in format `xyxyxyxy`. 
    
    Example:
    array = np.load('path/to/image.npy')
    model = YOLO(...)
    obbs = predict(model, array)
    """
    return model(image)[0].obb.xyxyxyxy.to("cpu").numpy()


def _save_result(image_path, results):
    save_dir = image_path.with_suffix("").with_name(image_path.stem + "_predicted")
    save_dir.mkdir(exist_ok=True)
    results[0].save(save_dir / "predict.jpg")
    save_json(results[0].obb.xyxyxyxy.to("cpu").tolist(), save_dir / "obb.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", help="Path to the model checkpoint.")
    parser.add_argument("--dir", "-d", help="Path to directory with images.", default=None)
    parser.add_argument("--image", "-i", help="Path to single local image.", default=None)
    parser.add_argument("--device", choices=["cuda", "mps", "cpu"], default="cpu", help="torch.device for inference.")
    
    args = parser.parse_args()
    
    main(args)