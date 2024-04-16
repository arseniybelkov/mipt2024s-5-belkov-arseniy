# iitp-modern-cv
IITP course on modern Computer Vision
## 16-04-2024
- https://www.kaggle.com/code/arseniybelkov/obb-detection/notebook?scriptVersionId=172346702 - модель обучается  
Взял [модель](https://docs.ultralytics.com/tasks/obb/#train) от [ultralytics](https://github.com/ultralytics/ultralytics), выбирал по принципу "легче всего запустить".
Сейчас она учится на grayscale картинках с искаженными баркодами, по одному коду на картинку, интенсивность фона - рандомная. 
## 27-03-2024
- https://www.kaggle.com/code/arseniybelkov/barcodes - каггл ноутубук с генерацией данных (needs to be thorougly tested ofc)

## 20-03-2024
Поставили задачу rotated rectangles detection  
- Tasks:
  - Архитектура,метрики и лосс функции (https://arxiv.org/pdf/2012.13135.pdf, https://arxiv.org/pdf/2205.12785.pdf, [RG](https://www.researchgate.net/publication/377163595_HODet_A_New_Detector_for_Arbitrary-Oriented_Rectangular_Object_in_Optical_Remote_Sensing_Imagery), https://github.com/lilanxiao/Rotated_IoU, https://github.com/jbwang1997/OBBDetection)
  - __Данные (найти открытые / ждем генерацию), получить боксы__ - программа минимум на 27-03-2024
  - _Обучить на сгенеренных картинках (минимум - баркод на однородном фоне)_ - программа максимум  
- Проблемы:  
  - В выше указанных статьях объекты +- одного скейла, у нас же разница может быть довольно большой (тут мб нас спасет Unet-like, или DETR)
  - Для квадратных qr-ов поворот не существеннен, для вытянутых существеннен
