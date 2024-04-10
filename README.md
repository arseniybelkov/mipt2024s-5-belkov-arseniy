# iitp-modern-cv
IITP course on modern Computer Vision
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
