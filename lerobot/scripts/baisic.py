from PyQt6.QtGui import QImage, QPixmap,qRgb
#server levels for ui
ConnectionClosed=3
ClientConnected=2
WaitCon=1
NotStarted=0

# Set models directory
models_dir =  "outputs/train/" #dont change 
dataset_dir = "outputs/dataset/" #dont change 


def ndarray_to_qimage(ndarray):
        if ndarray.ndim == 2:  # 灰度图像
            height, width = ndarray.shape
            qimage = QImage(width, height, QImage.Format.Format_Grayscale8)
            for y in range(height):
                for x in range(width):
                    qimage.setPixel(x, y, qRgb(ndarray[y, x], ndarray[y, x], ndarray[y, x]))
            return qimage
        elif ndarray.ndim == 3 and ndarray.shape[2] == 3:  # RGB图像
            height, width, _ = ndarray.shape
            qimage = QImage(width, height, QImage.Format.Format_RGB888)
            for y in range(height):
                for x in range(width):
                    r, g, b = ndarray[y, x]
                    qimage.setPixel(x, y, qRgb(r, g, b))
            return qimage
        elif ndarray.ndim == 3 and ndarray.shape[2] == 4:  # RGBA图像
            height, width, _ = ndarray.shape
            qimage = QImage(width, height, QImage.Format.Format_ARGB32)
            for y in range(height):
                for x in range(width):
                    r, g, b, a = ndarray[y, x]
                    qimage.setPixel(x, y, qRgb(r, g, b))
            return qimage
        else:
            raise ValueError("Unsupported ndarray shape")