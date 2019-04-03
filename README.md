# FSRCNN-OpenCV

C++ implementation of the Fast Super-Resolution Convolutional Neural Network (FSRCNN).

This implements two models: FSRCNN which is more accurate but slower and FSRCNN-s which is faster but less accurate. 

Based on this [paper](http://mmlab.ie.cuhk.edu.hk/projects/FSRCNN.html).


## Prerequisites
* \>Eigen-3.3.7
* \>OpenCV-3.4.5

## Usage

```bash
Usage: FSRCNN-OpenCV.exe <option(s)> SOURCE-IMG
Options:
        -f,--fast True/False    Use FSRCNN-s or FSRCNN
        -s,--scale 2/3  Specify the scale num
```
## Result

**Original Comic image:**

![](https://raw.githubusercontent.com/thinkerleolee/FSRCNN-OpenCV/master/result/comic.bmp)

**Bicubic Comic image X3:**

![](https://raw.githubusercontent.com/thinkerleolee/FSRCNN-OpenCV/master/result/res_bicubic_comic.bmp)

**FSRCNN Comic image X3:**

![](https://raw.githubusercontent.com/thinkerleolee/FSRCNN-OpenCV/master/result/res_fsrcnn_comic.bmp)

## References

- [igv/SRCNN-Tensorflow](https://github.com/igv/FSRCNN-TensorFlow)