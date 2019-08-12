# 项目介绍

根据论文<Combining Sketch and Tone for Pencil Drawing Production> 的算法实现将图片转描为铅笔画风格, 包含黑白效果以及彩色效果.

通过opencv检测手势, 当检测到四根手指时,自动拍下一张照片交由后端处理成为铅笔画照片.

### 文件作用

`pencil.py`: 本地测试用文件,读入一张`RGB`图片, 将其转化为黑白和彩色铅笔画图片;

`gesture.py`: 视频手势检测主文件;

`color.py`: 彩色铅笔画处理;

`black.py`: 黑白铅笔画处理;

`stitch.py`:  图片扩展处理;

`util.py`: 图片空间处理;

