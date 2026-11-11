# 导入OpenCV库，用于图像处理
import cv2
# 导入numpy库，用于数值计算和数组操作
import numpy as np

# 从cv2库中导入boundingRect函数，用于计算轮廓的外接矩形
from cv2 import boundingRect

# 读取图片文件（4.jpg为目标药片图片，需确保路径正确）
img = cv2.imread("4.jpg")

# 将彩色图片转换为灰度图像，减少计算量，聚焦亮度特征
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 对灰度图进行高斯模糊处理，核大小5×5，消除背景噪点
tem = cv2.GaussianBlur(img_gray, (5, 5), 0)

# 对模糊后的图像进行二值化处理
# 阈值191：亮度高于191的像素设为255（白色），低于的设为0（黑色）
# cv2.THRESH_BINARY：二值化模式，分割白色药片和背景
tem = cv2.threshold(tem, 191, 255, cv2.THRESH_BINARY)

# 创建9×9的矩形结构元素，用于形态学运算
kernel = np.ones((9, 9), np.uint8)
# 对二值化图像进行闭运算处理（先膨胀后腐蚀）
# 作用：补全药片轮廓的细微断裂，让轮廓更完整
th = cv2.morphologyEx(tem[1], cv2.MORPH_CLOSE, kernel)
# 提取阈值处理的返回值（tem[0]为使用的阈值，tem[1]为处理后的图像）
rv = tem[0]

# 查找图像中的外部轮廓（只保留最外层轮廓，忽略内部孔洞）
# cv2.RETR_EXTERNAL：只检索最外层轮廓
# cv2.CHAIN_APPROX_SIMPLE：压缩轮廓点，减少内存占用
min_c = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# 提取轮廓列表（min_c[0]为轮廓数组，min_c[1]为轮廓层级信息）
top = min_c[0]

# 筛选有效轮廓：面积在2000~120000之间
# 过滤掉过小的噪点轮廓和过大的背景轮廓，只保留药片轮廓
valid = [c for c in top if 2000 < cv2.contourArea(c) < 120000]
# 确定目标轮廓：优先选有效轮廓中面积最大的（即药片）；若无有效轮廓则选所有轮廓中最大的
bot = max(valid, key=cv2.contourArea) if len(valid) > 0 else max(top, key=cv2.contourArea)
# 计算目标轮廓的外接矩形，返回(x, y, w, h)：x/y为左上角坐标，w/h为宽/高
cv2_box = cv2.boundingRect(bot)

# 打印药片的坐标信息：左上角坐标 + 右下角坐标
print(f"topLeft, bottomRight {cv2_box[:2]} {(cv2_box[0]+cv2_box[2], cv2_box[1]+cv2_box[3])}")
# 重新读取原图（避免之前的灰度/二值化处理影响显示）
img = cv2.imread("4.jpg")
# 在原图上绘制白色矩形框（线宽4），框住药片轮廓
# cv2_box[:2]：左上角坐标，(cv2_box[0]+cv2_box[2], cv2_box[1]+cv2_box[3])：右下角坐标
# (255, 255, 255)：白色，4：线宽
cv2.rectangle(img, cv2_box[:2], (cv2_box[0]+cv2_box[2], cv2_box[1]+cv2_box[3]), (255, 255, 255), 4)
# 创建显示窗口并设置为可调整大小模式
cv2.namedWindow("img", cv2.WINDOW_NORMAL)
# 设置窗口显示大小为1200×800，方便查看
cv2.resizeWindow("img", 1200, 800)
# 显示绘制了框选的图片
cv2.imshow("img", img)
# 等待按键输入（按任意键关闭窗口）
cv2.waitKey(0)
# 销毁所有显示窗口，释放内存
cv2.destroyAllWindows()
