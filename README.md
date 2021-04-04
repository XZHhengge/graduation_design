## 基于双目定位的自动驾驶小车
## 难点：不管是树莓派还是主机下的图像处理能力都很慢，导致定位不够即使，小车电池电压下降，左右轮的PWM调速不能为固定值



##  未来构想图
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200922165719852.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQwOTY1MTc3,size_16,color_FFFFFF,t_70#pic_center)


## 总体分为两部分
# 硬件:树莓派3一个，L298N电机一个，树莓派官方摄像头一个，HC-SR04超声波模块一个.使用一个12V的电池两头供电,一个12V转到5V供到树莓派，一个12V直接供给电机
参考:
[Sunny的树莓派小车DIY教程（附视频）](https://shumeipai.nxez.com/2015/11/08/raspberry-pi-car-diy-tutorials-by-sunny.html)
[树莓派上使用HC-SR04超声波测距模块](https://shumeipai.nxez.com/2019/01/02/hc-sr04-ultrasonic-ranging-module-on-raspberry-pi.html)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200929100412877.png#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200929100436796.png#pic_center)


树莓派3的GPIO图
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020092217000072.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQwOTY1MTc3,size_16,color_FFFFFF,t_70#pic_center)
##  成品图:
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200922170717523.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQwOTY1MTc3,size_16,color_FFFFFF,t_70#pic_center)

# 软件部分
##  定位使用双目摄像头进行定位小车位置
先使用matlab对双目摄像头进行标定
参考 
[python、opencv 双目视觉测距代码](https://blog.csdn.net/ilovestudy2/article/details/106340085?utm_medium=distribute.pc_aggpage_search_result.none-task-blog-2~all~first_rank_v2~rank_v25-2-106340085.nonecase&utm_term=%E5%8F%8C%E7%9B%AE%E8%A7%86%E8%A7%89%E6%B5%8Bpython%E5%AE%9E%E7%8E%B0)
[使用OpenCV/python进行双目测距](https://www.cnblogs.com/zhiyishou/p/5767592.html)
[matlab双目标定（详细过程）](https://blog.csdn.net/qq_38236355/article/details/89280633)
[双目测距理论及其python实现！
](https://blog.csdn.net/dulingwen/article/details/98071584?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1.channel_param&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1.channel_param)
[摄像机标定和立体标定](https://blog.csdn.net/sunanger_wang/article/details/7744025)

**其中：**

```python
左右相机畸变系数:[k1, k2, p1, p2, k3] 
k1, k2, k3 = CameraParameters1.RadialDistortion[0, 1, 2], 
p1, p2 = CameraParameters1.TangentialDistortion[0, 1]
T = 平移矩阵 stereoParams.TranslationOfCamera2
R = 旋转矩阵 stereoParams.RotationOfCamera2
```
##  标定前
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200922190326461.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQwOTY1MTc3,size_16,color_FFFFFF,t_70#pic_center)
##  标定后
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200922190427696.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQwOTY1MTc3,size_16,color_FFFFFF,t_70#pic_center)
##  深度图
记录:
mode=cv2.STEREO_SGBM_MODE_HH 不同模式之间没有太特别差异
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200923110336448.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQwOTY1MTc3,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200923110409388.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQwOTY1MTc3,size_16,color_FFFFFF,t_70#pic_center)
##  总的来说，调节 num 、blockSize和UniquenessRatio是比较明显的

##  定位

```python
    '''
        位置的摆放如下，单位都是cm
    ---------------------------------MARK
    -                                                          -
    -                        CAR     					    -
    -                       / |         						-
    -                      /  |         						-
    -                     /   |         						-
    -                    /    |         						-
    -                   /     |        					    -
    ---------------CAMERA-----------------
    :param mark_pos_ofmap: 标记物在真实地图上的位置
    :param camera_pos_ofmap: 相机在真实地图上的位置
    :param power_pos_ofcamera: 电池在相机上的位置
    :param mark_pos_ofcamera: 标记物在相机上的位置
    :param threeD:   使用MATLAB矫正和opencv得到的深度矩阵
    '''
    #  计算深度
    car_deepth = threeD[power_pos_ofcamera[1]][power_pos_ofcamera[0]][-1]
    mark_deepth = threeD[mark_pos_ofcamera[1]][mark_pos_ofcamera[0]][-1]
    if car_center != 'inf' and mark_deepth != 'inf':
        if car_deepth > 0 and mark_deepth > 0:
            # car_deepth /= 10.0
            mark_deepth /= 10.0
            # if abs(mark_deepth - math.sqrt(mark_pos_ofmap[0]**2 + mark_pos_ofmap[1] ** 2)) > 50:
            #     print("标记与相机测量误差太大")
            # else:
            # 计算小车与标记的横向距离,两点之间的距离在除以5，这个5是通过实际测量和图像点计算出来的
            x_length = math.sqrt((power_pos_ofcamera[0] - mark_pos_ofcamera[0]) ** 2
                                 + ((power_pos_ofcamera[1] - mark_pos_ofcamera[1]) ** 2)) / 5.0
            # 在通过标记的坐标得到小车的横向坐标
            x = mark_pos_ofmap[0] - x_length
            # print("小车与标记的距离为{}, 横向坐标为{}".format(x_length, x))
            # 再用勾股定理得到y
            y = math.sqrt(car_deepth ** 2 - (x - camera_pos_ofmap[0]) ** 2)
            print("坐标为{},{}".format(x, y/10.0))
            return x, y / 10
```
**效果还可以**
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020092619254525.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQwOTY1MTc3,size_16,color_FFFFFF,t_70#pic_center)


## 小车自动驾驶：方法一：深度学习，方法二：通过车道的直角判断（不理想）

# 方法二的步骤：
 1.灰度化 
 2.高斯模糊
 3.Canny边缘检测
 4.不规则ROI区域截取
 5.霍夫直线检测
 6.车道计算
 7.直角计算



## opencv画车道
[[1]](http://codec.wang/#/opencv/basic/challenge-03-lane-road-detection)
[[2]](https://github.com/Sentdex/pygta5/blob/master/Tutorial%20Codes/Part%201-7/part-5-line-finding.py)

##  后面继续更新
