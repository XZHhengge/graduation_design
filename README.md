## 基于双目定位的自动驾驶小车
[github链接](https://github.com/XZHhengge/graduation_design)
##  基本构思：
1.使用树莓派作为小车的操作中心，树莓派通过摄像头进行道路的检测，进而使用PWM模式对L298N电机驱动模块进行对小车轮子的前进、后退、左右转弯。
2.主机使用MATLAB对双目摄像头进行标定，使用SGBM/BM算法进行对小车的空间的三维坐标的建立。
3.在主机pygame界面上构建出真实地图的平面图，通过双目摄像头获得到小车真实的三维坐标，主机在pygame上的平面图出小车的位置，并把真实的三维坐标和地图信息通过tcp发给树莓派小车。
树莓派小车通过道路检测和地图信息并使用Dijkstra进行最短路径的路线规划并进行自动驾驶，并实时通过接收真实的三维坐标进行自我矫正和分析。
## 难点：不管是树莓派还是主机下的图像处理能力都很慢，导致定位不够即使，小车电池电压下降，左右轮的PWM调速不能为固定值导致转弯困难



把raspberryPi文件夹移动树莓派，分别运行mian.py
.
├── car2.png  小车图  
├── config.py   配置文件
├── main.py
├── paint1.jpg   地图
├── pc  主机下文件
│   ├── camera_dir
│   │   ├── camera_configs2.py  # 摄像头标定后参数
│   │   ├── color_track.py  #  双目摄像头定位
│   │   ├── depth2.py        # 单个运行标定后效果文件
│   │   ├── depth.py         # 同上
│   │   ├── get_hsv.py       #  单个运行调整参数得到hsv
│   │   └── take_piture.py   #  同上上
│   ├── graph_dir             #  地图规划信息dir
│   │   ├── Graph3.py         # 地图规划sample脚本
│   │   └── Graph.py          #  被调用的地图规划
│   ├── process_image          # 图像处理dir，先在主机上写图像处理的代码
│   │   └── process.py          # 图像处理
│   ├── pygame_dir
│   │   ├── car.png             # 小车图
│   │   └── pygame_display.py   # pygame显示
│   ├── raspPi                  #与树莓派相关
│   ├── Raspberry_Pi3_GIOP.png
│   └── server_stream.py
├── raspberryPi         # 树莓派里
│   ├── config.py
│   ├── distance_check.py
│   ├── drive_car.py
│   ├── main.py
│   ├── path.py
│   ├── process_img.py
│   ├── README.md
│   ├── requirements.txt
│   ├── stream_client.py
│   └── test.py
├── README.md
└── requirements.txt

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
##  要注意Python驱动GPIO的模块为RPi.GPIO,下面这个BOARD模式指的是Pin#，不是NAME（如下图所示）
```python
import RPi.GPIO as GPIO
import time
from config import Road, Car
# 设置 GPIO 模式为 BOARD
GPIO.setmode(GPIO.BOARD)
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/2020092217000072.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQwOTY1MTc3,size_16,color_FFFFFF,t_70#pic_center)
##  成品图:
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200922170717523.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQwOTY1MTc3,size_16,color_FFFFFF,t_70#pic_center)
##  双目摄像头如下
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210404192953419.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQwOTY1MTc3,size_16,color_FFFFFF,t_70)

# 软件部分
##  使用双目摄像头进行定位小车
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
##  MARK（红色）如下图所示：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210404192758299.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQwOTY1MTc3,size_16,color_FFFFFF,t_70)


```python
def get_coordinate(mark_pos_ofcamera: tuple, power_pos_ofcamera: tuple,
                   camera_pos_ofmap: tuple, threeD, mark_pos_ofmap) -> tuple:
    '''
        一般位置的摆放如下，单位都是cm
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
**定位如下**
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020092619254525.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQwOTY1MTc3,size_16,color_FFFFFF,t_70#pic_center)
##  地图模拟
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201014095217807.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQwOTY1MTc3,size_16,color_FFFFFF,t_70#pic_center)
##  真实地图如下
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210404193056935.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQwOTY1MTc3,size_16,color_FFFFFF,t_70)
## opencv画车道参考：
[[1]](http://codec.wang/#/opencv/basic/challenge-03-lane-road-detection)
[[2]](https://github.com/Sentdex/pygta5/blob/master/Tutorial%20Codes/Part%201-7/part-5-line-finding.py)


