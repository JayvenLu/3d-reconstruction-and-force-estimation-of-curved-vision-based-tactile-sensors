
def init():
    global RESCALE, N_, M_, x0_, y0_, dx_, dy_, fps_
    RESCALE = 1

    """
    N_, M_: the row and column of the marker array
    x0_, y0_: 左上角那个理论 marker 点在图像坐标中的位置（单位：像素），在原始尺寸下测得
    dx_, dy_: 相邻 marker 在横向（x 方向）和纵向（y 方向）的像素间距（同样是在原始图像尺寸下的值）
    fps_: the desired frame per second, the algorithm will find the optimal solution in 1/fps seconds
    """

    # 曲面三指1号
    N_ = 7
    M_ = 7
    fps_ = 10

    # 多光谱三指1号
    # N_ = 7
    # M_ = 10
    # fps_ = 20

    # 多光谱三指3号
    # N_ = 6
    # M_ = 8
    # fps_ = 10


    # # 平面三指
    # x0_ = 67 / RESCALE
    # y0_ = 63 / RESCALE
    # dx_ = 102.17 / RESCALE
    # dy_ = 73 / RESCALE

    # # 拾玥五指(全局图像)
    # x0_ = 216 / RESCALE
    # y0_ = 104 / RESCALE
    # dx_ = 50.43 / RESCALE
    # dy_ = 42.27 / RESCALE

    # # 拾玥五指(全局图像)
    # x0_ = 176 / RESCALE
    # y0_ = 128 / RESCALE
    # dx_ = 51 / RESCALE
    # dy_ = 43 / RESCALE

    # 曲面三指1号 透视校正后
    # x0_ = 134 / RESCALE
    # y0_ = 121 / RESCALE
    # dx_ = 68 / RESCALE
    # dy_ = 45 / RESCALE

    # 曲面三指2号 透视校正后
    x0_ = 143 / RESCALE
    y0_ = 146 / RESCALE
    dx_ = 63 / RESCALE
    dy_ = 41 / RESCALE

    # 多光谱曲面三指1号
    # x0_ = 598 / RESCALE
    # y0_ = 416 / RESCALE
    # dx_ = 78 / RESCALE
    # dy_ = 97 / RESCALE

    # 多光谱曲面三指3号
    # x0_ = 674 / RESCALE
    # y0_ = 532 / RESCALE
    # dx_ = 79 / RESCALE
    # dy_ = 80 / RESCALE

    # 多光谱曲面三指3号 透视校正后
    # x0_ = 458 / RESCALE
    # y0_ = 366 / RESCALE
    # dx_ = 160 / RESCALE
    # dy_ = 105 / RESCALE

    # x0_ = 416 / RESCALE
    # y0_ = 433 / RESCALE
    # dx_ = 152.98 / RESCALE
    # dy_ = 93.72 / RESCALE