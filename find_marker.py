import math
from typing import List, Tuple, Any
import numpy as np
import os
import time

# 定义常量
MAXN = 30   # 网格最大行数（允许的最大 N）
MAXM = 30   # 网格最大列数（允许的最大 M）
MAXNM = MAXN * MAXM  # 最大点数上限 N*M
PI = 3.14159265

# 类型别名：只是为了类型提示更清晰
vd = List[float]
vvd = List[vd]


class Point_t:
    """简单的 2D 点结构，记录 x, y 像素坐标以及原始索引 id。"""

    def __init__(self, x: float = 0, y: float = 0, id: int = 0):
        self.x = x  # 像素 x 坐标
        self.y = y  # 像素 y 坐标
        self.id = id  # 该点在输入列表中的原始索引

    def __lt__(self, other: 'Point_t') -> bool:
        """用于排序：先按 x 升序，再按 y 升序。"""
        if self.x < other.x:
            return True
        if self.x == other.x and self.y < other.y:
            return True
        return False

    def __sub__(self, other: 'Point_t') -> 'Point_t':
        """向量减法：返回 self - other，表示从 other 到 self 的位移向量。"""
        return Point_t(self.x - other.x, self.y - other.y)

    def __add__(self, other: 'Point_t') -> 'Point_t':
        """向量加法：两个点的坐标相加。"""
        return Point_t(self.x + other.x, self.y + other.y)

    def __truediv__(self, other: float) -> 'Point_t':
        """标量除法：坐标分别除以一个标量，常用于取平均位移。"""
        return Point_t(self.x / other, self.y / other)


class Matching:
    """将散乱的检测点与规则网格进行匹配的核心类。

    整体思路：
    - 已知一个 N×M 的规则网格 O（未变形时的理论位置）；
    - 每一帧检测得到若干散乱点 C（marker 实际位置）；
    - 需要找到一个匹配方案：将 C 中的点尽可能地映射到 O 上的某些网格点，
      同时允许：
        * 有些网格点没有对应的观测点（缺失）;
        * 有些观测点是冗余的（多余）。

    算法特点：
    - 使用深度优先搜索 (DFS) 结合代价函数进行最优匹配搜索；
    - 代价由两部分组成：
        1) 单个点的位移大小 (K1 * |flow|^2)；
        2) 相邻点之间流场差异 (K2 * |flow_i - flow_j|^2)，保证流场平滑；
    - 对于没有观测点的网格位置，通过周围邻居的流场进行插值推断 (infer)。

    参数
    ------
    N_, M_ : int
        网格行数和列数。
    fps_ : int
        允许单帧搜索的最大时间，单位为“帧率”，即每帧搜索时长约为 1/fps 秒。
    x0_, y0_ : float
        左上角网格点在图像中的初始坐标。
    dx_, dy_ : float
        水平/垂直方向相邻网格点之间的间距（像素）。
    """

    def __init__(self, N_: int = 8, M_: int = 8, fps_: int = 30,
                 x0_: float = 80.0, y0_: float = 15.0, dx_: float = 21.0, dy_: float = 21.0,
                 preset_grid_path   : str = None, method: str = 'dfs'):
        # 网格参数
        self.x_0 = x0_  # 左上角 x
        self.y_0 = y0_  # 左上角 y
        self.dx = dx_   # 列方向步长（水平间距）
        self.dy = dy_   # 行方向步长（垂直间距）

        self.N = N_     # 网格行数
        self.M = M_     # 网格列数
        self.NM = N_ * M_  # 网格点总数
        self.fps = fps_    # DFS 时间剪枝所用的帧率参数

        self.method = method.lower()

        # 初始化参考网格
        self.O = [[Point_t() for _ in range(M_)] for _ in range(N_)]

        # === 加载标定网格 (对两种方法都至关重要) ===
        loaded_preset = False
        if preset_grid_path and os.path.exists(preset_grid_path):
            try:
                grid_data = np.load(preset_grid_path)
                if grid_data.shape[0] == self.NM:
                    print(f"[Matching] Loaded preset grid: {preset_grid_path} | Method: {self.method.upper()}")
                    idx = 0
                    for i in range(N_):
                        for j in range(M_):
                            self.O[i][j].x = float(grid_data[idx][0])
                            self.O[i][j].y = float(grid_data[idx][1])
                            idx += 1
                    loaded_preset = True
            except Exception as e:
                print(f"[Matching] Error loading preset: {e}")

        self.using_preset_grid = loaded_preset

        if not loaded_preset:
            print("[Matching] Warning: Using rectangular grid fallback.")
            for i in range(N_):
                for j in range(M_):
                    self.O[i][j].x = x0_ + j * dx_
                    self.O[i][j].y = y0_ + i * dy_

        # flag_record 控制是否在第一帧结束后把 O 更新为当前匹配结果
        # 这样后续帧的“参考网格”会随着材料变形逐帧更新，跟踪更稳定
        self.flag_record = 1

        # 一些基于 dx 的距离/阈值参数，控制候选前驱和流场平滑程度
        self.dmin = (dx_ * 0.5) ** 2           # 点间最小距离平方（用于 precessor）
        self.dmax = (dx_ * 1.8) ** 2           # 点间最大距离平方（用于 precessor）
        self.theta = 70.0                      # 允许的最大角度差（度）
        self.moving_max = dx_ * 2              # 单点允许的最大移动距离
        self.flow_difference_threshold = dx_ * 0.8  # 相邻流场允许的最大差异
        # 匹配总代价上限，过大时直接剪枝；用 dx 归一化，兼容不同网格尺度
        self.cost_threshold = 15000 * (dx_ / 21) ** 2

        # Row[i], Col[i] 记录第 i 个观测点 C[i] 被放到网格的哪一行哪一列
        self.Row = [-1] * MAXNM
        self.Col = [-1] * MAXNM

        # Dist[i][j] 记录 C[i] 与 C[j] 之间的距离平方（只在 i>j 时使用）
        self.Dist = [[0.0 for _ in range(MAXNM)] for _ in range(MAXNM)]

        # done[row] 标记某一行是否已经被占用作为“新行”的起点
        self.done = [0] * MAXN
        # occupied[i][j] 记录网格点(i, j)对应的观测点索引：
        #   -1: 目前无人占用
        #   >=0: 该索引对应的 C[...] 占据
        #   -2: 该网格点被认为是“缺失点”（没有匹配的真实观测点）
        self.occupied = [[-1 for _ in range(MAXM)] for _ in range(MAXN)]

        # first[row] 记录该行第一个匹配点的 y 坐标，用于保持各行之间的上下顺序约束
        self.first = [0] * MAXN

        # degree[i][j] 存储点 i、j 相对的“倾斜角”度数，用于 precessor 中的角度筛选
        self.degree = [[0.0 for _ in range(MAXNM)] for _ in range(MAXNM)]

        # DFS 起始时间，用于控制搜索时间不超过 1/fps
        self.time_st = 0.0
        self.current_frame_id = None
        self.current_run_timed_out = False
        self.timeout_frame_ids = []

        # n: 当前帧检测到的观测点数量
        self.n = 0
        # MinRow/MinCol: 记录当前找到的最优解中，每个点 C[i] 对应的网格位置
        self.MinRow = [-1] * MAXNM
        self.MinCol = [-1] * MAXNM
        # MinOccupied: 记录最优解下的网格占据情况
        self.MinOccupied = [[-1 for _ in range(MAXM)] for _ in range(MAXN)]
        # minf: 当前搜索到的最小总代价
        self.minf = -1.0

        # D[i][j]: 某一网格点(i,j)在当前搜索分支下被推断/匹配到的位置
        self.D = [[Point_t() for _ in range(MAXM)] for _ in range(MAXN)]
        # C: 存放所有观测点的数组（排序后）
        self.C = [Point_t() for _ in range(MAXNM)]
        # MinD: 最优解时每个网格点的最终位置（C 点或推断出的 D 点）
        self.MinD = [[Point_t() for _ in range(MAXM)] for _ in range(MAXN)]

        # K1, K2 为代价函数中位移项与流场平滑项的权重系数
        self.K1 = 0.1
        self.K2 = 1.0

    def init(self, centers: vvd) -> None:
        """初始化当前帧的观测点列表 C，并预计算距离和角度表。

        参数
        ------
        centers : list[list[float]]
            每个元素为 [x, y]，表示当前帧检测到的一个 marker 坐标。
        """
        self.n = len(centers)

        # 将输入的 (x, y) 拷贝到内部的 C 数组中
        for i in range(self.n):
            self.C[i].x = centers[i][0]
            self.C[i].y = centers[i][1]
            self.C[i].id = i

        # 每一帧开始前重置状态
        self.done = [0] * MAXN
        self.occupied = [[-1 for _ in range(MAXM)] for _ in range(MAXN)]
        self.minf = -1.0

        # 将前 n 个点排序（按 x, 再按 y），便于后续“前驱”关系的判断
        self.C[:self.n] = sorted(self.C[:self.n])

        # 预计算任意两点之间的距离平方 Dist 和相对角度 degree
        for i in range(self.n):
            for j in range(i):
                self.Dist[i][j] = self.dist_sqr(self.C[i], self.C[j])
                if self.Dist[i][j] == 0:
                    self.degree[i][j] = 0.0
                else:
                    # 这里的 degree 实际上是两点连线与水平线的夹角（取绝对值），单位度
                    self.degree[i][j] = math.asin(
                        abs(self.C[i].y - self.C[j].y) / math.sqrt(self.Dist[i][j])) * 180.0 / PI

    def set_current_frame_id(self, frame_id: int) -> None:
        self.current_frame_id = int(frame_id)

    def get_timeout_frame_ids(self) -> List[int]:
        return list(self.timeout_frame_ids)

    def did_current_run_timeout(self) -> bool:
        return self.current_run_timed_out

    @staticmethod
    def dist_sqr(a: Point_t, b: Point_t) -> float:
        """返回两点间距离的平方，避免频繁开平方。"""
        return (a.x - b.x) ** 2 + (a.y - b.y) ** 2

    @staticmethod
    def sum(a: Point_t) -> float:
        """返回向量 a 的平方模长： |a|^2 = ax^2 + ay^2。"""
        return a.x ** 2 + a.y ** 2

    def precessor(self, i: int, j: int) -> bool:
        """判断点 C[j] 是否可以作为 C[i] 的“前驱点”。

        这里的含义是：若认为在网格中第 i 个点紧挨着第 j 个点（水平相邻），
        则两点之间的距离和倾斜角要落在合理范围内：
        - dmin <= |Ci - Cj|^2 <= dmax
        - -theta <= 角度 <= theta
        """
        return (self.theta >= self.degree[i][j] >= -self.theta
                and self.dmax >= self.Dist[i][j] >= self.dmin)

    def calc_cost(self, i: int) -> float:
        """计算将第 i 个观测点 C[i] 放到当前网格位置 (Row[i], Col[i]) 的增量代价。

        代价包括：
        1) 该点自身位移大小：K1 * |flow_i|^2；
        2) 与左、上、下相邻网格点的流场差异：K2 * |flow_i - flow_neighbor|^2，
           若流场差异过大（>= flow_difference_threshold），则对该对邻居施加巨额惩罚（1e8），
           以避免产生畸变过大的局部匹配方案。
        """
        cost = 0.0
        c = 0.0
        left = -1
        up = -1
        down = -1

        # 当前点的流：从理论位置 O[row][col] 到观测位置 C[i]
        flow1 = self.C[i] - self.O[self.Row[i]][self.Col[i]]
        cost += self.K1 * self.sum(flow1)

        # 检查左侧点（同一行，前一列）
        if self.Col[i] > 0:
            left = self.occupied[self.Row[i]][self.Col[i] - 1]
            if left > -1:
                flow2 = self.C[left] - self.O[self.Row[i]][self.Col[i] - 1]
                c = self.sum(flow2 - flow1)
                # 若流场差异过大，则认为这种匹配几乎不可能，给一个极大惩罚
                if math.sqrt(c) >= self.flow_difference_threshold:
                    c = 1e8
                cost += self.K2 * c

        # 检查上方点
        if self.Row[i] > 0:
            up = self.occupied[self.Row[i] - 1][self.Col[i]]
            if up > -1:
                flow2 = self.C[up] - self.O[self.Row[i] - 1][self.Col[i]]
                c = self.sum(flow2 - flow1)
                if math.sqrt(c) >= self.flow_difference_threshold:
                    c = 1e8
                cost += self.K2 * c

        # 检查下方点
        if self.Row[i] < self.N - 1:
            down = self.occupied[self.Row[i] + 1][self.Col[i]]
            if down > -1:
                flow2 = self.C[down] - self.O[self.Row[i] + 1][self.Col[i]]
                c = self.sum(flow2 - flow1)
                if math.sqrt(c) >= self.flow_difference_threshold:
                    c = 1e8
                cost += self.K2 * c

        return cost

    def infer(self) -> float:
        """对所有当前未被观测点占据的网格点进行“流场外推”，并计算相应代价。

        主要做两件事：
        1) 对 occupied[i][j] <= -1 的网格点，利用其邻居(及扩展邻居)的流场，
           推断该点的位移 D[i][j] - O[i][j]；
        2) 保证推断得到的流场在空间上平滑：
           - 每个空点与周围邻居的流差平方之和加入总代价；
           - 若左/右边界上的点整体偏移方向异常，则加上边界惩罚项。
        """
        cost = 0.0
        boarder_nb = 0
        # 四邻方向：左、上、右、下
        dirs = [[0, -1], [-1, 0], [0, 1], [1, 0]]

        # 第一步：根据周围已匹配点的流场，推断每个“空网格点”的流
        for i in range(self.N):
            for j in range(self.M):
                if self.occupied[i][j] <= -1:  # 该网格点目前没有真实观测点
                    moving = Point_t(0, 0)  # 流场累加器
                    cnt = 0                 # 参与平均的邻居数量

                    # 1) 先看四个方向的“二阶”邻居，类似外推：flow_nx_ny + (flow_nx_ny - flow_nnx_nny)
                    for k in range(4):
                        nx = i + dirs[k][0]
                        ny = j + dirs[k][1]
                        nnx = nx + dirs[k][0]
                        nny = ny + dirs[k][1]

                        if 0 <= nnx < self.N and 0 <= nny < self.M:
                            if self.occupied[nx][ny] > -1 and self.occupied[nnx][nny] > -1:
                                flow_nx_ny = self.C[self.occupied[nx][ny]] - self.O[nx][ny]
                                flow_nnx_nny = self.C[self.occupied[nnx][nny]] - self.O[nnx][nny]
                                # 线性外推：flow(ny) + (flow(ny) - flow(nny))
                                moving += flow_nx_ny + (flow_nx_ny - flow_nnx_nny)
                                cnt += 1

                    # 2) 若上一步没有找到足够邻居，则在 1×1 邻域内简单平均
                    if cnt == 0:
                        d = 1
                        for x in range(i - d, i + d + 1):
                            for y in range(j - d, j + d + 1):
                                if 0 <= x < self.N and 0 <= y < self.M and self.occupied[x][y] > -1:
                                    moving += self.C[self.occupied[x][y]] - self.O[x][y]
                                    cnt += 1

                    # 3) 若还不够，再在 2×2 范围内做平均
                    if cnt == 0:
                        d = 2
                        for x in range(i - d, i + d + 1):
                            for y in range(j - d, j + d + 1):
                                if 0 <= x < self.N and 0 <= y < self.M and self.occupied[x][y] > -1:
                                    moving += self.C[self.occupied[x][y]] - self.O[x][y]
                                    cnt += 1

                    # 最终推断位置 D[i][j] = O[i][j] + 平均流
                    self.D[i][j] = self.O[i][j] + (moving / (cnt + 1e-6) if cnt != 0 else moving)

                    # 边界约束：如果左/右边界的推断位移方向“不合理”，统计 boarder_nb
                    if j == 0 and self.D[i][j].y >= self.O[i][j].y - self.dy / 2.0:
                        boarder_nb += 1
                    if j == self.N - 1 and self.D[i][j].y <= self.O[i][j].y + self.dy / 2.0:
                        boarder_nb += 1

                    # 该空点自身的位移代价
                    cost += self.K1 * self.sum(self.D[i][j] - self.O[i][j])

        # 若边界上存在大量“反常行为”，则添加一个巨大的惩罚，避免这种解被选中
        if boarder_nb >= self.N - 1:
            cost += 1e7

        # 第二步：对所有空网格点，与其邻居的流场差异也要计入代价，保持整体平滑
        for i in range(self.N):
            for j in range(self.M):
                if self.occupied[i][j] <= -1:
                    flow1 = self.D[i][j] - self.O[i][j]
                    for k in range(4):
                        nx = i + dirs[k][0]
                        ny = j + dirs[k][1]
                        if 0 <= nx < self.N and 0 <= ny < self.M:
                            # 邻居是已匹配的真实点
                            if self.occupied[nx][ny] > -1:
                                flow2 = self.C[self.occupied[nx][ny]] - self.O[nx][ny]
                                cost += self.K2 * self.sum(flow2 - flow1)
                            # 邻居也是推断点（仅对左/上方向 k<2 限制，避免重复计算）
                            elif k < 2 and self.occupied[nx][ny] <= -1:
                                flow2 = self.D[nx][ny] - self.O[nx][ny]
                                cost += self.K2 * self.sum(flow2 - flow1)

        return cost

    def dfs(self, i: int, cost: float, missing: int, spare: int) -> None:
        """深度优先搜索所有可能的匹配方案，并维护最小代价解。

        参数
        ------
        i : int
            当前正在处理的观测点下标（在排序后的 C 中的索引）。
        cost : float
            到目前为止，已经累积的代价。
        missing : int
            尚允许设置为“缺失”的网格点数量（空位数目）。
        spare : int
            尚允许舍弃（不匹配到网格）的观测点数量。

        递归过程中对于每个点 C[i]，会尝试几种情况：
        1) 作为某个已知点 C[j] 的横向前驱（同一行，列+1）；
        2) 作为新行的第一个点（Row[i] = 某新行, Col[i] = 0）；
        3) 插入若干缺失网格点后，再放到某行的下一个位置；
        4) 将该点视为多余点（spare > 0 时）。

        通过大量剪枝：
        - 时间限制（不超过 1/fps 秒）；
        - 当前代价已超过已有最优解或全局阈值 cost_threshold；
        来控制搜索复杂度。
        """
        # 1) 时间剪枝：若超过 1/fps 秒，直接返回，避免单帧卡死
        if self.fps > 0:
            current_time = time.time()
            if (current_time - self.time_st) >= 1.0 / self.fps:
                if not self.current_run_timed_out:
                    self.current_run_timed_out = True
                    if self.current_frame_id is not None and self.current_frame_id not in self.timeout_frame_ids:
                        self.timeout_frame_ids.append(self.current_frame_id)
                return

        # 2) 代价剪枝：若当前代价已经不优，则不再深入
        if self.minf != -1 and cost >= self.minf:
            return
        if cost >= self.cost_threshold:
            return

        # 3) 递归终止：所有点都已处理完
        if i >= self.n:
            # 对剩余空网格点进行推断，并加入代价
            total_cost = cost + self.infer()
            if self.minf == -1 or total_cost < self.minf:
                # 更新当前最优解
                self.minf = total_cost
                # 记录每个观测点在网格中的行列
                for j in range(self.n):
                    self.MinRow[j] = self.Row[j]
                    self.MinCol[j] = self.Col[j]
                    if self.Row[j] >= 0:
                        # 将 D 填入真实观测点的位置
                        self.D[self.Row[j]][self.Col[j]].x = self.C[j].x
                        self.D[self.Row[j]][self.Col[j]].y = self.C[j].y

                # 记录网格占据状态和 D 的最终值
                for j in range(self.N):
                    for k in range(self.M):
                        self.MinOccupied[j][k] = self.occupied[j][k]
                        self.MinD[j][k].x = self.D[j][k].x
                        self.MinD[j][k].y = self.D[j][k].y
            return

        # =============== 情况 1：尝试“前驱”匹配（水平相邻） ===============
        count = 0
        for j in range(i):
            if self.precessor(i, j):  # C[j] 可以作为 C[i] 的前驱
                self.Row[i] = self.Row[j]
                self.Col[i] = self.Col[j] + 1
                count += 1

                # 列边界检查
                if self.Col[i] >= self.M:
                    continue
                # 该网格位置是否已被其他点占据
                if self.occupied[self.Row[i]][self.Col[i]] > -1:
                    continue

                # 上下行顺序合理性检查
                valid = True
                if self.Row[i] > 0 and self.occupied[self.Row[i] - 1][self.Col[i]] > -1:
                    if self.C[i].y <= self.C[self.occupied[self.Row[i] - 1][self.Col[i]]].y:
                        valid = False
                if self.Row[i] < self.N - 1 and self.occupied[self.Row[i] + 1][self.Col[i]] > -1:
                    if self.C[i].y >= self.C[self.occupied[self.Row[i] + 1][self.Col[i]]].y:
                        valid = False
                if not valid:
                    continue

                # 同一列纵向顺序检查：保证上方的点 y 更小，下方的点 y 更大
                vflag = 0
                for k in range(self.N):
                    same_col = self.occupied[k][self.Col[i]]
                    if same_col > -1:
                        if (k < self.Row[i] and self.C[same_col].y > self.C[i].y) or \
                                (k > self.Row[i] and self.C[same_col].y < self.C[i].y):
                            vflag = 1
                            break
                if vflag == 1:
                    continue

                # 尝试将 C[i] 放入该网格位置，并继续 DFS
                self.occupied[self.Row[i]][self.Col[i]] = i
                new_cost = cost + self.calc_cost(i)
                self.dfs(i + 1, new_cost, missing, spare)
                # 回溯：恢复占据状态
                self.occupied[self.Row[i]][self.Col[i]] = -1

        # =============== 情况 2：作为“新行”的第一个点 ===============
        for j in range(self.N):
            if self.done[j] == 0:  # 该行尚未被使用
                flag = 0
                # 检查与已有行的上下顺序是否冲突
                for k in range(self.N):
                    if self.done[k]:
                        if (k < j and self.first[k] > self.C[i].y) or \
                                (k > j and self.first[k] < self.C[i].y):
                            flag = 1
                            break
                if flag == 1:
                    continue

                # 标记该行为已使用，并记录该行第一个点的 y 值
                self.done[j] = 1
                self.first[j] = self.C[i].y
                self.Row[i] = j
                self.Col[i] = 0
                self.occupied[j][0] = i

                new_cost = cost + self.calc_cost(i)
                self.dfs(i + 1, new_cost, missing, spare)

                # 回溯，恢复状态
                self.done[j] = 0
                self.occupied[j][0] = -1

        # =============== 情况 3：插入若干缺失点后再放置当前点 ===============
        for m in range(1, missing + 1):  # 一次可以插入 1..missing 个缺失点
            for j in range(self.N):
                # 限制当前点在该行的垂直移动范围
                if abs(self.C[i].y - self.O[j][0].y) > self.moving_max:
                    continue

                # 找到该行当前最后一个被占据的列 k
                k = self.M - 1
                while k >= 0 and self.occupied[j][k] == -1:
                    k -= 1

                # 检查插入缺失点后，是否还在列范围内
                if k + m + 1 >= self.M:
                    continue

                # 检查当前点移动到新列后的欧式距离是否过大
                if math.sqrt(self.sum(self.C[i] - self.O[j][k + m + 1])) > self.moving_max:
                    continue

                # 将中间的 m 个网格点标记为“-2：缺失点”
                for t in range(1, m + 1):
                    self.occupied[j][k + t] = -2

                # 将当前点放到第 k+m+1 列
                self.Row[i] = j
                self.Col[i] = k + m + 1
                self.occupied[j][self.Col[i]] = i

                new_cost = cost + self.calc_cost(i)
                self.dfs(i + 1, new_cost, missing - m, spare)

                # 回溯：清空刚才填的缺失点与当前点
                for t in range(1, m + 2):
                    self.occupied[j][k + t] = -1

        # =============== 情况 4：将该观测点视为冗余点（不放入网格） ===============
        if spare > 0:
            self.Row[i] = -1
            self.Col[i] = -1
            self.dfs(i + 1, cost, missing, spare - 1)

    def run(self) -> None:
        """对当前帧进行匹配搜索并更新内部最优解。

        步骤：
        1) 根据当前观测点个数 n 与网格点个数 N*M，计算：
           - missing = max(0, N*M - n)  允许的“缺失网格点”数量；
           - spare   = max(0, n - N*M)  允许的“冗余观测点”数量；
        2) 从 i=0 开始执行 DFS 搜索，寻找代价最小的匹配方案；
        3) 若第一次搜索没有找到解（minf 依然为 -1），多尝试几次，
           并适度放宽 missing / spare 上限；
        4) 若 flag_record==1（通常只在第一帧），
           将当前最优匹配的观测点位置写回 O，作为下一帧的参考网格。"""
        # 记录本帧搜索开始的时间，用于时间剪枝
        self.time_st = time.time()
        self.current_run_timed_out = False
        # 缺失网格点数 = 网格点 - 观测点（若为负则为 0）
        missing = max(0, self.N * self.M - self.n)
        # 冗余观测点数 = 观测点 - 网格点（若为负则为 0）
        spare = max(0, self.n - self.N * self.M)

        # 首次搜索
        self.dfs(0, 0.0, missing, spare)

        # 多次尝试：若一次都没找到可行解（minf 仍为 -1），
        # 则适度放宽缺失点和冗余点的数量，再重新 DFS
        for t in range(1, 4):
            if self.minf == -1:
                # 重置状态
                self.done = [0] * MAXN
                self.occupied = [[-1 for _ in range(MAXM)] for _ in range(MAXN)]
                self.dfs(0, 0.0, missing + 1, spare + 1)

        # 背景帧 frame_id == 0 匹配成功后，将该帧匹配到的 marker 位置写回参考网格 O。
        # 这样即使初始使用了 preset grid，后续跟踪也会以真实背景帧匹配结果作为参考。
        if self.flag_record == 1 and self.minf != -1 and self.current_frame_id == 0:
            self.flag_record = 0
            for i in range(self.n):
                if self.MinRow[i] >= 0 and self.MinCol[i] >= 0:
                    self.O[self.MinRow[i]][self.MinCol[i]].x = self.C[i].x
                    self.O[self.MinRow[i]][self.MinCol[i]].y = self.C[i].y

    def get_flow(self) -> Tuple[vvd, vvd, vvd, vvd, vvd, vvd, vvd, vvd]:
        """将当前最优匹配方案转换为位移场形式并返回。

        返回
        ------
        Ox, Oy : 2D list[float]
            每个网格点原始参考位置 O 的 x, y 坐标。
        Cx, Cy : 2D list[float]
            每个网格点在当前帧的匹配/推断后的位置（MinD）的 x, y 坐标。
        Occupied : 2D list[float]
            网格占据状态（来自 MinOccupied）：
                -1 : 空网格点
                -2 : 推断为缺失网格点
                >=0: 对应的观测点索引
        dx, dy : 2D list[float]
            每个网格点的位移分量：
                dx = Cx - Ox  （向右为正）
                dy = Oy - Cy  （向上为正，与图像 y 轴向下相反）
        distance : 2D list[float]
            每个网格点的位移模长 sqrt(dx^2 + dy^2)。
        """
        Ox: vvd = [[0.0 for _ in range(self.M)] for _ in range(self.N)]
        Oy: vvd = [[0.0 for _ in range(self.M)] for _ in range(self.N)]
        Cx: vvd = [[0.0 for _ in range(self.M)] for _ in range(self.N)]
        Cy: vvd = [[0.0 for _ in range(self.M)] for _ in range(self.N)]
        dx: vvd = [[0.0 for _ in range(self.M)] for _ in range(self.N)]
        dy: vvd = [[0.0 for _ in range(self.M)] for _ in range(self.N)]
        distance: vvd = [[0.0 for _ in range(self.M)] for _ in range(self.N)]
        Occupied: vvd = [[0.0 for _ in range(self.M)] for _ in range(self.N)]

        for i in range(self.N):
            for j in range(self.M):
                Ox[i][j] = self.O[i][j].x
                Oy[i][j] = self.O[i][j].y
                Cx[i][j] = self.MinD[i][j].x
                Cy[i][j] = self.MinD[i][j].y
                dx[i][j] = Cx[i][j] - Ox[i][j]  # 向右为正
                dy[i][j] = Oy[i][j] - Cy[i][j]  # 向上为正（图像坐标系 y 向下）
                distance[i][j] = np.sqrt(dx[i][j] ** 2 + dy[i][j] ** 2)
                Occupied[i][j] = self.MinOccupied[i][j]

        return (Ox, Oy, Cx, Cy, Occupied, dx, dy, distance)

    def test(self) -> Tuple[float, float]:
        """简单返回当前网格的步长 dx, dy，用于调试。"""
        return (self.dx, self.dy)
