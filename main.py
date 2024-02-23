import random
import math
import copy
import time
import bezier
import matplotlib.pyplot as plt
import numpy as np
import bezier
from quad_program import quad_program_for_perching as QPP
import sympy as sp
from matplotlib.patches import Circle
from numpy.polynomial.polynomial import Polynomial
import scipy.io as scio
import matplotlib.patches as patches
class RRT:

    # 初始化
    def __init__(self,
                 obstacle_list,         # 障碍物
                 rand_area,             # 采样的区域
                 expand_dis=2.0,        # 步长
                 goal_sample_rate=30,   # 目标采样率
                 max_iter=200):         # 最大迭代次数

        self.start = None
        self.goal = None
        self.min_rand = rand_area[0]
        self.max_rand = rand_area[1]
        self.expand_dis = expand_dis
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.obstacle_list = obstacle_list
        self.node_list = None

    def calculate_angle(self, node1, node2, node3):
        # 计算向量
        vector1 = [node2.x - node1.x, node2.y - node1.y]
        vector2 = [node2.x - node3.x, node2.y - node3.y]

        # 向量点积
        dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]

        # 向量长度
        norm1 = math.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
        norm2 = math.sqrt(vector2[0] ** 2 + vector2[1] ** 2)

        # 避免除以零
        if norm1 * norm2 == 0:
            return 0

        # 保证 acos 的输入在 [-1, 1] 的范围内
        cos_angle = dot_product / (norm1 * norm2)
        cos_angle = max(min(cos_angle, 1.0), -1.0)

        # 计算角度
        angle = math.acos(cos_angle)

        # 转换为度
        angle = math.degrees(angle)

        return angle
        
    def get_final_course(self, last_index):
        """ 回溯路径 """
        path = [[self.goal.x, self.goal.y]]
        while self.node_list[last_index].parent is not None:
            node = self.node_list[last_index]
            path.append([node.x, node.y])
            last_index = node.parent
        path.append([self.start.x, self.start.y])
        return path
    def get_final_course_bezier(self):

        path = self.get_path()
        if path is None or len(path) < 3:
            return None

        bezier_paths = []

        # 第一个线段的处理
        # [第一个线段的处理方式与之前相同]
        mid_point1 = [(path[0][0] + path[1][0]) / 2, (path[0][1] + path[1][1]) / 2]
        one_fourth_point2 = [path[1][0] * 0.75 + path[2][0] * 0.25,
                             path[1][1] * 0.75 + path[2][1] * 0.25]
        mid_point2 = [(path[1][0] + path[2][0]) / 2, (path[1][1] + path[2][1]) / 2]

        control_points_first = [path[0], mid_point1, path[1], one_fourth_point2, mid_point2]
        nodes_first = np.asfortranarray(control_points_first).T
        curve_first = bezier.Curve(nodes_first, degree=4)
        bezier_first = curve_first.evaluate_multi(np.linspace(0.0, 1.0, 1000)).T.tolist()
        bezier_paths.extend(bezier_first)
        # 对于中间的线段
        for i in range(1, len(path) - 3):
            mid_point1 = [(path[i][0] + path[i + 1][0]) / 2, (path[i][1] + path[i + 1][1]) / 2]
            mid_point2 = [(path[i + 1][0] + path[i + 2][0]) / 2, (path[i + 1][1] + path[i + 2][1]) / 2]

            # 计算 3/4 点和 1/4 点
            three_fourth_point = [path[i][0] * 0.25 + path[i + 1][0] * 0.75,
                                  path[i][1] * 0.25 + path[i + 1][1] * 0.75]
            one_fourth_point = [path[i + 1][0] * 0.75 + path[i + 2][0] * 0.25,
                                path[i + 1][1] * 0.75 + path[i + 2][1] * 0.25]

            # 为当前段创建贝塞尔曲线的控制点
            control_points = [mid_point1, three_fourth_point, path[i + 1], one_fourth_point, mid_point2]
            nodes = np.asfortranarray(control_points).T

            # 创建贝塞尔曲线
            curve = bezier.Curve(nodes, degree=4)
            s_vals = np.linspace(0.0, 1.0, 1000)
            bezier_segment = curve.evaluate_multi(s_vals).T.tolist()
            bezier_paths.extend(bezier_segment)

        # 处理最后两个线段
        mid_point_last_2 = [(path[-3][0] + path[-2][0]) / 2, (path[-3][1] + path[-2][1]) / 2]
        three_fourth_point_last_2 = [path[-2][0] * 0.25 + path[-3][0] * 0.75,
                                     path[-2][1] * 0.25 + path[-3][1] * 0.75]
        mid_point_last = [(path[-2][0] + path[-1][0]) / 2, (path[-2][1] + path[-1][1]) / 2]

        control_points_last = [mid_point_last_2, three_fourth_point_last_2, path[-2], mid_point_last, path[-1]]
        nodes_last = np.asfortranarray(control_points_last).T
        curve_last = bezier.Curve(nodes_last, degree=4)
        bezier_last = curve_last.evaluate_multi(np.linspace(0.0, 1.0, 1000)).T.tolist()
        bezier_paths.extend(bezier_last)

        return bezier_paths

    '''def draw_graph(self, rnd=None, path=None):
        # ... [existing code to draw nodes, obstacles, etc.] ...'''
    def get_curve_count(self):
        path = self.get_path()
        get_curve_count = len(path) -1
        return get_curve_count
        # Draw Bezier curve path
    def get_curve_list(self):

        path = self.get_path()
        if path is None or len(path) < 3:
            return None

        bezier_curves = []

        # 第一个线段的处理
        # [第一个线段的处理方式与之前相同]
        mid_point1 = [(path[0][0] + path[1][0]) / 2, (path[0][1] + path[1][1]) / 2]
        one_fourth_point2 = [path[1][0] * 0.75 + path[2][0] * 0.25,
                             path[1][1] * 0.75 + path[2][1] * 0.25]
        mid_point2 = [(path[1][0] + path[2][0]) / 2, (path[1][1] + path[2][1]) / 2]

        control_points = [path[0], mid_point1, path[1], one_fourth_point2, mid_point2]
        nodes = np.asfortranarray(control_points).T
        curve = bezier.Curve(nodes, degree=4)
        bezier_curves.append(curve)
        # 对于中间的线段
        for i in range(1, len(path) - 3):
            mid_point1 = [(path[i][0] + path[i + 1][0]) / 2, (path[i][1] + path[i + 1][1]) / 2]
            mid_point2 = [(path[i + 1][0] + path[i + 2][0]) / 2, (path[i + 1][1] + path[i + 2][1]) / 2]

            # 计算 3/4 点和 1/4 点
            three_fourth_point = [path[i][0] * 0.25 + path[i + 1][0] * 0.75,
                                  path[i][1] * 0.25 + path[i + 1][1] * 0.75]
            one_fourth_point = [path[i + 1][0] * 0.75 + path[i + 2][0] * 0.25,
                                path[i + 1][1] * 0.75 + path[i + 2][1] * 0.25]

            # 为当前段创建贝塞尔曲线的控制点
            control_points = [mid_point1, three_fourth_point, path[i + 1], one_fourth_point, mid_point2]
            nodes = np.asfortranarray(control_points).T

            # 创建贝塞尔曲线
            curve = bezier.Curve(nodes, degree=4)
            s_vals = np.linspace(0.0, 1.0, 1000)
            bezier_segment = curve.evaluate_multi(s_vals).T.tolist()
            bezier_curves.append(curve)

        # 处理最后两个线段
        mid_point_last_2 = [(path[-3][0] + path[-2][0]) / 2, (path[-3][1] + path[-2][1]) / 2]
        three_fourth_point_last_2 = [path[-2][0] * 0.25 + path[-3][0] * 0.75,
                                     path[-2][1] * 0.25 + path[-3][1] * 0.75]
        mid_point_last = [(path[-2][0] + path[-1][0]) / 2, (path[-2][1] + path[-1][1]) / 2]

        control_points_last = [mid_point_last_2, three_fourth_point_last_2, path[-2], mid_point_last, path[-1]]
        nodes_last = np.asfortranarray(control_points_last).T
        curve_last = bezier.Curve(nodes_last, degree=4)
        bezier_curves.append(curve_last)

        return bezier_curves

    def get_bezier_count(self):
        path = self.get_path()
        get_bezier_count = len(path)-3
        return get_bezier_count
    
    def rrt_planning(self, start, goal, animation=True):
        self.start = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])
        self.node_list = [self.start]
        path = None
        a = True
        for i in range(self.max_iter):
                # 1. 在环境中随机采样点
            rnd = self.sample()

                # 2. 找到结点树中距离采样点最近的结点
            n_ind = self.get_nearest_list_index(self.node_list, rnd)
            nearest_node = self.node_list[n_ind]

                # 3. 在采样点的方向生长一个步长，得到下一个树的结点。
            theta = math.atan2(rnd[1] - nearest_node.y, rnd[0] - nearest_node.x)
            new_node = self.get_new_node(theta, n_ind, nearest_node)
            if len(self.node_list) > 1 and nearest_node.parent is not None:
                    # 获取前一个节点和前前一个节点
                pre_node = nearest_node
                if pre_node.parent is not None:
                     pre_pre_node = self.node_list[pre_node.parent]

                        # 计算夹角
                     angle = self.calculate_angle(pre_pre_node, pre_node, new_node)

                    # 如果夹角小于120度，则跳过这个节点
                if angle > 120:
                    a = True
                else:
                    a = False
                # 4. 检测碰撞，检测到新生成的结点的路径是否会与障碍物碰撞
            no_collision = self.check_segment_collision(new_node.x, new_node.y, nearest_node.x, nearest_node.y)
            if no_collision and a:
                self.node_list.append(new_node)

                    # 一步一绘制
               # if animation:
                   # time.sleep(1)
                    #self.draw_graph(new_node, path)

                    # 判断新结点是否临近目标点
                if self.is_near_goal(new_node):
                    if self.is_near_goal(new_node):
                        pre_pre_node = nearest_node
                        pre_node = new_node

                        angle = self.calculate_angle(pre_pre_node, pre_node, self.goal)
                        if angle > 90:
                            a = True
                        else:
                            a = False
                    if self.check_segment_collision(new_node.x, new_node.y,
                                                    self.goal.x, self.goal.y) and a :
                        last_index = len(self.node_list) - 1
                        path = self.get_final_course(last_index)  # 回溯路径
                        path1 = self.get_final_course_bezier()

                        #if animation:
                           # self.draw_graph(new_node, path1)
                        return path


    def get_point_on_bezier_curve(self,t,curve):
        """
        Get a point on the Bezier curve for a given t.

        :param t: A value between 0 and 1 to specify the point on the Bezier curve.
        :return: The coordinates of the point on the curve.
        """
        point=curve.evaluate(t)

        return point
    def sample(self):
        """ 在环境中采样点的函数，以一定的概率采样目标点 """
        if random.randint(0, 100) > self.goal_sample_rate:
            rnd = [random.uniform(self.min_rand, self.max_rand),
                   random.uniform(self.min_rand, self.max_rand)]
        else:
            rnd = [self.goal.x, self.goal.y]
        return rnd

    @staticmethod
    def get_nearest_list_index(nodes, rnd):
        """ 计算树中距离采样点距离最近的结点 """
        d_list = [(node.x - rnd[0]) ** 2 + (node.y - rnd[1]) ** 2
                  for node in nodes]
        min_index = d_list.index(min(d_list))
        return min_index

    def get_new_node(self, theta, n_ind, nearest_node):
        """ 计算新结点 """
        new_node = copy.deepcopy(nearest_node)

        new_node.x += self.expand_dis * math.cos(theta)
        new_node.y += self.expand_dis * math.sin(theta)

        new_node.cost += self.expand_dis
        new_node.parent = n_ind

        return new_node

    def check_segment_collision(self, x1, y1, x2, y2):
        """ 检测碰撞 """
        if self.obstacle_list is None:
            return True
        else :
            for (ox, oy, radius) in self.obstacle_list:
                dd = self.distance_squared_point_to_segment(
                    np.array([x1, y1]),
                    np.array([x2, y2]),
                    np.array([ox, oy])
                )
                if dd <= radius ** 2:
                    return False
        return True

    @staticmethod
    def distance_squared_point_to_segment(v, w, p):
        """ 计算线段 vw 和 点 p 之间的最短距离"""
        if np.array_equal(v, w):    # 点 v 和 点 w 重合的情况
            return (p - v).dot(p - v)

        l2 = (w - v).dot(w - v)     # 线段 vw 长度的平方
        t = max(0, min(1, (p - v).dot(w - v) / l2))
        projection = v + t * (w - v)
        return (p - projection).dot(p - projection)

    def draw_graph(self, rnd=None, path=None):

        plt.clf()

        # 绘制新的结点
        if rnd is not None:
            plt.plot(rnd.x, rnd.y, '^k')

        # 绘制路径
        for node in self.node_list:
            if node.parent is not None:
                if node.x or node.y is not None:
                    plt.plot([node.x, self.node_list[node.parent].x],
                             [node.y, self.node_list[node.parent].y],
                             '-g')

        # 绘制起点、终点
        plt.plot(self.start.x, self.start.y, "og")
        plt.plot(self.goal.x, self.goal.y, "or")

        # 绘制障碍物
        if self.obstacle_list is None:
            pass
        else:
            for (ox, oy, size) in self.obstacle_list:
                plt.plot(ox, oy, "ok", ms=30 * size)

        # 绘制路径'''

        if path is not None:
            bezier_x, bezier_y = zip(*path)
            plt.plot(bezier_x, bezier_y, '-r')

        # 绘制图的设置
        plt.axis([-2, 18, -2, 15])
        plt.grid(True)
        plt.pause(0.01)

    def is_near_goal(self, node):
        d = self.line_cost(node, self.goal)
        if d < self.expand_dis:
            return True
        return False

    def calculate_bezier_tangents(self, t,curve):
        """
        Calculate the tangents (unit vectors) of the Bezier curve at given points.

        :param num_points: Number of points along the curve to calculate tangents.
        :return: A list of tangent unit vectors at each point.
        """
        # 获取路径并创建贝塞尔曲线


        # 计算曲线上各点的切线
        tangent = []
            # 计算曲线在 t 处的切线（导数）
        point = curve.evaluate(t)

        tangent = curve.evaluate_hodograph(t).reshape(-1)
            # 标准化为单位向量
        tangent_unit = tangent / np.linalg.norm(tangent, axis=0)
        
        return tangent_unit


    @staticmethod
    def line_cost(node1, node2):
        return math.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2)

    '''def get_final_course(self, last_index):
        """ 回溯路径 """
        path = [[self.goal.x, self.goal.y]]
        while self.node_list[last_index].parent is not None:
            node = self.node_list[last_index]
            path.append([node.x, node.y])
            last_index = node.parent
        path.append([self.start.x, self.start.y])
        return path'''



    # 示例：对于 bezier_curves 中的每条曲线计算 t 与 l 的关系

    #def precompute

    def  precompute_bezier_lengths(self,curve):

        #Precompute the cumulative lengths of the Bezier curve for different values of t.

        self.bezier_lengths = []
        prev_point = curve.evaluate(0.0)
        cum_length = 0.0
        for t in np.linspace(0.0, 1.0, 1000):  # Adjust the resolution as needed
            point = curve.evaluate(t)
            cum_length += np.linalg.norm(point - prev_point)
            self.bezier_lengths.append((t, cum_length))

            prev_point = point
        t_values = np.array([tpl[0] for tpl in self.bezier_lengths])
        length_values = np.array([tpl[1] for tpl in self.bezier_lengths])

        # 使用 numpy 的 polyfit 进行多项式拟合，度数可调整
        coefs = np.polynomial.polynomial.polyfit(length_values, t_values, 5)
        fitted_poly = Polynomial(coefs)

        # 返回拟合的多项式对象
        return fitted_poly

    def calculate_bezier_curve_length(self,curve):
        """
        Calculate the total length of the Bezier curve.
        """
        t_values = np.linspace(0, 1, 1000)
        points = curve.evaluate_multi(t_values)
        distances = np.sqrt(np.sum(np.diff(points, axis=1) ** 2, axis=0))
        length = np.sum(distances)
        return length

    def get_curve_length_intervals(self,bezier_curves):

        length_intervals = []
        total_length = 0

        for curve in bezier_curves:
            length = self.calculate_bezier_curve_length(curve)
            length_intervals.append((total_length, total_length + length))
            total_length += length

        return length_intervals

    def get_curve_total_length(self,bezier_curves):
        if bezier_curves is None:
            return 0
        """
        计算贝塞尔曲线列表中每段曲线的长度区间。

        :param bezier_curves: 贝塞尔曲线对象的列表。
        :return: 每段曲线的长度区间的列表。
        """
        length_intervals = []
        total_length = 0

        for curve in bezier_curves:
            length = self.calculate_bezier_curve_length(curve)
            length_intervals.append((total_length, total_length + length))
            total_length += length

        return total_length
    def get_path(self):
        """
        Get the list of points that form the path from start to goal.
        """
        if self.node_list is None:
            return None

        path = [[self.goal.x, self.goal.y]]
        last_index = len(self.node_list) - 1
        while self.node_list[last_index].parent is not None:
            node = self.node_list[last_index]
            path.append([node.x, node.y])
            last_index = node.parent
        path.append([self.start.x, self.start.y])
        path.reverse()
        return path

    def get_bezier_curve_points(self, num_points=100):
        path = self.get_path()
        if path is None or len(path) < 3:
            return None

        bezier_paths = []

        # 第一个线段的处理
        # [第一个线段的处理方式与之前相同]
        mid_point1 = [(path[0][0] + path[1][0]) / 2, (path[0][1] + path[1][1]) / 2]
        one_fourth_point2 = [path[1][0] * 0.75 + path[2][0] * 0.25,
                             path[1][1] * 0.75 + path[2][1] * 0.25]
        mid_point2 = [(path[1][0] + path[2][0]) / 2, (path[1][1] + path[2][1]) / 2]

        control_points_first = [path[0], mid_point1, path[1], one_fourth_point2, mid_point2]
        nodes_first = np.asfortranarray(control_points_first).T
        curve_first = bezier.Curve(nodes_first, degree=4)
        bezier_first = curve_first.evaluate_multi(np.linspace(0.0, 1.0, num_points)).T.tolist()
        bezier_paths.extend(bezier_first)
        # 对于中间的线段
        for i in range(1, len(path) - 3):
            mid_point1 = [(path[i][0] + path[i + 1][0]) / 2, (path[i][1] + path[i + 1][1]) / 2]
            mid_point2 = [(path[i + 1][0] + path[i + 2][0]) / 2, (path[i + 1][1] + path[i + 2][1]) / 2]

            # 计算 3/4 点和 1/4 点
            three_fourth_point = [path[i][0] * 0.25 + path[i + 1][0] * 0.75,
                                  path[i][1] * 0.25 + path[i + 1][1] * 0.75]
            one_fourth_point = [path[i + 1][0] * 0.75 + path[i + 2][0] * 0.25,
                                path[i + 1][1] * 0.75 + path[i + 2][1] * 0.25]

            # 为当前段创建贝塞尔曲线的控制点
            control_points = [mid_point1, three_fourth_point, path[i + 1], one_fourth_point, mid_point2]
            nodes = np.asfortranarray(control_points).T

            # 创建贝塞尔曲线
            curve = bezier.Curve(nodes, degree=4)
            s_vals = np.linspace(0.0, 1.0, num_points)
            bezier_segment = curve.evaluate_multi(s_vals).T.tolist()
            bezier_paths.extend(bezier_segment)

        # 处理最后两个线段
        mid_point_last_2 = [(path[-3][0] + path[-2][0]) / 2, (path[-3][1] + path[-2][1]) / 2]
        three_fourth_point_last_2 = [path[-2][0] * 0.25 + path[-3][0] * 0.75,
                                     path[-2][1] * 0.25 + path[-3][1] * 0.75]
        mid_point_last = [(path[-2][0] + path[-1][0]) / 2, (path[-2][1] + path[-1][1]) / 2]

        control_points_last = [mid_point_last_2, three_fourth_point_last_2, path[-2], mid_point_last, path[-1]]
        nodes_last = np.asfortranarray(control_points_last).T
        curve_last = bezier.Curve(nodes_last, degree=4)
        bezier_last = curve_last.evaluate_multi(np.linspace(0.0, 1.0, num_points)).T.tolist()
        bezier_paths.extend(bezier_last)

        return bezier_paths
    '''def get_bezier_curve_points(self, num_points=100):
        """
        根据 RRT 路径点计算并返回贝塞尔曲线的路径点。

        :param num_points: 要计算的贝塞尔曲线上的点数。
        :return: 贝塞尔曲线上的点列表。
        """
        rrt_path = self.get_path()
        if rrt_path is None:
            return None

        # 转换为贝塞尔曲线的控制点格式
        nodes = np.asfortranarray(rrt_path).T

        # 创建贝塞尔曲线对象
        curve = bezier.Curve(nodes, degree=len(rrt_path) - 1)

        # 计算贝塞尔曲线上的点
        bezier_path = []
        for t in np.linspace(0.0, 1.0, num_points):
            point = curve.evaluate(t)
            bezier_path.append(point.T[0])
        bezier_path.reverse()
        return bezier_path'''


    def evaluate_bezier_curve(self,control_points,t):
        """
        Evaluate a point on a Bezier curve for a given parameter t.

        :param control_points: A list of control points for the Bezier curve.
                               Each control point should be a tuple (x, y).
        :param t: The parameter value (between 0 and 1) for which to evaluate the curve.
        :return: The (x, y) coordinates of the point on the Bezier curve.
        """
        n = len(control_points)
        temp_points = list(control_points)

        # Apply De Casteljau's Algorithm
        for r in range(1, n):
            for i in range(n - r):
                temp_points[i] = (1 - t) * np.array(temp_points[i]) + t * np.array(temp_points[i + 1])

        return temp_points[0]
    @staticmethod
    def get_path_len(path):
        """ 计算路径的长度 """
        path_length = 0
        for i in range(1, len(path)):
            node1_x = path[i][0]
            node1_y = path[i][1]
            node2_x = path[i - 1][0]
            node2_y = path[i - 1][1]
            path_length += math.sqrt((node1_x - node2_x) ** 2 + (node1_y - node2_y) ** 2)
        return path_length

    def polynomial_derivative(self,g_t):
        """
        计算多项式 g(t) 关于 t 的导数。

        :param coefficients: 多项式 g(t) 的系数，按照幂次递减的顺序排列。
        :return: 表示 g(t) 导数的符号表达式。
        """
        t = sp.symbols('t')
        
        g_t_sympy = sum(coef * t ** i for i, coef in enumerate(reversed(g_t.coefficients)))

        # Now you can use the polynomial_derivative function

        g_t_derivative = sp.diff(g_t_sympy, t)
        return g_t_derivative

    def bezier_derivatives(self, t_value):
        """
        计算贝塞尔曲线在给定 t 值处的一阶和二阶导数（速度和加速度向量）。

        :param t_value: 参数 t 的值，范围在 0 到 1 之间。
        :return: 速度向量和加速度向量。
        """
        if self.node_list is None:
            return None, None

        path = [[self.goal.x, self.goal.y]]
        last_index = len(self.node_list) - 1
        while self.node_list[last_index].parent is not None:
            node = self.node_list[last_index]
            path.append([node.x, node.y])
            last_index = node.parent
        path.append([self.start.x, self.start.y])

        path = np.array(path).T  # Transpose to match the input format for bezier
        nodes = np.asfortranarray(path)

        # 定义符号变量 t
        t = sp.symbols('t')

        # 创建贝塞尔曲线并计算其一阶和二阶导数
        self.curve = bezier.Curve(nodes, degree=len(path[0]) - 1)
        velocity = self.curve.evaluate_hodograph(t_value)

        #acceleration = curve.evaluate_hodograph(t_value, derivative=2)

        return velocity.flatten()
    



    def plot_path(self, start_t, end_t, num_points=100):

        t_values = np.linspace(start_t, end_t, num_points)
        x_values = []
        y_values = []
        tangents = []

        for t in t_values:
            p_d_x, p_d_y = self.return_p_d_x_d_y(t)
            v_d_x, v_d_y = self.return_v_d_x_d_y(t)
            x_values.append(p_d_x)
            y_values.append(p_d_y)
            tangents.append((v_d_x, v_d_y))

        plt.figure(figsize=(8, 6))
        plt.plot(x_values, y_values, label='Path', color='blue', marker='o')

        # 绘制每个点的单位切向量
        for i in range(len(tangents)):
            plt.quiver(x_values[i], y_values[i], float(tangents[i][0]), float(tangents[i][1]),
                       angles='xy', scale_units='xy', scale=1, color='red')
        print(type(tangents[0][0]))
        plt.title('Path with Tangents Visualization')
        plt.xlabel('p_d_x')
        plt.ylabel('p_d_y')
        plt.legend()
        plt.grid(True)
        plt.show()

    def get_velocity_components(self, start_t, end_t, num_points=100):
        """
        计算并返回在给定的t值范围内的速度分量。

        :param start_t: t的起始值。
        :param end_t: t的结束值。
        :param num_points: 在start_t和end_t之间计算的点数。
        :return: 两个数组，分别包含v_d_x和v_d_y的值。
        """
        t_values = np.linspace(start_t, end_t, num_points)
        v_d_x_values = []
        v_d_y_values = []

        for t in t_values:
            v_d_x, v_d_y = self.return_v_d_x_d_y(t)
            v_d_x_values.append(v_d_x)
            v_d_y_values.append(v_d_y)

        return v_d_x_values, v_d_y_values

    def find_length_from_interval_start(self,l, length_intervals):
        """
        找出给定长度 l 所在的长度区间，并计算 l 与区间左边界的差值。

        :param l: 给定的长度。
        :param length_intervals: 贝塞尔曲线长度区间的列表。
        :return: l 与其所在区间左边界的差值，如果未找到对应区间则返回 None。
        """
        for start, end in length_intervals:
            if start <= l < end:
                return l - start
        return 0

    def create_obstacle_list(self,flag, p_d_x, p_d_y):
        obstacles = []  # Initialize an empty list for obstacles
        radius = 1.0  # You can set the radius as needed

        if flag == 1:
            # If flag is 1, add an obstacle at (p_d_x, p_d_y)
            obstacles.append((p_d_x, p_d_y, radius))


        return obstacles

    def find_interval_index(self,length, length_intervals):
        """
        在长度区间列表中找到给定长度所属的区间的索引。

        :param length: 要查找的长度。
        :param length_intervals: 每段曲线的长度区间列表。
        :return: 包含给定长度的区间的索引，如果没有找到，则返回 None。
        """
        for index, (start, end) in enumerate(length_intervals):
            if start <= length < end:
                return index
        return 0
    def plot_plane_trajectory(self,coordinates,obstacles):
        """
    绘制飞机轨迹的二维图。

    :param coordinates: numpy.ndarray, 每一行是一个时刻的 [x, y, z] 坐标。
    """
        plt.figure(figsize=(10, 6))

        # 提取 x 和 y 坐标
        x = coordinates[:, 0]
        y = coordinates[:, 1]

        # 绘制飞机轨迹
        plt.plot(x, y, marker='o', label='Trajectory')

        # 绘制障碍物
        for obs in obstacles:
            obs_x, obs_y, radius = obs
            obstacle = patches.Circle((obs_x, obs_y), radius, color='red', alpha=0.5)
            plt.gca().add_patch(obstacle)
            plt.plot(obs_x, obs_y, 'x', color='red')  # 标记障碍物中心

        # 设置图表标题和标签
        plt.title("Plane Trajectory in XY Plane with Obstacles")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.grid(True)
        plt.legend()
        plt.show()
    
    def return_p_d_x_d_y(self,t):
        b=self.g_t(t)
        
        
        ladex=self.find_interval_index(b,self.length_list)
        
        hanshu=self.precompute_bezier_lengths(self.bezier_curve_list[ladex])
        l = self.find_length_from_interval_start(b,self.length_list)
        hanshu=self.precompute_bezier_lengths(self.bezier_curve_list[ladex])
        t = hanshu(l)
        point = self.bezier_curve_list[ladex].evaluate(t)
        
        p_d_x=point[0]
        p_d_y=point[1]
        return (p_d_x,p_d_y)


    def return_v_d_x_d_y(self, t):
        b=self.g_t(t)
        
        
        ladex=self.find_interval_index(b,self.length_list)
        
        hanshu=self.precompute_bezier_lengths(self.bezier_curve_list[ladex])
        l = self.find_length_from_interval_start(b,self.length_list)
        hanshu=self.precompute_bezier_lengths(self.bezier_curve_list[ladex])
        t_value = hanshu(l)
        
        g_derivative_t = self.polynomial_derivative(self.g_t)
        v_shuru = g_derivative_t.subs(sp.symbols('t'), t).evalf()
        tangent = self.calculate_bezier_tangents(t_value,self.bezier_curve_list[ladex])
        v_d_x = v_shuru*tangent[0]
        v_d_y = v_shuru * tangent[1]

        return (v_d_x,v_d_y)
    
    def cal_xishu(self):
        self.bezier_curve_list = self.get_curve_list()
        print (self.bezier_curve_list)
        total_length = self.get_curve_total_length(self.bezier_curve_list)
        print(total_length)
        total_time = total_length/0.5
        print(total_time)
        qpp1 = QPP(7, total_time);
    #          ↑  ↑
    #        阶数 模态自变量，初态默认为0
        Coef = [1, 1, 1, 1]  # vel, acc, jerk, snap
        qpp1.set_coeff(Coef[0], Coef[1], Coef[2], Coef[3])
        qpp1.add_pos_constraint_1_by_1(0., 0)  # 位置约束
        qpp1.add_pos_constraint_1_by_1(total_time, total_length)

        qpp1.add_vel_constraint_1_by_1(0., 0.3)  # 速度约束
        qpp1.add_vel_constraint_1_by_1(total_time, 0.5)
        x_traj_coefficient = np.mat(qpp1.opt())
        #print(x_traj_coefficient)
        array = np.array(x_traj_coefficient)
    # print(array.shape)
        array_transpose = array.reshape(1, -1)
        # print((array.ndim))
    # print(array_transpose.ndim)
    # print(array_transpose)
        b = array_transpose.flatten()
    # print((b.ndim))
       # print(np.shape(b))
        self.g_t = np.poly1d(b, variable='t')
        self.length_list = self.get_curve_length_intervals(self.bezier_curve_list)
        return total_time
class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.cost = 0.0
        self.parent = None
