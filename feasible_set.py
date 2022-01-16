import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class feasible:
    def __init__(self,n) -> None:
        self.n=n
        self.rho4 = np.array([[1.0, -0.2, -0.5, 0.0],
                         [-0.2, 1.0, 0.0, 0.1],
                         [-0.5, 0.0, 1.0, -0.3],
                         [0.0, 0.1, -0.3, 1.0]])  # 输入相关系数矩阵
        self.e4 = np.array([0.065, 0.1, 0.08, 0.09])  # 输入期望收益向量
        self.sigma4 = np.array([0.15, 0.3, 0.26, 0.21])  # 输入标准差向量

    def covariance_maxtirc(self,sigma, rho):  # 函数作用：由标准差、相关系数矩阵生成协方差矩阵
        V = []
        for i in range(len(sigma)):
            V.append([])
            for j in range(len(sigma)):
                V[i].append(rho[i, j] * sigma[j] * sigma[i])
        return V


    def is_pos_def(self,V):  # 函数作用：判断协方差矩阵是否是正定矩阵
        if np.all(np.linalg.eigvals(V) > 0):
            return 1
        else:
            print("您的协方差矩阵非正定！请重新输入！")
            return 0


    def feasible_set(self,e, sigma, V, n):  # 生成可行集
        E1 = []
        S1 = []
        E2 = []
        S2 = []
        #有卖空限制：生成权重向量并计算组合的期望收益和标准差
        for i in range(n):
            w0 = np.random.random(len(e))
            wa = w0/sum(w0)
            E1.append(np.dot(e, np.transpose(wa)))  # dot是矩阵乘法
            S1.append(np.sqrt(np.dot(np.dot(wa, V), np.transpose(wa))))
        #无卖空限制：生成权重向量并计算组合的期望收益和标准差

        for i in range(n):
            w1 = -1 + 2 * np.random.random(len(e))
            wb = w1/sum(w1)
            E2.append(np.dot(e, np.transpose(wb)))
            if np.dot(np.dot(wb, V), np.transpose(wb)) <= 0:
                print(np.dot(np.dot(wb, V), np.transpose(wb)), wb, sum(wb))
            S2.append(np.sqrt(np.dot(np.dot(wb, V), np.transpose(wb))))

        ##########画图##########################
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        # matplotlib画图中中文显示会有问题，需要这两行设置默认字体
        plt.xlabel('sigma')
        plt.ylabel('E(r)')
        plt.xlim(xmin=min(S1)-0.1, xmax=max(sigma)+0.1)
        plt.ylim(ymin=min(e)-0.05, ymax=max(e)+0.06)
        # 画两条（0-9）的坐标轴并设置轴标签x，y
        alpha1 = 0.4
        alpha0 = 0.4
        if len(e) == 2:
            colors1 = 'yellow'  # 点的颜色
            colors2 = 'darkorange'
            colors3 = 'yellowgreen'
            alpha0 = 0

        if len(e) == 3:
            colors1 = '#00CED1'  # 点的颜色
            colors2 = '#DC143C'
            colors3 = 'gold'
        if len(e) == 4:
            colors1 = 'darkviolet'  # 点的颜色
            colors2 = 'blue'
            colors3 = 'red'
        area = np.pi * 2 ** 2  # 点面积
        # 画散点图
        plt.scatter(S2, E2, s=area, c=colors2, alpha=alpha0, label='无卖空限制')
        plt.scatter(S1, E1, s=area, c=colors1, alpha=alpha1, label='有卖空限制')
        plt.scatter(sigma, e, marker='^', c=colors3, label='原始资产')
        plt.legend()
        plt.title("可行集")
        plt.savefig('./feasible_set.png', dpi=600, bbox_inches='tight')
        return 0


    def grade_down(self,e, sigma, rho):  # 将矩阵降级
        drho = []
        for i in range(len(e)-1):
            drho.append([])
            for j in range(len(e)-1):
                drho[i].append(rho[i, j])
        de = e[0:len(e)-1]
        dsigma = sigma[0:len(e)-1]
        return [de, dsigma, drho]


    def feasible_set_evolution(self,e4, sigma4, rho4):  # 生成可行集的另外一种方式
        [e3, sigma3, rho3] = self.grade_down(e4, sigma4, rho4)
        e3 = np.array(e3)
        sigma3 = np.array(sigma3)
        rho3 = np.array(rho3)
        [e2, sigma2, rho2] = self.grade_down(e3, sigma3, rho3)
        e2 = np.array(e2)
        sigma2 = np.array(sigma2)
        rho2 = np.array(rho2)
        V4 = self.covariance_maxtirc(sigma4, rho4)
        V3 = self.covariance_maxtirc(sigma3, rho3)
        V2 = self.covariance_maxtirc(sigma2, rho2)
        Bool4 = self.is_pos_def(V4)
        Bool3 = self.is_pos_def(V3)
        Bool2 = self.is_pos_def(V2)
        if Bool4 == 1:
            self.feasible_set(e4, sigma4, V4, self.n)
            self.feasible_set(e3, sigma3, V3, self.n)
            self.feasible_set(e2, sigma2, V2, self.n)
        plt.savefig('./feasible_set.png', dpi=600, bbox_inches='tight')
        plt.show()
        return 0


    ###############################主函数###############################
    ###########输入参数############
    print("同学您好，请您输入标准差和期望收益时不要太夸张")


    # #自定义计算
    # ###########计算方差协方差矩阵，#########
    # V4=covariance_maxtirc(sigma4,rho4)
    # ###########判断方差协方差是否正定#######
    # Bool4=is_pos_def(V4)
    # ########计算可行集###############
    # if Bool4==1:
    #     feasible_set(e4,sigma4,V4,n)
    # plt.show()

    #观看演化
    def run(self):
        self.feasible_set_evolution(self.e4, self.sigma4, self.rho4)

