import gc
import os
import sys
import time
import warnings
from functools import wraps

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Intel sklearn优化，非intel需要关闭此选项,同时循环运行可能会导致一定的内存泄漏，优化模型参数的时候需要关闭
# patch_sklearn()
import sklearn.cluster
import sklearn.metrics
import sklearn.svm
from catboost import CatBoostClassifier, Pool
from hyperopt import fmin, hp, tpe
from sklearnex import patch_sklearn
from tqdm import tqdm

warnings.filterwarnings("ignore")
# from memory_profiler import profile


# 计算运算时间的装饰器
def cal_time_cost(function):
    @wraps(function)
    def func(*args, **kwargs):
        print("-" * 80)
        print(f"Running : {function.__name__:25}")
        t0 = time.perf_counter()
        result = function(*args, **kwargs)
        t1 = time.perf_counter()
        print(f"Ends    : {function.__name__:25} cost: {t1-t0:.4f}s")
        return result

    return func


# 屏蔽stdout的装饰器
def blockstdout(func):
    def wrapper(*args, **kwargs):
        sys.stdout = open(os.devnull, "w")
        results = func(*args, **kwargs)
        sys.stdout = sys.__stdout__
        return results

    return wrapper


class BagOfVisualWords:
    def __init__(self) -> None:
        # 定义需要的变量
        self.imgs_train = []
        self.imgs_test = []
        self.label_train = []
        self.label_test = []
        self.kmeans = None
        self.rep_train = None
        self.rep_test = None
        self.save_dir = "./output"
        os.makedirs(self.save_dir, exist_ok=True)

        # 定义不需要重复的步骤
        # 读取数据
        self.read_15scene()
        # 提取sift特征 获得描述子，关键点和图像形状数组
        self.des_train, self.kp_train, self.shp_train = self.sift_extract(
            self.imgs_train
        )
        self.des_test, self.kp_test, self.shp_test = self.sift_extract(self.imgs_test)
        pass

    @cal_time_cost
    # @profile
    def read_15scene(self):
        dataset_path = "./data/15-Scene"  # 一共4485图片 2250 训练 2235 测试
        self.imgs_train = []
        self.imgs_test = []
        self.label_train = []
        self.label_test = []
        for root, _, files in os.walk(dataset_path):
            files.sort()
            for i in range(len(files)):  # TODO 检查排序是否符合预期
                img_path = os.path.join(root, files[i])
                img_label = os.path.basename(os.path.dirname(img_path))
                img_data = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if i < 150:
                    self.imgs_train.append(img_data)
                    self.label_train.append(int(img_label))
                else:
                    self.imgs_test.append(img_data)
                    self.label_test.append(int(img_label))

        self.label_train = np.array(self.label_train, dtype=np.int32)
        self.label_test = np.array(self.label_test, dtype=np.int32)
        return

    @cal_time_cost
    # @profile
    def sift_extract(self, imgs):
        """
        提取图像的sift特征 返回所有特征点的sift特征向量
        imgs 是图像列表
        """
        # 定义sift
        sift_des = []  # sift描述子
        kp_list = []  # 特征点列表
        shape_list = []  # 图像形状列表
        for i in tqdm(range(len(imgs)), desc="sift_extract"):
            sift = cv2.SIFT_create(
                nfeatures=0,  # 要保留的最佳特征的数量 0
                nOctaveLayers=3,  #
                contrastThreshold=0.04,  # 对比度滤除阈值，阈值越大，检测器产生的特征越少 0.04
                edgeThreshold=10,  # 用于过滤掉类似边缘的特征的阈值。边缘阈值越大，过滤掉的特征越少（保留的特征越多）10
            )
            kp, des = sift.detectAndCompute(imgs[i], None)
            kp_list.append(kp)  # keypoint对象列表 通过每个元素的pt属性求坐标
            sift_des.append(des)
            shape_list.append(imgs[i].shape)

        # TODO 抑制？
        return sift_des, kp_list, shape_list  # 返回(n,(numofkp,128))

    @cal_time_cost
    # @profile
    def get_visual_vocab(self, sifts, vocab_size=2000):
        """
        对全部描述子进行聚类，得到视觉特征词典
        """
        kmeans = sklearn.cluster.MiniBatchKMeans(
            n_clusters=vocab_size,
            random_state=42,
            batch_size=1024,
            verbose=0,
        )
        kmeans.fit(sifts)
        # 返回kmeans 可以调用predict进行te'zheng分类
        # 测试
        # print(kmeans.predict(sifts[:10]))
        return kmeans

    @cal_time_cost
    # @profile
    def get_img_representation(self, sift_des, kp_list, shape_list, vocab_size):
        """
        sift_des 图像特征描述向量列表 (n,(num_of_kp,128))
        kp_list 图像对应的关键点对象列表（通过pt求坐标）
        shape_list 图像形状list
        sift_label 图像sift标签
        """
        # kms 计算特征描述子所属的类别，计数类别并归一化
        img_representation = np.zeros((len(sift_des), 21 * vocab_size))  # 图像表示矩阵
        for img_id in range(len(sift_des)):
            assert len(sift_des[img_id]), len(kp_list[img_id])
            # 针对每个图像 计算图像描述向量
            img_x, img_y = shape_list[img_id]  # 图像大小
            delta_x, delta_y = np.ceil(img_x / 4), np.ceil(img_y / 4)  # 网格、

            l2 = np.zeros((16, vocab_size))
            # 另一种写法
            kpts = [kp_list[img_id][i].pt for i in range(len(kp_list[img_id]))]
            kpts = np.array(kpts)  # n,2 # 关键点坐标
            grim_ids = [
                int(np.floor(kpts[i][1] / delta_x) + np.floor(kpts[i][0] / delta_y) * 4)
                for i in range(
                    len(kpts)
                )  # 列是反的 https://blog.csdn.net/luoyang7891/article/details/106472505
            ]  # 属于哪个词
            cat_ids = self.kmeans.predict(sift_des[img_id])  # 预测属于哪个词袋
            for i in range(len(kpts)):
                try:
                    l2[grim_ids[i], cat_ids[i]] += 1
                except IndexError:
                    print("IndexError")
                    print(img_x, img_y)
                    print(delta_x, delta_y)
                    print(i)
                    print(kpts[i][0], kpts[i][1])
                    exit()

            l1 = np.zeros((4, vocab_size))
            for grim_id, i in enumerate([0, 2, 8, 10]):
                l1[grim_id] = l2[i] + l2[i + 1] + l2[i + 4] + l2[i + 5]

            l0 = np.zeros((1, vocab_size))
            l0[0] += np.sum(l1, axis=0)

            # 将不同的层次表示拼接起来
            # 权重 https://www.cnblogs.com/lxjshuju/p/7300861.html 1/2^(L-l)
            l1 = l1 / 2
            l0 = l0 / 4
            img_representation[img_id] = np.concatenate(
                [l0.flatten(), l1.flatten(), l2.flatten()]
            )

        # img_label = np.array(sift_label)
        return img_representation  # (dataset_size,21*vocab_size)

    @cal_time_cost
    # @profile
    def train(self, C=100, kernel="rbf"):
        # 实现训练和
        # 支持向量机
        # https://scikit-learn.org/stable/modules/svm.html#multi-class-classification
        # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
        svc = sklearn.svm.SVC(
            kernel=kernel,
            C=C,
            decision_function_shape="ovr",
        )
        # svc = sklearn.svm.SVC(kernel="rbf", C=1000, decision_function_shape="ovr")
        svc.fit(self.rep_train, self.label_train)
        # sklearn.metrics.confusion_matrix()
        return svc

    @cal_time_cost
    # @profile
    def evaluate(self):
        # 通过得到的模型进行评估
        assert self.classfier is not None
        pred = self.classfier.predict(self.rep_test)
        pred = np.squeeze(pred)  #

        metrics = {
            "precision": sklearn.metrics.precision_score(
                self.label_test, pred, average="weighted"
            ),
            "recall": sklearn.metrics.recall_score(
                self.label_test, pred, average="weighted"
            ),
            "f1": sklearn.metrics.f1_score(self.label_test, pred, average="weighted"),
            "accuracy": sklearn.metrics.accuracy_score(self.label_test, pred),
        }
        print(metrics)
        return pred, metrics

    # @profile
    def pipeline(
        self, vocab_size=2000, C=100, kernel="rbf", cm_label="precision", plt_cm=True
    ):
        # 写入整体训练重复流程，用于优化参数
        # 计算视觉词典
        self.classfier = None
        self.rep_train = None
        self.rep_test = None
        sift_all = np.concatenate(self.des_train + self.des_test, axis=0)
        self.kmeans = self.get_visual_vocab(sifts=sift_all, vocab_size=vocab_size)
        del sift_all
        # 计算图像表示
        self.rep_train = self.get_img_representation(
            self.des_train, self.kp_train, self.shp_train, vocab_size=vocab_size
        )
        self.rep_test = self.get_img_representation(
            self.des_test, self.kp_test, self.shp_test, vocab_size=vocab_size
        )
        self.classfier = self.train(C=C, kernel=kernel)
        self.pred_test, self.metrics = self.evaluate()
        if plt_cm:
            self.plt_confusion_matrix(
                text=f"{cm_label}_{vocab_size}_{C}_{kernel}"
            )  # 混淆矩阵
        # 释放内存
        del self.kmeans, self.classfier, self.rep_train, self.rep_test
        gc.collect()
        return self.metrics  # precison,recall,f1,accuracy

    @cal_time_cost
    def plt_confusion_matrix(self, text=""):
        # cfmx = sklearn.metrics.confusion_matrix(
        #     self.label_test, self.test_pred, normalize="true"
        # )
        plt.rcParams.update({"font.size": 6})
        sklearn.metrics.ConfusionMatrixDisplay.from_predictions(
            self.label_test,
            self.pred_test,
            values_format=".3g",
            normalize="true",
        )
        # plt.text()

        plt.show()
        plt.savefig(
            os.path.join(self.save_dir, f"confusion_matrix_{text}.png"), dpi=300
        )
        pass

    # @profile
    def optimize(self, metric="precision"):
        assert metric in ["precision", "recall", "f1", "accuracy"]

        space = {
            "vocab_size": hp.randint("vocab_size", 90, 1000),
            "C": hp.uniform("C", 0, 20),
            "kernel": hp.choice("kernel", ["sigmoid", "rbf"]),
        }

        @blockstdout
        def obj(params):
            m = self.pipeline(
                params["vocab_size"],
                params["C"],
                params["kernel"],
                cm_label=metric,
                plt_cm=False,
            )
            return -m[metric]

        best = fmin(
            fn=obj,
            space=space,
            algo=tpe.suggest,
            max_evals=100,
        )
        print(best)

    # =============================================================================

    @cal_time_cost
    def train_catboost(
        self,
        iterations=100,
        lr=0.01,
        depth=6,
        l2_leaf_reg=3,
    ):
        m = CatBoostClassifier(
            iterations=iterations,
            learning_rate=lr,
            depth=depth,
            l2_leaf_reg=l2_leaf_reg,
            task_type="CPU",
            # early_stopping_rounds=5,
            logging_level="Silent",
        )
        train_pool = Pool(self.rep_train, label=self.label_train)
        m.fit(train_pool)
        return m

    def gbdt_pipeline(
        self,
        vocab_size=2000,
        iterations=100,
        lr=0.01,
        depth=6,
        l2_leaf_reg=3,
        cm_label="precision",
        plt_cm=True,
    ):
        # 写入整体训练重复流程，用于优化参数
        # 计算视觉词典
        self.classfier = None
        self.rep_train = None
        self.rep_test = None
        sift_all = np.concatenate(self.des_train + self.des_test, axis=0)
        self.kmeans = self.get_visual_vocab(sifts=sift_all, vocab_size=vocab_size)
        del sift_all
        # 计算图像表示
        self.rep_train = self.get_img_representation(
            self.des_train, self.kp_train, self.shp_train, vocab_size=vocab_size
        )
        self.rep_test = self.get_img_representation(
            self.des_test, self.kp_test, self.shp_test, vocab_size=vocab_size
        )
        self.classfier = self.train_catboost(
            iterations=iterations,
            lr=lr,
            depth=depth,
            l2_leaf_reg=l2_leaf_reg,
        )
        self.rep_test = Pool(self.rep_test, self.label_test)
        self.pred_test, self.metrics = self.evaluate()
        if plt_cm:
            self.plt_confusion_matrix(text=f"{cm_label}_CatBoost")  # 混淆矩阵
        # 释放内存
        del self.kmeans, self.classfier, self.rep_train, self.rep_test
        gc.collect()
        return self.metrics  # precison,recall,f1,accuracy

    def gbdt_optimize(self, metric="precision"):
        assert metric in ["precision", "recall", "f1", "accuracy"]

        space = {
            "vocab_size": hp.randint("vocab_size", 200, 600),
            "lr": hp.uniform("lr", 0.005, 0.2),
            "depth": hp.randint("depth", 2, 10),
            "l2_leaf_reg": hp.randint("l2_leaf_reg", 2, 12),
            # "kernel": hp.choice("kernel", ["sigmoid", "rbf"]),
        }

        @blockstdout
        def obj(params):
            m = self.gbdt_pipeline(
                vocab_size=params["vocab_size"],
                lr=params["lr"],
                depth=params["depth"],
                l2_leaf_reg=params["l2_leaf_reg"],
                cm_label=metric,
                plt_cm=False,
            )
            return -m[metric]

        best = fmin(
            fn=obj,
            space=space,
            algo=tpe.suggest,
            max_evals=150,
        )
        print(best)


if __name__ == "__main__":
    model = BagOfVisualWords()
    model.optimize("f1")  # precision,recall,f1,accuracy

    # model.pipeline(vocab_size=344, C=12.2, kernel="rbf")  # precision最佳
    # model.pipeline(
    #     vocab_size=513, C=11.2, kernel="rbf", cm_label="accuracy"
    # )  # accuracy最佳
    # for i in range(50):
    #     model.pipeline()

    # model.gbdt_pipeline()
    # model.gbdt_optimize("precision")

    pass