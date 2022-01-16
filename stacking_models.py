import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

_N_FOLDS = 5  # 采用5折交叉验证
kf = KFold(n_splits=_N_FOLDS, shuffle=True, random_state=42)  # sklearn的交叉验证模块，用于划分数据


def get_oof(clf, X_train, y_train, X_test):
    # X_train: 1000 * 10
    # y_train: 1 * 1000
    # X_test : 500 * 10
    oof_train = np.zeros((X_train.shape[0], 1))  # 1000 * 1  Stacking后训练数据的输出
    oof_test_skf = np.empty((_N_FOLDS, X_test.shape[0], 1))  # 5 * 500 * 1，oof_test_skf[i]代表第i折交叉验证产生的模型对测试集预测结果

    for i, (train_index, test_index) in enumerate(kf.split(X_train)): # 交叉验证划分此时的训练集和验证集
        kf_X_train = X_train[train_index]  # 800 * 10 训练集
        kf_y_train = y_train[train_index]  # 1 * 800 训练集对应的输出
        kf_X_val = X_train[test_index]  # 200 * 10  验证集

        clf.fit(kf_X_train, kf_y_train)  # 当前模型进行训练

        oof_train[test_index] = clf.predict(kf_X_val).reshape(-1, 1)  # 对当前验证集进行预测， 200 * 1
        oof_test_skf[i, :] = clf.predict(X_test).reshape(-1, 1)  # 对测试集预测 oof_test_skf[i, :] : 500 * 1

    oof_test = oof_test_skf.mean(axis=0)  # 对每一则交叉验证的结果取平均
    return oof_train, oof_test  # 返回当前分类器对训练集和测试集的预测结果


# 将数据换成你的数据
X_train = np.random.random((1000, 10))  # 1000 * 10
y_train = np.random.random_integers(0, 1, (1000,))  # 1000
X_test = np.random.random((500, 10))  # 500 * 10

# 将你的每个分类器都调用get_oof函数，并把它们的结果合并，就得到了新的训练和测试数据new_train,new_test
new_train, new_test = [], []
for clf in [LinearRegression(), RandomForestRegressor()]:
    oof_train, oof_test = get_oof(clf, X_train, y_train, X_test)
    new_train.append(oof_train)
    new_test.append(oof_test)

new_train = np.concatenate(new_train, axis=1)
new_test = np.concatenate(new_test, axis=1)

# 用新的训练数据new_train作为新的模型的输入，stacking第二层
clf = RandomForestRegressor()
clf.fit(new_train, y_train)
clf.predict(new_test)