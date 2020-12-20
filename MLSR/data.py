import pandas as pd
from numpy import random


class DataSet:
    """
    Class for data set manipulation
    """

    def __init__(self, filename: str = '17_19_Data.csv', strong_file=None, weak_file=None, encode='gbk'):
        if weak_file is None:
            self.data = pd.read_csv(filename, encoding=encode)
            self.data = self.data.apply(lambda x: x.replace("\n", ""))
            self.weak_data = None
            self.strong_data = None
        else:
            self.weak_data = pd.read_csv(weak_file, encoding=encode)
            self.strong_data = pd.read_csv(strong_file, encoding=encode)

    @staticmethod
    def special_init(data: pd.DataFrame) -> pd.DataFrame:
        """
        我们使用的数据集中的特殊处理
        一共有16个特征，发现第1772个数据在原数据集里列填串了，第8297个数据无标签，删去
        院系、专业、民族、出生年月、校区没有用，删掉
        :param data:
        :return:
        """
        data = data[data.columns[:16]]
        data.drop(labels=[1772, 8297], inplace=True)
        data.drop(['院系', '专业', '出生年月', '所在校区'], axis=1, inplace=True)
        return data

    def shuffle_and_pick(self, out_path: str = 'rand_select') -> tuple:
        """
        随机打乱后随机抽取的400个样本，用于人工再标注细化标签
        :param out_path:
        :return:
        """
        self.data["rand"] = random.uniform(0, 1, len(self.data))
        self.data.sort_values(by="rand", inplace=True)  # 对data随机排序
        self.data.drop('rand', axis=1, inplace=True)
        self.weak_data = self.data[:400]
        self.weak_data.to_csv(out_path+'_weak.csv', encoding="utf-8")
        self.strong_data.to_csv(out_path + '_strong.csv', encoding="utf-8")
        self.strong_data = self.data[400:]
        return self.weak_data, self.strong_data

    @staticmethod
    def do_nation_policy(d:pd.Series):
        d["享受国家政策资助情况"] = d["享受国家政策资助情况"].fillna('无')

        d["建档立卡贫困户"] = ''
        index = d["享受国家政策资助情况"].str.contains("立卡")
        d.loc[index, "建档立卡贫困户"] = 1
        data.loc[-index, "建档立卡贫困户"] = 0

        data["城乡低保户"] = ''
        index = data["享受国家政策资助情况"].str.contains("低保")
        data.loc[index, "城乡低保户"] = 1
        data.loc[-index, "城乡低保户"] = 0

        data["五保户"] = ''
        index = data["享受国家政策资助情况"].str.contains("五保")
        data.loc[index, "五保户"] = 1
        data.loc[-index, "五保户"] = 0

        data["农村特困供养"] = ''
        index = data["享受国家政策资助情况"].str.contains("农村特困供养")
        data.loc[index, "农村特困供养"] = 1
        data.loc[-index, "农村特困供养"] = 0

        data["孤残学生"] = ''
        index = data["享受国家政策资助情况"].str.contains("孤残")
        data.loc[index, "孤残学生"] = 1
        data.loc[-index, "孤残学生"] = 0

        data["军烈属或优抚子女"] = ''
        index = data["享受国家政策资助情况"].str.contains("军烈属")
        data.loc[index, "军烈属或优抚子女"] = 1
        data.loc[-index, "军烈属或优抚子女"] = 0
