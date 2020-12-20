import pandas as pd


class DataSet:
    """
    Class for data set manipulation
    """

    def __init__(self, filename: str = '17_19_Data.csv'):
        self.data = pd.read_csv(filename, encoding="gbk")
        self.data = self.data.apply(lambda x: x.replace("\n", ""))

    @staticmethod
    def special_init(data: pd.DataFrame):
        """
        我们使用的数据集中的特殊处理
        一共有16个特征，发现第1772个数据在原数据集里列填串了，第8297个数据无标签，删去
        院系、专业、民族、出生年月、校区没有用，删掉
        :param data:
        :return:
        """
        data = data[data.columns[:16]]
        data.drop(labels=[1772, 8297], inplace=True)
        data.drop(['院系', '专业', '民族', '出生年月', '所在校区'], axis=1, inplace=True)
        return data

    @staticmethod