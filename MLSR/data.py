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
        self.weak_data.to_csv(out_path + '_weak.csv', encoding="utf-8")
        self.strong_data.to_csv(out_path + '_strong.csv', encoding="utf-8")
        self.strong_data = self.data[400:]
        return self.weak_data, self.strong_data

    @staticmethod
    def do_nation_policy(d: pd.DataFrame, drop: bool = True, is_income_contained: bool = True):
        """

        :param d:
        :param drop:
        :param is_income_contained:
        :return:
        """
        d = pd.DataFrame(d)
        d["建档立卡贫困户"] = d["享受国家政策资助情况"].str.contains("立卡", na=False)
        d["城乡低保户"] = d["享受国家政策资助情况"].str.contains("低保", na=False)
        if is_income_contained:

        d["五保户"] = d["享受国家政策资助情况"].str.contains("五保", na=False)
        d["孤残学生"] = d["享受国家政策资助情况"].str.contains("孤残", na=False)
        d["军烈属或优抚子女"] = d["享受国家政策资助情况"].str.contains("军烈属", na=False)
        if drop:
            d.drop('享受国家政策资助情况', inplace=True)
        return d

    @staticmethod
    def do_income(d: pd.Series, drop: bool = True, fill_to_no_income: bool = True):

        # 家庭主要经济来源，这一项每人只有一种选择（主要经济来源），因此不需要哑变量

        def f(x: str):


        if fill_to_no_income:
            d["家庭主要经济来源"] = d["家庭主要经济来源"].fillna('父母均下岗')

        index = d["家庭主要经济来源"].str.contains("生意|经营|从商|经商|地摊|摆摊|杂货铺|店|卖|买|个体|餐|理发|手工|个体|水果摊|蒸馒头|股票")
        d.loc[index, "家庭主要经济来源"] = "做生意"

        index = (data["家庭主要经济来源"] == "城镇") | (data["家庭主要经济来源"] == "父母劳作") | (data["家庭主要经济来源"] == "家长工资") | (
                    data["家庭主要经济来源"] == "父亲、母亲") | (data["家庭主要经济来源"] == "父母微薄收入") | (data["家庭主要经济来源"] == "工薪") | (
                            data["家庭主要经济来源"] == "基本工资") | (data["家庭主要经济来源"] == "职工工资") | (
                            data["家庭主要经济来源"] == "父母工资") | (data["家庭主要经济来源"] == "父母工作收入") | (
                            data["家庭主要经济来源"] == "父母") | (data["家庭主要经济来源"] == "工作") | (data["家庭主要经济来源"] == "父母工作") | (
                            data["家庭主要经济来源"] == "父母收入") | (data["家庭主要经济来源"] == "工资收入") | (
                            data["家庭主要经济来源"] == "父母工资") | (data["家庭主要经济来源"] == "工资") | (data["家庭主要经济来源"].str.contains(
            '打工|务工|农民工|工地|零工|临时工|工人|临工|短工|小工|散工|出租车|货车|教师|苦力|司机|体力劳动|保安|看守|送货|公交车|裁缝|保姆|上班|工活|教书|清洁工|营业员|城市|普通职工|诊所|超市工作|跑保险|打杂|干活|杂工|十字绣|代教|职员|瓷砖|看门|建房子|职工|房屋出租|房租|自由职业|副业|父母工资收入|父母劳动收入|劳务报酬|父母的工资|做工|劳动收入|卫生所从医|工作收入|不固定|不稳定|无稳定|非固定|非稳定|无固定|没有固定'))
        data.loc[index, "家庭主要经济来源"] = "打工"

        index = data["家庭主要经济来源"].str.contains(
            '务农|农作|农民|农收|农业|农务|农村|农活|耕|种植|种地|种粮|庄稼|田|农产品|土地|葡萄|果树|果园|畜|玉米|梨园|牧|养殖|苹果|枣')
        data.loc[index, "家庭主要经济来源"] = "务农"

        index = (data["家庭主要经济来源"] == "未写") | (data["家庭主要经济来源"] == "暂无") | (data["家庭主要经济来源"] == "无") | (
                    data["家庭主要经济来源"] == "兄长") | (data["家庭主要经济来源"] == "姐姐的工资") | (data["家庭主要经济来源"] == "哥哥工作") | (
                            data["家庭主要经济来源"] == "姐姐 哥哥") | (data["家庭主要经济来源"] == "姐姐的工资收入") | (
                            data["家庭主要经济来源"] == "哥哥工资") | (data["家庭主要经济来源"] == "本人及奶奶的低保金") | (
                            data["家庭主要经济来源"] == "姐姐工资") | (data["家庭主要经济来源"] == "现靠父母过去的工资") | (
                            data["家庭主要经济来源"] == "靠姑姑接济") | (data["家庭主要经济来源"] == "父亲无固定工作，现停业在家\n母亲一直无工作") | data[
                    "家庭主要经济来源"].str.contains('父母无业|双方失业|父母均无业|父母下岗|均下岗|双下岗|亲友|接济|救济|资助|勤工俭学|经济扶持|补助|补贴|寄养家庭|父母离岗工资|社保')
        data.loc[index, "家庭主要经济来源"] = "父母均下岗"

        index = data["家庭主要经济来源"].str.contains('低保|最低生活保障')
        data.loc[index, "城乡低保户"] = 1  # 有些人在前面没写低保，应该加上
        data.loc[index, "家庭主要经济来源"] = "父母均下岗"

        index = (data["家庭主要经济来源"] == "父母一方下岗") | (data["家庭主要经济来源"] == "父亲每月工资") | (data["家庭主要经济来源"] == "母亲基本工资") | (
                    data["家庭主要经济来源"] == "父亲的薪水") | (data["家庭主要经济来源"] == "父亲基本工资") | (data["家庭主要经济来源"] == "父亲和兄长收入") | (
                            data["家庭主要经济来源"] == "父亲姐姐工资") | (data["家庭主要经济来源"] == "母亲单位工资") | (
                            data["家庭主要经济来源"] == "父亲上岗") | (data["家庭主要经济来源"] == "父亲劳务派遣") | (
                            data["家庭主要经济来源"] == "父亲固定工资收入") | (data["家庭主要经济来源"] == "父亲个人工资") | (
                            data["家庭主要经济来源"] == "父亲微薄工资") | (data["家庭主要经济来源"] == "4050公益岗位收入") | (
                            data["家庭主要经济来源"] == "父兄工资") | (data["家庭主要经济来源"] == "爸爸") | (data["家庭主要经济来源"] == "父亲的收入") | (
                            data["家庭主要经济来源"] == "父亲的工作") | (data["家庭主要经济来源"] == "父亲") | (data["家庭主要经济来源"] == "母亲") | (
                    data["家庭主要经济来源"].str.contains(
                        '一方|母亲固定收入|父亲固定收入|母亲下岗|父亲下岗|父亲失业|母亲失业|一人工资|一人的工资|爸爸工资|妈妈工资|爸爸的工资|妈妈的工资|父亲的工资|父亲工资|母亲的工资|母亲工资|父亲工作|母亲工作|爸爸工作|妈妈工作|父亲上班|母亲上班|父亲收入|母亲收入|父亲无业'))
        data.loc[index, "家庭主要经济来源"] = "父母一方下岗"

        index = data["家庭主要经济来源"].str.contains('退休|养老|退养|病休')
        data.loc[index, "家庭主要经济来源"] = "退休金或养老金"

        index = data["家庭主要经济来源"].str.contains('内退|病退|退职')
        data.loc[index, "家庭主要经济来源"] = "退职金或病退金"