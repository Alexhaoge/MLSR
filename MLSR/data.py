import pandas as pd
from numpy import random, zeros
from jieba import suggest_freq, cut


class DataSet:
    """
    数据处理的工具类
    """

    def __init__(self, filename: str = None, encode='gbk'):
        """
        导入一个数据集，将原始特征、弱标签和细化的强标签分为features, label, strong_label
        三个属性。label中特别困难为0，一般困难为1，不困难为2；strong_label将四个细化的困难级别
        设为0~3，0最困难，且strong_label中不考虑“不困难”（也就是label=2）的情况。

        Args:
            filename: 文件路径
            encode: 文件编码
        """
        self.features_name = {}
        if filename is None:
            self.features = pd.DataFrame()
            self.label = pd.Series()
            return
        data = pd.read_csv(filename, encoding=encode)
        data = data.apply(lambda x: x.replace("\n", ""))
        if '专家判定等级' in data.columns:
            self.strong_label = data['专家判定等级'] - 1
            self.label = self.strong_label // 2
        else:
            self.strong_label = pd.Series([None] * len(data))
            self.label = data['院系认定贫困类型'].apply(lambda x: 0 if '特' in x else 1)
        self.features = data.drop(['院系认定贫困类型', '专家判定等级'], axis=1, errors='ignore')

    def merge(self, y):
        """
        将一个DataSet加入当前的DataSet尾部
        Args:
            y: 要加入的DataSet

        Returns: 新的DataSet
        todo Notes: 半监督时，strong_label合并有问题，使用静态的static_merge可以解决

        """
        self.features.append(y.features, ignore_index=True)
        self.label.append(y.label, ignore_index=True)
        self.strong_label.append(y.strong_label, ignore_index=True)
        return self

    @staticmethod
    def static_merge(x, y):
        """
        将DataSet y拼接到DataSet x后面，返回一个新的数据集
        Args:
            x: DataSet
            y: DataSet

        Returns: 新的DataSet
        """
        z = DataSet()
        z.features_name = x.features_name
        z.features = pd.concat([x.features, y.features], ignore_index=True)
        z.label = pd.concat([x.label, y.label], ignore_index=True)
        z.strong_label = pd.concat([x.strong_label, y.strong_label], ignore_index=True)
        return z

    def split_by_weak_label(self, reset_index: bool = True):
        """
        将初次分类的得到的结果中，评为特别困难和一般困难的分开挑出来
        Returns: (DataSet, DataSet) 两个DataSet对象，第一个是特别困难，第二个是一般困难

        """
        x0 = DataSet()
        x1 = DataSet()
        index = self.label[self.label == 0].index
        x0.features = self.features.take(index)
        x0.label = self.label.take(index)
        x0.strong_label = self.strong_label.take(index)
        index = self.label[self.label == 1].index
        x1.features = self.features.take(index)
        x1.label = self.label.take(index)
        x1.strong_label = self.strong_label.take(index)
        x0.features_name = self.features_name
        x1.features_name = self.features_name
        if reset_index:
            return x0.reset_index(), x1.reset_index()
        else:
            return x0, x1

    def reset_index(self):
        """
        将数据及重新标号
        Notes: 调用pandas.reset_index时inplace为True
        Returns: 重新标号后的DataSet

        """
        self.features.reset_index(drop=True, inplace=True)
        self.label.reset_index(drop=True, inplace=True)
        self.strong_label.reset_index(drop=True, inplace=True)
        return self

    def convert_to_ssl(self):
        len_name = len(self.features_name)
        self.features_name['f'+str(len_name)] = '院系认定贫困类型'
        self.features['f'+str(len_name)] = self.label
        index = self.label[self.label == 2].index
        self.features.drop(index, inplace=True)
        self.label.drop(index, inplace=True)
        self.strong_label.drop(index, inplace=True)
        return self.reset_index()

    @staticmethod
    @DeprecationWarning
    def special_init(data: pd.DataFrame) -> pd.DataFrame:
        """
        我们使用的数据集中的特殊处理
        一共有16个特征，发现第1772个数据在原数据集里列填串了，第8297个数据无标签，删去
        院系、专业、出生年月、校区没有用，删掉
        Warnings: 这个函数没什么用了，之后删掉
        Args:
            data:待处理的pandas.DataFrame

        Returns:特殊处理后的pandas.DataFrame

        """
        data = data[data.columns[:16]]
        data.drop(labels=[1772, 8297], inplace=True, errors='ignore')
        data.drop(['院系', '专业', '出生年月', '所在校区'], axis=1, inplace=True, errors='ignore')
        return data

    @staticmethod
    @DeprecationWarning
    def shuffle_and_pick(data, out_path: str = 'rand_select') -> tuple:
        """
        随机打乱后随机抽取的400个样本，用于人工再标注细化标签
        Warnings: 这个函数写得不太好，之后删掉
        Args:
            data:待处理数据，pandas.DataFrame
            out_path:保存文件路径

        Returns:划分后的两个pandas.DataFrame

        """
        data['rand'] = random.uniform(0, 1, len(data))
        data.sort_values(by="rand", inplace=True)  # 对data随机排序
        data.drop('rand', axis=1, inplace=True)
        weak_data = data[:400]
        strong_data = data[400:]
        weak_data.to_csv(out_path + '_weak.csv', encoding='utf-8')
        strong_data.to_csv(out_path + '_strong.csv', encoding='utf-8')
        return weak_data, strong_data

    @staticmethod
    def do_nation_policy(data: pd.DataFrame) -> pd.DataFrame:
        """
        处理“享受国家政策资助情况”一列
        Args:
            data: 待处理的pandas.DataFrame，建议传入所有特征

        Returns:pandas.DataFrame,分类哑变量

        """
        d = pd.DataFrame()
        d["建档立卡贫困户"] = data["享受国家政策资助情况"].str.contains("立卡", na=False)
        d["城乡低保户"] = data["享受国家政策资助情况"].str.contains("低保", na=False)
        if '家庭主要经济来源' in data.columns:
            d['城乡低保户'] |= data['家庭主要经济来源'].str.contains('低保|最低生活保障', na=False)
        d["五保户"] = data["享受国家政策资助情况"].str.contains("五保", na=False)
        d["孤残学生"] = data["享受国家政策资助情况"].str.contains("孤残", na=False)
        if '突发事件情况' in data.columns:
            d['孤残学生'] |= data["突发事件情况"].str.contains("父母双亡|父母去世|孤残|孤儿|重大疾病、突发意外致残|本人视力残疾|本人严重烫伤", na=False)
        d["军烈属或优抚子女"] = data["享受国家政策资助情况"].str.contains("军烈属", na=False)
        return d

    @staticmethod
    def do_income(data: pd.DataFrame, fill_to_no_income: bool = True) -> pd.DataFrame:
        """
        家庭主要经济来源
        Args:
            data:
            fill_to_no_income:

        Returns:

        """
        d = pd.DataFrame()
        d["家庭主要经济来源"] = data["家庭主要经济来源"]
        if fill_to_no_income:
            d["家庭主要经济来源"].fillna('父母均下岗', inplace=True)
        business_str = '生意|经营|从商|经商|地摊|摆摊|杂货铺|店|卖|' + \
                       '买|个体|餐|理发|手工|个体|水果摊|蒸馒头|股票'
        d['经商'] = data['家庭主要经济来源'].str.contains(business_str, na=False)

        farm_str = '务农|农作|农民|农收|农业|农务|农村|农活|耕|种植|种地|' + \
                   '种粮|庄稼|田|农产品|土地|葡萄|果树|果园|畜|玉米|梨园|牧|养殖|苹果|枣'
        d['务农'] = data['家庭主要经济来源'].str.contains(farm_str, na=False)

        retire_str = '退休|养老|退养|病休|内退|病退|退职'
        d['退休'] = data['家庭主要经济来源'].str.contains(retire_str, na=False)

        low_str = '低保|最低生活保障'
        d['低保'] = data['家庭主要经济来源'].str.contains(low_str, na=False)

        work_list = [
            '城镇', '父母劳作', '家长工资', '父亲、母亲', '父母微薄收入', '工薪', '基本工资',
            '职工工资', '父母工资', '父母工作收入', '父母', '工作', '父母工作', '父母收入',
            '工资收入', '工资'
        ]
        work_str = '打工|务工|农民工|工地|零工|临时工|工人|临工|短工|小工|散工|' + \
                   '出租车|货车|教师|苦力|司机|体力劳动|保安|看守|送货|公交车|裁缝|' + \
                   '保姆|上班|工活|教书|清洁工|营业员|城市|普通职工|诊所|超市工作|' + \
                   '跑保险|打杂|干活|杂工|十字绣|代教|职员|瓷砖|看门|建房子|职工|' + \
                   '房屋出租|房租|自由职业|副业|父母工资收入|父母劳动收入|劳务报酬|' + \
                   '父母的工资|做工|劳动收入|卫生所从医|工作收入|不固定|不稳定|' + \
                   '无稳定|非固定|非稳定|无固定|没有固定'
        d['打工'] = data['家庭主要经济来源'].apply(lambda x: True if x in work_list else False)
        d['打工'] |= data['家庭主要经济来源'].str.contains(work_str, na=False)

        both_unemployed_list = [
            '未写', '暂无', '无', '兄长', '姐姐的工资', '哥哥工作', '姐姐 哥哥',
            '姐姐的工资收入', '哥哥工资', '本人及奶奶的低保金', '姐姐工资', '现靠父母过去的工资',
            '靠姑姑接济', '父亲无固定工作，现停业在家\n母亲一直无工作'
        ]
        both_unemployed_str = '父母无业|双方失业|父母均无业|父母下岗|均下岗|双下岗|' + \
                              '亲友|接济|救济|资助|勤工俭学|经济扶持|补助|补贴|' + \
                              '寄养家庭|父母离岗工资|社保'
        d['父母均下岗'] = data['家庭主要经济来源'].apply(lambda x: True if x in both_unemployed_list else False)
        d['父母均下岗'] |= data['家庭主要经济来源'].str.contains(both_unemployed_str, na=False)

        one_unemployed_list = [
            '父母一方下岗', '父亲每月工资', '母亲基本工资', '父亲的薪水', '父亲基本工资',
            '父亲和兄长收入', '父亲姐姐工资', '母亲单位工资', '父亲上岗', '父亲劳务派遣',
            '父亲固定工资收入', '父亲个人工资', '父亲微薄工资', '4050公益岗位收入', '父兄工资',
            '爸爸', '父亲的收入', '父亲的工作', '父亲', '母亲'
        ]
        one_unemployed_str = '一方|母亲固定收入|父亲固定收入|母亲下岗|父亲下岗|' + \
                             '父亲失业|母亲失业|一人工资|一人的工资|爸爸工资|妈妈工资|' + \
                             '爸爸的工资|妈妈的工资|父亲的工资|父亲工资|母亲的工资|' + \
                             '母亲工资|父亲工作|母亲工作|爸爸工作|妈妈工作|父亲上班|' + \
                             '母亲上班|父亲收入|母亲收入|父亲无业'
        d['父母一方下岗'] = data['家庭主要经济来源'].apply(lambda x: True if x in one_unemployed_list else False)
        d['父母一方下岗'] |= data['家庭主要经济来源'].str.contains(one_unemployed_str, na=False)
        d['家庭人均年收入'] = data['家庭人均年收入']
        d.drop('家庭主要经济来源', inplace=True, axis=1, errors='ignore')
        return d

    @staticmethod
    def do_education(s: pd.Series) -> pd.DataFrame:
        """
        对每一行用jieba进行分词，对结果进行遍历，搜索关键词并记录其词性，记为cut_type
        在cut_type中找寻如下pattern，并总结出大学阶段、高中阶段、义务教育阶段各有多少人：
        个数和家庭成员都有可能出现，但个数为家庭成员前一个词，因此先检测个数再紧跟着检测家庭成员
        年级和学校都有可能出现，但年级一定出现在学校之后，因此先检测学校再检测年级
        个数 -> 学校/年级/学校&年级 -> 非学校或年级：这几个人都属于该学校
        （个数 ->）家庭成员 -> 学校/年级/学校&年级 -> 非学校或年级：这几个人都属于该学校
        （个数 ->）家庭成员 -> 学校/年级/学校&年级 -> 学校/年级/学校&年级 -> 非学校或年级：首先保证两个学校阶段相同，则这种家庭成员分别属于这个阶段；否则人工处理

        Args:
            s: 输入的pandas.Series

        Returns:

        """
        s = s.fillna('无')
        zero_tmp1 = ['暂无', '独生', '无在读', '无在受', '无其他', '无成员', '无高中', '无正在']
        zero_tmp2 = ['0', '0人', '0人在读高中或大学']

        def zero_fun(x):
            if x in zero_tmp2:
                return True
            else:
                for _i in zero_tmp1:
                    if _i in x:
                        return True
            return False

        s = s.apply(lambda x: '无' if zero_fun(x) else x)

        # 创建关键字
        number1 = {"一个", "1个", "一人", "1人", "一位", "1位"}
        number2 = {"两个", "2个", "两人", "2人", "二人", "两位", "2位", "二位"}
        number3 = {"三个", "3个", "三人", "3人", "三位", "3位"}
        number4 = {"四个", "4个", "四人", "4人", "四位", "4位"}

        member = {"哥哥", "姐姐", "弟弟", "妹妹", "侄女", "侄子"}
        sp_member = {"哥", "兄", "姐", "弟", "妹", "大哥", "二哥", "大弟", "二弟", "三弟", "四弟", "五弟", "小弟", "大姐", "长姐", "二姐", "三姐",
                     "大妹", "小妹", "二妹", "三妹", "四妹"}
        invalid_member = {"爸爸", "父亲", "妈妈", "母亲", "爷爷", "奶奶", "外祖父", "外祖母", "姥爷", "姥姥", "伯伯", "婆婆", "外公", "外婆"}
        invalid_sp_member = {"爸", "妈", "爷", "奶", "祖父", "祖母", "父", "母"}

        grad = {"幼儿园毕业", "刚毕业", "小学毕业", "小学未毕业", "初中毕业", "初中未毕业", "高中毕业", "大专毕业", "专科毕业", "大学毕业", "三本毕业",
                "应届毕业", "大学已毕业", "大学刚毕业", "本科毕业", "研究生毕业", "硕士毕业", "博士毕业"}
        sp_grad = {"毕业", "未受教育", "未接受教育", "文盲", "无学历", "没上过学", "肄业", "未上学"}
        college = {"研究生", "读研", "博士", "硕士", "大学", "本科", "专升本", "大学生", "河工大"}
        sp_college = {"考研", "大专", "学院", "高职", "专科", "职业技术学校"}
        gr_college = {"大一", "大二", "大三", "大四", "研一", "研二", "研三", "博二", "读博"}
        high_school = {"高中", "高专", "职高", "职专", "职中", "中专", "中学", "职业高中"}
        gr_high_school = {"高一", "高二", "高三", "高考"}

        compulsory = {"初中", "小学"}
        gr_compulsory = {"初一", "初二", "初三", "初中三年级", "一年级", "二年级", "三年级", "四年级", "五年级", "六年级", "七年级", "八年级", "九年级",
                         "义务教育阶段"}
        others = {"幼儿园", "学前班", "学前教育"}

        # 手动检查了前200个数据，发现一些分词结果有误
        suggest_word = [
            '兄弟', '高二', '小学毕业', '初中毕业', '高中毕业', '职高毕业', '大学毕业',
            '大专毕业', '1人', '1个', '1位', '2人', '2个', '2位', '3人', '3个',
            '3位', '4人', '西安交通大学', '海南师范大学', '河工大',
            '中国科学院', '北京航空航天大学', '北京大学', '义务教育阶段'
        ]
        suggest_split = [
            ('父', '母'), ('兄', '妹'), ('兄', '妹'), ('姐', '弟'),
            ('姐', '妹'), ('读', '高二'), ('中医药', '大学'), ('农', '学院')
        ]
        for i in suggest_word:
            suggest_freq(tuple(i), True)
        for i in suggest_split:
            suggest_freq(i, True)

        # 对每个字符串进行分词，并且找寻其中关键字，记录关键字的词性和位置
        arr = zeros(shape=(len(s), 3))
        df = pd.DataFrame(arr, columns=["大学", "高中", "义务教育"])
        row = -1  # 记录str在s中的位置

        for str_i in s:
            row += 1
            seg_list = cut(str_i, cut_all=False)  # 对每个字符串进行分词
            output = list(seg_list)
            cut_type = []  # 记录某个关键字的词性
            cut_loc = []  # 记录某个关键字在str中的位置
            loc = 0  # loc为某个词在原字符串中的位置(1~len)

            for cut_i in output:  # cut为分词结果中的每个词
                loc += 1
                if cut_i in number1:
                    cut_type.append("number1")
                    cut_loc.append(loc - 1)  # cut在原字符串中的索引值(0~len-1)
                elif cut_i in number2:
                    cut_type.append("number2")
                    cut_loc.append(loc - 1)
                elif cut_i in number3:
                    cut_type.append("number3")
                    cut_loc.append(loc - 1)
                elif cut_i in number4:
                    cut_type.append("number4")
                    cut_loc.append(loc - 1)
                elif cut_i in member:
                    cut_type.append("member")
                    cut_loc.append(loc - 1)
                elif cut_i in sp_member:
                    cut_type.append("sp_member")
                    cut_loc.append(loc - 1)
                elif cut_i in invalid_member:
                    cut_type.append("invalid_member")
                    cut_loc.append(loc - 1)
                elif cut_i in invalid_sp_member:
                    cut_type.append("invalid_sp_member")
                    cut_loc.append(loc - 1)
                elif cut_i in grad:
                    cut_type.append("grad")
                    cut_loc.append(loc - 1)
                elif cut_i in sp_grad:
                    cut_type.append("sp_grad")
                    cut_loc.append(loc - 1)
                elif cut_i in college:
                    cut_type.append("college")
                    cut_loc.append(loc - 1)
                elif cut_i in sp_college:
                    cut_type.append("sp_college")
                    cut_loc.append(loc - 1)
                elif cut_i in gr_college:
                    cut_type.append("gr_college")
                    cut_loc.append(loc - 1)
                elif cut_i in high_school:
                    cut_type.append("high_school")
                    cut_loc.append(loc - 1)
                elif cut_i in gr_high_school:
                    cut_type.append("gr_high_school")
                    cut_loc.append(loc - 1)
                elif cut_i in compulsory:
                    cut_type.append("compulsory")
                    cut_loc.append(loc - 1)
                elif cut_i in gr_compulsory:
                    cut_type.append("gr_compulsory")
                    cut_loc.append(loc - 1)
                elif cut_i in others:
                    cut_type.append("others")
                    cut_loc.append(loc - 1)

            serial = 0
            cut_type.append(" ")
            cut_type.append(" ")  # 确保检测到最后一位也能检测其后两位的元素的词性
            u_num = 0  # 大学生人数
            h_num = 0  # 高中生人数
            c_num = 0  # 义务教育人数
            number = 0  # 待定的人数
            while serial < len(cut_type) - 2:
                current_group = []
                # 每一个pattern起始的词只能为个数或家庭成员
                if (cut_type[serial] in ["number1", "number2", "number3", "number4", "member", "sp_member",
                                         "invalid_member", "invalid_sp_member"]):
                    current_group.append(cut_type[serial])
                    for forward in range(serial + 1, len(cut_type) - 1):  # 开始检测该pattern内后面的词
                        current_group.append(cut_type[forward])
                        if ((cut_type[forward] in [" ", "grad", "sp_grad", "college", "sp_college", "gr_college",
                                                   "high_school", "gr_high_school", "compulsory", "gr_compulsory",
                                                   "others"]) and (
                                cut_type[forward + 1] in [" ", "number1", "number2", "number3", "number4", "member",
                                                          "sp_member", "invalid_member", "invalid_sp_member"])):
                            serial = forward + 1  # 找到学校或年级 -> 非学校或年级，这一组完成，定位至下一组首个词
                            break

                    # 此时一组已经检测完成，对其进行匹配
                    group_serial = 0
                    current_group.append(" ")
                    current_group.append(" ")  # 确保检测到最后一位也能检测其后两位的元素是否为学校

                    while group_serial < len(current_group) - 2:
                        # 个数 -> 学校 -> 非学校或年级
                        if ((current_group[group_serial] in ["number1", "number2", "number3", "number4"]) and (
                                current_group[group_serial + 1] in ["college", "sp_college", "gr_college"])):
                            if current_group[group_serial] == "number1":
                                u_num += 1
                            elif current_group[group_serial] == "number2":
                                u_num += 2
                            elif current_group[group_serial] == "number3":
                                u_num += 3
                            elif current_group[group_serial] == "number4":
                                u_num += 4

                        elif ((current_group[group_serial] in ["number1", "number2", "number3", "number4"]) and (
                                current_group[group_serial + 1] in ["high_school", "gr_high_school"])):
                            if current_group[group_serial] == "number1":
                                h_num += 1
                            elif current_group[group_serial] == "number2":
                                h_num += 2
                            elif current_group[group_serial] == "number3":
                                h_num += 3
                            elif current_group[group_serial] == "number4":
                                h_num += 4

                        elif ((current_group[group_serial] in ["number1", "number2", "number3", "number4"]) and (
                                current_group[group_serial + 1] in ["compulsory", "gr_compulsory"])):
                            if current_group[group_serial] == "number1":
                                c_num += 1
                            elif current_group[group_serial] == "number2":
                                c_num += 2
                            elif current_group[group_serial] == "number3":
                                c_num += 3
                            elif current_group[group_serial] == "number4":
                                c_num += 4

                        # (个数 ->) 家庭成员 -> 学校 -> 非学校
                        elif ((current_group[group_serial] in ["member", "sp_member"]) and (
                                current_group[group_serial + 1] in ["grad", "sp_grad", "college", "sp_college",
                                                                    "gr_college", "high_school", "gr_high_school",
                                                                    "compulsory", "gr_compulsory", "others"]) and (
                                      current_group[group_serial + 2] not in ["grad", "sp_grad", "college",
                                                                              "sp_college", "gr_college", "high_school",
                                                                              "gr_high_school", "compulsory",
                                                                              "gr_compulsory", "others"])):
                            if current_group[group_serial - 1] == "number2":
                                number += 2
                            elif current_group[group_serial - 1] == "number3":
                                number += 3
                            elif current_group[group_serial - 1] == "number4":
                                number += 4
                            else:
                                number += 1

                            if current_group[group_serial + 1] in ["college", "sp_college", "gr_college"]:
                                u_num += number
                            elif current_group[group_serial + 1] in ["high_school", "gr_high_school"]:
                                h_num += number
                            elif current_group[group_serial + 1] in ["compulsory", "gr_compulsory"]:
                                c_num += number

                            number = 0

                        # 家庭成员 -> 多个学校 -> 非学校
                        elif ((current_group[group_serial] in ["member", "sp_member"]) and (
                                current_group[group_serial + 1] in ["grad", "sp_grad", "college", "sp_college",
                                                                    "gr_college", "high_school", "gr_high_school",
                                                                    "compulsory", "gr_compulsory", "others"]) and (
                                      current_group[group_serial + 2] in ["grad", "sp_grad", "college", "sp_college",
                                                                          "gr_college", "high_school", "gr_high_school",
                                                                          "compulsory", "gr_compulsory", "others"])):
                            group_serial += 1
                            while group_serial < len(current_group) - 2:
                                if (current_group[group_serial] not in ["grad", "sp_grad", "college", "sp_college",
                                                                        "gr_college", "high_school", "gr_high_school",
                                                                        "compulsory", "gr_compulsory", "others"]):
                                    break

                                elif current_group[group_serial] in ["college", "sp_college", "gr_college"]:
                                    u_num += 1
                                elif current_group[group_serial] in ["high_school", "gr_high_school"]:
                                    h_num += 1
                                elif current_group[group_serial] in ["compulsory", "gr_compulsory"]:
                                    c_num += 1
                                group_serial += 1

                        # 多个家庭成员 -> 学校 -> 非学校
                        elif ((current_group[group_serial] in ["member", "sp_member"]) and (
                                current_group[group_serial + 1] in ["member", "sp_member", "invalid_member",
                                                                    "invalid_sp_member"])):
                            group_serial += 1
                            while group_serial < len(current_group) - 2:
                                if (current_group[group_serial] not in ["member", "sp_member", "invalid_member",
                                                                        "invalid_sp_member"]):
                                    break
                                elif current_group[group_serial] in ["member", "sp_member"]:
                                    number += 1
                                group_serial += 1

                            if current_group[group_serial] in ["college", "sp_college", "gr_college"]:
                                u_num += number
                            elif current_group[group_serial] in ["high_school", "gr_high_school"]:
                                h_num += number
                            elif current_group[group_serial] in ["compulsory", "gr_compulsory"]:
                                c_num += number

                            number = 0

                        group_serial += 1

                else:  # 该词不为个数或家庭成员，则访问下一个
                    serial += 1

            df.iloc[row] = [u_num, h_num, c_num]
        return df

    @staticmethod
    def do_accident(s: pd.Series):
        """
        识别突发事件情况
        部分处理思路如下：
        在cut_type中找寻如下pattern，并总结出每个人发生什么事情：
        每一个人可能同时做多件事情，而这多件事情肯定是按顺序排列的，这些事情之间一定不出现另一个人名
        无论在哪里出现divorce，一定为父母离异，但是如果出现在“父母”之后，则需要把这个“父母”与divorce绑定
        这里jieba无法将“父母”、“父/母”、“父(母)”、“父亲（母亲）”分开，所以需要加一个判断条件，会用在后面无业、患病、去世中
        一个人之后可能跟着多个illness，应全部与其绑定。若人是祖父母，则统计其是否患病；父母则看是否有重病，且应将父母辨别开；兄弟姐妹只统计重疾。
        有可能出现人 -> illness ->dead。所有人都有可能dead，dead需要与之前最近的一个人或连续的多个人绑定，但只统计父或母去世。
        Args:
            s:待处理的pandas.Series

        Returns:处理后得到的哑变量特征，pandas.Dataframe格式

        """
        s = s.fillna('无')
        zero_tmp1 = ['无', '否', '没有', '正常', '暂无']
        s = s.apply(lambda x: '无' if x in zero_tmp1 else x)

        # 创建关键字
        dad = {"爸爸", "父亲", "爸", "父"}
        mom = {"妈妈", "母亲", "妈", "母"}
        grand_parents = {"老人", "长辈", "祖父母", "爷爷", "奶奶", "外祖父", "外祖母",
                         "姥爷", "姥姥", "外公", "外婆"}
        sp_grand_parents = {"爷", "奶", "祖父", "祖母"}
        siblings = {"哥哥", "姐姐", "弟弟", "妹妹"}
        sp_siblings = {"哥", "兄", "姐", "弟", "妹", "大哥", "二哥", "大弟",
                       "二弟", "三弟", "四弟", "五弟", "小弟", "大姐", "长姐",
                       "二姐", "三姐", "大妹", "小妹", "二妹", "三妹", "四妹"}
        invalid_member = {"我", "本人", "自己", "侄女", "侄子", "伯伯", "伯母",
                          "大伯", "二伯", "三伯", "伯父", "婆婆", "舅舅", "小姑",
                          "姑姑", "二姑", "大舅", "叔叔", "叔父", "二叔", "老叔", "舅妈"}
        divorce = {"单亲", "离婚", "离异"}
        unemployed = {"无业", "一方无业", "均无业", "失业", "无法工作", "下岗",
                      "公司破产", "无工作", "待业", "倒闭", "离职", "未有收入",
                      "无收入", "停产", "失去稳定工作", "没能工作"}
        dead = {"去世", "离世", "病逝", "死亡", "治丧", "身亡", "病故"}
        illness = {
            "病", "病了", "就医", "发病", "有病", "多病", "车祸", "住院", "养病", "疾病",
            "病情", "受伤", "顽疾", "服药", "腰伤", "慢性疾病", "普通疾病", "一般疾病",
            "生病", "患病", "带病", "患疾", "皮肤病", "高血压", "高血糖", "高血脂",
            "风湿", "类风湿", "风湿病", "心脏病", "糖尿病", "高血脂", "三高", "囊肿",
            "肝囊肿", "结石", "肾结石", "胆结石", "结石病", "尿结石", "肾囊肿",
            "肾积水", "脑溢血", "脑血栓", "心脑血管疾病", "心脑疾病", "青光眼", "慢阻肺",
            "中风", "白内障", "肺结核", "冠心病", "甲亢", "癫痫", "股骨头坏死",
            "风湿", "腿脚不便", "精神病", "精神疾病", "精神分裂症", "精神性疾病", "气胸",
            "胃穿孔", "骨折", "骨裂", "红斑狼疮", "腰椎间盘突出", "腰间盘突出", "关节炎",
            "骨质增生", "胃溃疡", "手术", "腿疾", "胃病", "感染", "胰腺炎", "溃烂",
            "摔伤", "腿伤", "睡眠障碍", "工伤", "视网膜", "白癜风", "关节病", "颈椎病",
            "胆囊炎", "坠楼", "瘸", "贫血", "脱髓鞘", "事故", "意外事故", "服药", "体弱",
            "卷入机器", "气管炎", "支气管炎", "卧病", "交通事故", "吃药", "胃出血",
            "脑出血", "颅内出血", "子宫肌瘤", "腰椎", "颈椎", "腰椎病", "后遗症",
            "割伤", "脑垂体瘤", "脊椎炎", "扎伤", "烫伤", "肺气肿", "卧床", "断裂",
            "眼疾", "伤手", "摔了", "吃药", "旧病复发", "切断", "摔到", "意外事故"}
        serious_illness = {
            "大病", "病重", "重病", "重疾", "重大疾病", "肌无力", "肿瘤", "瘤", "白血病",
            "癌", "患癌", "癌症", "肝癌", "食道癌", "卵巢癌", "甲状腺癌", "肺癌", "宫颈癌",
            "脑癌", "直肠癌", "乳腺癌", "胃癌", "肺腺癌", "贲门癌", "喷门癌", "食道癌",
            "肠癌", "乳癌", "结肠癌", "前列腺癌", "致癌", "肾癌", "淋巴癌", "食道癌",
            "心梗", "心肌梗塞", "脑中风", "移植", "搭桥", "支架", "肾炎", "肾病综合征",
            "肾综合", "严重肾病", "肾衰竭", "尿毒症", "截肢", "肝病", "肝硬化", "肝炎",
            "干重活", "做重活", "不能工作", "失去劳动力", "丧失劳动力", "丧失行动力",
            "无法劳作", "丧失劳动能力", "失去部分劳动力", "失去全部劳动力", "无法承受过重劳动",
            "失去行动能力", "干不了", "干重活", "不能劳作", "不得剧烈运动", "脑梗", "脑梗塞",
            "脑梗死", "脑膜炎", "脑膜瘤", "昏迷", "聋", "失聪", "耳聋", "聋哑人", "聋哑",
            "失明", "瘫痪", "偏瘫", "脑瘫", "截瘫", "致瘫", "帕金森", "瓣膜病", "痴呆",
            "老年痴呆", "老年痴呆症", "烧伤", "火烧", "语言", "贫血", "主动脉", "残疾",
            "残废", "伤残", "摔断", "砸断", "神志不清", "骨髓瘤", "致残", "脑萎缩", "脑血管",
            "脑结核", "半身不遂", "致盲", "病危", "再生性贫血障碍", "生活无法自理", "做手术"
        }

        suggest_word = [
            '严重肾病', '肾病综合征', '失去劳动力', '丧失劳动力', '丧失行动力',
            '无法劳作', '不能劳作', '失去部分劳动力', '失去部分劳动力', '失去行动能力',
            '丧失劳动能力', '无法承受过重劳动', '心脑血管疾病', '心脑疾病', '肺腺癌',
            '脑梗', '不得剧烈运动', '重大疾病', '普通疾病', '一般疾病', '一方无业',
            '均无业', '股骨头坏死', '干重活', '做重活', '腿脚不便', '无法工作', '工作',
            '精神性疾病', '精神官能症', '公司破产', '腰椎间盘突出', '腰间盘突出', '腿疾',
            '无工作', '慢性疾病', '睡眠障碍', '卷入机器', '颅内出血', '子宫肌瘤',
            '未有收入', '脑垂体瘤', '病了', '再生性贫血障碍', '摔了', '生活无法自理',
            '失去稳定工作', '没能工作', '旧病复发', '摔到'
        ]
        for _word in suggest_word:
            suggest_freq(tuple(_word), True)
        suggest_split = [
            ('癫痫', '病'), ('卧床', '不起'), ('黑色素', '瘤'), ('单亲', '家庭'), ('恶性', '肿瘤'),
            ('断', '腿'), ('断', '手'), ('家', '父'), ('家', '母')
        ]
        for _split in suggest_split:
            suggest_freq(_split, True)

        # 对每个字符串进行分词，并且找寻其中关键字，记录关键字的词性和位置
        arr = zeros(shape=(len(s), 11))
        df = pd.DataFrame(arr, columns=[
            "祖父母患病", "父母离异", "父亲（母亲）患普通疾病", "父母患普通疾病", "父亲（母亲）无业", "父母均无业", "兄弟姐妹患重疾",
            "父亲（母亲）患重疾", "父母患重疾", "父亲（母亲）去世", "突发重大自然灾害"
        ])
        index = s.str.contains(
            "灾|病虫害|霜冻|地震|台风|洪水|大水|大旱|干旱|冰雹|暴风雨|暴雨|下雪|雷劈|自然状况|" +
            "禽流感|高温|减产|倒伏|淹|涝|庄稼大量死亡|自然状况|自然天气状况|减产|泥石流|猪瘟|庄稼无收")
        adjusted_index = 0  # 由于index中少了一些索引，无法直接匹配到df中，需要调整index
        for i in index:
            if i:
                df.loc[adjusted_index, "突发重大自然灾害"] = 1
            adjusted_index += 1

        row = -1  # 记录str在s中的位置

        for _str in s:
            row += 1
            seg_list = cut(_str, cut_all=False)  # 对每个字符串进行分词
            output = list(seg_list)
            cut_type = []  # 记录某个关键字的词性
            cut_loc = []  # 记录某个关键字在str中的位置
            loc = 0  # loc为某个词在原字符串中的位置(1~len)

            for _cut in output:  # cut为分词结果中的每个词
                loc += 1
                if _cut in dad:
                    cut_type.append("dad")
                    cut_loc.append(loc - 1)  # cut在原字符串中的索引值(0~len-1)

                elif _cut in mom:
                    cut_type.append("mom")
                    cut_loc.append(loc - 1)

                elif _cut in grand_parents:
                    cut_type.append("grand_parents")
                    cut_loc.append(loc - 1)

                elif _cut in sp_grand_parents:
                    cut_type.append("sp_grand_parents")
                    cut_loc.append(loc - 1)

                elif _cut in siblings:
                    cut_type.append("siblings")
                    cut_loc.append(loc - 1)

                elif _cut in sp_siblings:
                    cut_type.append("sp_siblings")
                    cut_loc.append(loc - 1)

                elif _cut in invalid_member:
                    cut_type.append("invalid_member")
                    cut_loc.append(loc - 1)

                elif _cut in divorce:
                    cut_type.append("divorce")
                    cut_loc.append(loc - 1)

                elif _cut in unemployed:
                    cut_type.append("unemployed")
                    cut_loc.append(loc - 1)

                elif _cut in dead:
                    cut_type.append("dead")
                    cut_loc.append(loc - 1)

                elif _cut in illness:
                    cut_type.append("illness")
                    cut_loc.append(loc - 1)

                elif _cut in serious_illness:
                    cut_type.append("serious_illness")
                    cut_loc.append(loc - 1)

            serial = 0
            cut_type.append(" ")
            cut_type.append(" ")  # 确保检测到最后一位也能检测其后两位的元素的词性
            cut_loc.append(" ")
            cut_loc.append(" ")

            while serial < len(cut_type) - 2:
                flag_dad = False
                flag_mom = False
                flag_parents = False  # 判断是父或母还是父与母
                flag_grand_parents = False
                flag_sp_grand_parents = False
                flag_siblings = False
                flag_sp_siblings = False
                flag_divorce = False
                flag_unemployed = False
                flag_dead = False
                flag_illness = False
                flag_serious_illness = False  # 用来检测每组中是否出现相应的关键词，若有则按照规则赋予相应的类1

                current_group = []
                current_loc = []

                # 每一个pattern起始的词只能为家庭成员
                if (cut_type[serial] in [
                    "dad", "mom", "grand_parents", "sp_grand_parents",
                    "siblings", "sp_siblings", "invalid_member"
                ]):
                    current_group.append(cut_type[serial])
                    current_loc.append(cut_loc[serial])  # 用来判断父母是否连着

                    for forward in range(serial + 1, len(cut_type) - 1):  # 开始检测该pattern内后面的词
                        current_group.append(cut_type[forward])
                        current_loc.append(cut_loc[forward])
                        if ((cut_type[forward] in [" ", "divorce", "unemployed", "dead", "illness",
                                                   "serious_illness"]) and (
                                cut_type[forward + 1] in [" ", "dad", "mom", "grand_parents", "sp_grand_parents",
                                                          "siblings", "sp_siblings", "invalid_member"])):
                            serial = forward + 1  # 找到行为 -> 人，这一组完成，定位至下一组首个词
                            break

                    # 此时一组已经检测完成，对其进行匹配
                    group_serial = 0
                    current_group.append(" ")
                    current_group.append(" ")  # 确保检测到最后一位也能检测其后两位的元素
                    current_loc.append(" ")
                    current_loc.append(" ")
                    current_loc.append(" ")
                    current_loc.append(" ")

                    while group_serial < len(current_group) - 2:
                        # 一个或多个家庭成员 -> 一件或多件事情 -> 非事情
                        if (current_group[group_serial] in ["dad", "mom", "grand_parents", "sp_grand_parents",
                                                            "siblings", "sp_siblings", "invalid_member", "divorce",
                                                            "unemployed", "dead", "illness", "serious_illness"]):
                            if (current_group[group_serial + 1] not in [" ", "dad", "mom", "grand_parents",
                                                                        "sp_grand_parents", "siblings",
                                                                        "sp_siblings", "invalid_member", "divorce",
                                                                        "unemployed", "dead", "illness",
                                                                        "serious_illness"]):
                                break

                            if current_group[group_serial] == "dad":
                                flag_dad = True
                                if ((current_group[group_serial + 1] == "mom") and (
                                        current_loc[group_serial + 1] == current_loc[group_serial] + 1)):
                                    flag_parents = True

                            elif current_group[group_serial] == "mom":
                                flag_mom = True
                                if ((current_group[group_serial + 1] == "dad") and (
                                        current_loc[group_serial + 1] == current_loc[group_serial] + 1)):
                                    flag_parents = True

                            elif current_group[group_serial] == "grand_parents":
                                flag_grand_parents = True

                            elif current_group[group_serial] == "sp_grand_parents":
                                flag_sp_grand_parents = True

                            elif current_group[group_serial] == "siblings":
                                flag_siblings = True

                            elif current_group[group_serial] == "sp_siblings":
                                flag_sp_siblings = True

                            elif current_group[group_serial] == "divorce":
                                flag_divorce = True

                            elif current_group[group_serial] == "unemployed":
                                flag_unemployed = True

                            elif current_group[group_serial] == "dead":
                                flag_dead = True

                            elif current_group[group_serial] == "illness":
                                flag_illness = True

                            elif current_group[group_serial] == "serious_illness":
                                flag_serious_illness = True

                            # 一个pattern中的词已经全部找到，对该pattern进行考察
                            if ((flag_grand_parents or flag_sp_grand_parents) and (
                                    flag_illness or flag_serious_illness)):
                                df.loc[row, "祖父母患病"] = 1

                            if flag_divorce:
                                df.loc[row, "父母离异"] = 1

                            if flag_illness and (flag_dad or flag_mom) and not flag_parents:
                                df.loc[row, "父亲（母亲）患普通疾病"] = 1

                            if flag_illness and flag_dad and flag_mom and flag_parents:
                                df.loc[row, "父母患普通疾病"] = 1

                            if flag_unemployed and (flag_dad or flag_mom) and not flag_parents:
                                df.loc[row, "父亲（母亲）无业"] = 1

                            if flag_unemployed and flag_dad and flag_mom and flag_parents:
                                df.loc[row, "父母均无业"] = 1

                            if flag_serious_illness and (flag_siblings or flag_sp_siblings):
                                df.loc[row, "兄弟姐妹患重疾"] = 1

                            if flag_serious_illness and (flag_dad or flag_mom) and not flag_parents:
                                df.loc[row, "父亲（母亲）患重疾"] = 1

                            if flag_serious_illness and flag_dad and flag_mom and flag_parents:
                                df.loc[row, "父母患重疾"] = 1

                            if flag_dead and (flag_dad or flag_mom) and not flag_parents:
                                df.loc[row, "父亲（母亲）去世"] = 1

                        group_serial += 1

                else:  # 该词不为个数或家庭成员，则访问下一个
                    serial += 1
        return df

    @staticmethod
    def do_scholarship(s: pd.Series) -> pd.DataFrame:
        """
        识别在校期间获得助学金情况
        Args:
            s:待处理的特征，pandas.Series

        Returns:返回三个特征的pandas.DataFrame，助学金个数（离散），助学金总金额（连续），
        获得的国家助学金类型（分类变量，0为未获得，1为国家二等助学金，2为国家一等助学金）

        """
        d = pd.DataFrame()
        d['在校受奖励资助情况'] = s.fillna('无')
        scholar_map = {
            '慧明': 5000, '欧莱雅': 5000, '喜来健': 5000, '中海油': 5000,
            '承锋': 5000, '清茗雅轩': 3000, '盛帆': 3000, '福慧': 2000,
            '柏年': 2000, '圣恩纳': 2000, '香港好友': 2000, '国泰': 5000,
            '思源': 4000, '宋声扬': 5000, '长城': 3000, '交通': 2500,
            '冯顾丽华': 2000, '电装': 3000, '圆梦启航': 6600
        }

        def func(x):
            cnt = 0
            tot = 0
            is_national = 0
            for k, v in scholar_map.items():
                if k in x:
                    cnt += 1
                    tot += v
            if '国家' in x or '国助' in x:
                cnt += 1
                if "一" in x:
                    is_national = 2
                    tot += 3800
                else:
                    is_national = 1
                    tot += 2800
            return cnt, tot, is_national

        d['tmp'] = d['在校受奖励资助情况'].apply(func)
        d[['助学金个数', '助学金金额', '国助类型']] = d['tmp'].apply(pd.Series)
        d.drop(['在校受奖励资助情况', 'tmp'], axis=1, inplace=True)
        return d

    @staticmethod
    def do_resident_type(s: pd.Series) -> pd.Series:
        """
        识别户口类型，缺失值视为城镇户口
        Args:
            s:待处理的特征，pandas.Series

        Returns:返回pandas.Series

        """
        d = s.fillna('城镇')
        return d.apply(lambda x: ('非' in x) ^ ('农' in x))

    @staticmethod
    def do_household(s: pd.Series) -> pd.Series:
        """
        识别家庭人口数量，缺失则视为三口之家
        Args:
            s:待处理的特征，pandas.Series

        Returns:返回pandas.Series

        """
        d = s.fillna(3)
        to_be_replace = {
            '一': '1', '二': '2', '三': '3', '四': '4', '五': '5', '六': '6',
            '七': '7', '八': '8', '九': '9', '人': '', '口': ''
        }

        def func(x):
            if isinstance(x, str):
                for k, v in to_be_replace.items():
                    x = x.replace(k, v)
            return int(x)

        return d.apply(func)

    @staticmethod
    def do_loan(s: pd.Series):
        """
        识别是否贷款，缺失值视为无贷款
        Notes: 只识别生源地和校园地助学贷款，其他贷款不在认定考虑范围内
        Args:
            s:待处理的特征，pandas.Series

        Returns:返回pandas.Series

        """
        res = s.fillna('-')
        yes_list = ['是', '√', '19500', '32000', '助学', '生源地', '校园', '扶贫', '学', '有', '贷款']
        no_list = ['否', '无', '未', '-', '房', '50000', '10万', '借', '商业', '家', '父亲']

        def fun(x):
            loan = False
            for i in yes_list:
                if i in x:
                    loan = True
                    break
            for j in no_list:
                if j in x:
                    return False
            return loan

        return res.apply(fun)

    @staticmethod
    def do_ethnic_group(s: pd.Series) -> pd.Series:
        """
        识别是否为少数民族，缺失值视为汉族
        Args:
            s: 待处理的特征，pandas.Series

        Returns:返回pandas.Series

        """
        d = s.fillna('汉')
        return d.apply(lambda x: False if '汉' in x else True)

    def generate_feature(self):
        """
        按顺序将原始特征转为可使用的特征，并将特征重命名为f1,f2,f3....
        Returns:处理好的DataSet对象，特征名映射在features_name属性中

        """
        d = [
            DataSet.do_nation_policy(self.features[['享受国家政策资助情况', '突发事件情况', '家庭主要经济来源']]),
            DataSet.do_income(self.features[['家庭主要经济来源', '家庭人均年收入']]),
            DataSet.do_education(self.features['家庭其他成员在受教育情况']),
            DataSet.do_accident(self.features['突发事件情况']),
            DataSet.do_scholarship(self.features['在校受奖励资助情况']),
            DataSet.do_ethnic_group(self.features['民族']),
            DataSet.do_household(self.features['家庭人口']),
            DataSet.do_loan(self.features['是否贷款']),
            DataSet.do_resident_type(self.features['入学前户口性质'])
        ]
        new_f = pd.concat(d, axis=1, copy=False)
        new_f['父母均下岗'] |= new_f['父母均无业']
        new_f['父母一方下岗'] |= new_f['父亲（母亲）无业']
        new_f.drop(['父母均无业', '父亲（母亲）无业'], axis=1, inplace=True)
        self.features = new_f
        self.features_name = {'f' + str(i): x for i, x in enumerate(self.features.columns)}
        self.features.columns = self.features_name.keys()
        return

    @staticmethod
    def data_augment(n: int = 1000, filename: str = None):
        """数据增强
        一般数据集中没有非经济困难的，但是这样的模型并不够鲁棒，
        所以我们需要按照一定规则生成非困难的样本，增强数据
        Notes: 收入肯定不是正态分布的，但是想不好用什么，暂时采用对数正态，然后把低保线设在5%分位数

        Args:
            n: 生成的数据条数
            filename: 生成数据的保存路径

        Returns: DataSet对象

        """
        d = DataSet()
        d.features_name = {
            'f0': '建档立卡贫困户', 'f1': '城乡低保户', 'f2': '五保户', 'f3': '孤残学生',
            'f4': '军烈属或优抚子女', 'f5': '经商', 'f6': '务农', 'f7': '退休',
            'f8': '低保', 'f9': '打工', 'f10': '父母均下岗', 'f11': '父母一方下岗',
            'f12': '家庭人均年收入', 'f13': '大学', 'f14': '高中', 'f15': '义务教育',
            'f16': '祖父母患病', 'f17': '父母离异', 'f18': '父亲（母亲）患普通疾病',
            'f19': '父母患普通疾病', 'f20': '兄弟姐妹患重疾', 'f21': '父亲（母亲）患重疾',
            'f22': '父母患重疾', 'f23': '父亲（母亲）去世', 'f24': '突发重大自然灾害',
            'f25': '助学金个数', 'f26': '助学金金额', 'f27': '国助类型', 'f28': '民族',
            'f29': '家庭人口', 'f30': '是否贷款', 'f31': '入学前户口性质'
        }
        d.label = pd.Series([2] * n)
        d.strong_label = pd.Series([None] * n)
        f = pd.DataFrame()
        for i in ['f0', 'f1', 'f2', 'f3', 'f8', 'f25', 'f26', 'f27', 'f30']:
            f[i] = pd.Series(zeros(n, dtype='int32'))
        f['f4'] = pd.Series(random.binomial(1, 0.002, n))
        f['f10'] = pd.Series(random.binomial(1, 0.002, n))
        f['f11'] = pd.Series(random.binomial(1, 0.02, n))
        f['f12'] = pd.Series(random.lognormal(10.1811, 0.1892, n))
        f['f13'] = pd.Series(random.binomial(3, 0.01, n))
        f['f14'] = pd.Series(random.binomial(3, 0.01, n))
        f['f15'] = pd.Series(random.binomial(3, 0.035, n))
        f['f16'] = pd.Series(random.binomial(1, 0.15, n))
        f['f17'] = pd.Series(random.binomial(1, 0.01, n))
        f['f18'] = pd.Series(random.binomial(1, 0.01, n))
        f['f19'] = pd.Series(random.binomial(1, 0.05, n))
        f['f20'] = pd.Series(random.binomial(1, 0.003, n))
        f['f21'] = pd.Series(random.binomial(1, 0.008, n))
        f['f22'] = pd.Series(random.binomial(1, 0.00036, n))
        f['f23'] = pd.Series(random.binomial(1, 0.008, n))
        f['f24'] = pd.Series(random.binomial(1, 0.01, n))
        f['f28'] = pd.Series(random.binomial(1, 0.05, n))
        f['f29'] = pd.Series(random.binomial(7, 0.4, n))
        f['f31'] = pd.Series(random.binomial(1, 0.1, n))
        f['source'] = f['f31'].apply(lambda x: random.random_integers(1, 15 if x else 7))
        f['f5'] = f['source'].apply(lambda x: x & 1)
        f['f7'] = f['source'].apply(lambda x: (x >> 1) & 1)
        f['f9'] = f['source'].apply(lambda x: (x >> 2) & 1)
        f['f6'] = f['source'].apply(lambda x: (x >> 3) & 1)
        f.drop('source', axis=1, inplace=True)
        d.features = f
        return d
