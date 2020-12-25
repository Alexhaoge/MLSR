import pandas as pd
from numpy import random, zeros
from jieba import suggest_freq, cut


# import re


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
        Args:
            data:

        Returns:

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
    def do_nation_policy(d: pd.DataFrame,
                         drop: bool = True,
                         is_income_contained: bool = True,
                         is_emergency_contained: bool = True
                         ) -> pd.DataFrame:
        """
        处理“享受国家政策资助情况”一列
        Args:
            d:
            drop:
            is_income_contained:
            is_emergency_contained:

        Returns:

        """
        d = pd.DataFrame(d)
        d["建档立卡贫困户"] = d["享受国家政策资助情况"].str.contains("立卡", na=False)
        d["城乡低保户"] = d["享受国家政策资助情况"].str.contains("低保", na=False)
        if is_income_contained:
            d['城乡低保户'] |= d['家庭主要经济来源'].str.contains('低保|最低生活保障', na=False)
        d["五保户"] = d["享受国家政策资助情况"].str.contains("五保", na=False)
        d["孤残学生"] = d["享受国家政策资助情况"].str.contains("孤残", na=False)
        if is_emergency_contained:
            d['孤残学生'] |= d["突发事件情况"].str.contains("父母双亡|父母去世|孤残|孤儿|重大疾病、突发意外致残|本人视力残疾|本人严重烫伤", na=False)
        d["军烈属或优抚子女"] = d["享受国家政策资助情况"].str.contains("军烈属", na=False)
        if drop:
            d.drop('享受国家政策资助情况', inplace=True)
        return d

    @staticmethod
    def _do_income(s: str):
        # don't know how
        pass

    @staticmethod
    def do_income(data: pd.Series, fill_to_no_income: bool = True) -> pd.DataFrame:

        # 家庭主要经济来源，这一项每人只有一种选择（主要经济来源）

        if fill_to_no_income:
            data["家庭主要经济来源"] = data["家庭主要经济来源"].fillna('父母均下岗')
        d = pd.DataFrame()

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
        return d

    @staticmethod
    def education(s: pd.Series) -> pd.DataFrame:
        """

        Args:
            s:

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

        grad = {"幼儿园毕业", "刚毕业", "小学毕业", "小学未毕业", "初中毕业", "初中未毕业", "高中毕业", "大专毕业", "专科毕业", "大学毕业", "三本毕业", "本科毕业",
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

            #         print(str)
            #         print(output)
            #         print(cut_type)
            #         print(cut_loc)

            #         在cut_type中找寻如下pattern，并总结出大学阶段、高中阶段、义务教育阶段各有多少人：
            #         个数和家庭成员都有可能出现，但个数为家庭成员前一个词，因此先检测个数再紧跟着检测家庭成员
            #         年级和学校都有可能出现，但年级一定出现在学校之后，因此先检测学校再检测年级
            #         个数 -> 学校/年级/学校&年级 -> 非学校或年级：这几个人都属于该学校
            #         （个数 ->）家庭成员 -> 学校/年级/学校&年级 -> 非学校或年级：这几个人都属于该学校
            #         （个数 ->）家庭成员 -> 学校/年级/学校&年级 -> 学校/年级/学校&年级 -> 非学校或年级：首先保证两个学校阶段相同，则这种家庭成员分别属于这个阶段；否则人工处理

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
