# -*- coding: utf-8 -*-
# 导入必要的库
import argparse  # 用于解析命令行参数
import os  # 用于操作系统相关功能
import random  # 用于生成随机数
import re  # 用于正则表达式

import yaml  # 用于解析YAML文件


def get_cur_path():
    """获取当前文件的绝对路径"""
    # 返回当前文件所在目录的绝对路径
    return os.path.dirname(__file__)


# 定义默认配置文件路径
DEFAULT_FILE = os.path.join(get_cur_path(), "default.yaml")


class Config(object):
    """配置解析器类，用于合并多个来源的配置参数"""

    def __init__(self, config_file=None, variable_dict=None, is_resume=False):
        """初始化配置解析器
        
        Args:
            config_file: 用户定义的配置文件路径
            variable_dict: 运行时变量字典 
            is_resume: 是否从检查点恢复训练
        """
        self.is_resume = is_resume  # 是否恢复训练标志
        self.config_file = config_file  # 用户配置文件路径
        # 按优先级从低到高加载各种配置
        self.console_dict = self._load_console_dict()  # 加载命令行参数(最高优先级)
        self.default_dict = self._load_config_files(DEFAULT_FILE)  # 加载默认配置(最低优先级)
        self.file_dict = self._load_config_files(config_file)  # 加载用户配置文件
        self.variable_dict = self._load_variable_dict(variable_dict)  # 加载运行时变量
        self.config_dict = self._merge_config_dict()  # 合并所有配置

    def get_config_dict(self):
        """获取最终合并后的配置字典"""
        return self.config_dict

    @staticmethod
    def _load_config_files(config_file):
        """加载并解析YAML配置文件
        
        Args:
            config_file: 要加载的YAML文件路径
            
        Returns:
            解析后的配置字典
        """
        config_dict = dict()  # 初始化空字典
        loader = yaml.SafeLoader  # 使用安全的YAML加载器
        
        # 添加对浮点数的隐式解析器
        loader.add_implicit_resolver(
            "tag:yaml.org,2002:float",  # YAML浮点数标签
            re.compile(  # 匹配浮点数的正则表达式
                """^(?:
                     [-+]?[0-9][0-9_]*\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
                    |[-+]?[0-9][0-9_]*[eE][-+]?[0-9]+
                    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
                    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
                    |[-+]?\\.(?:inf|Inf|INF)
                    |\\.(?:nan|NaN|NAN))$""",
                re.X,
            ),
            list("-+0123456789."),  # 允许的字符
        )

        if config_file is not None:  # 如果提供了配置文件路径
            with open(config_file, "r", encoding="utf-8") as fin:  # 打开文件
                config_dict.update(yaml.load(fin.read(), Loader=loader))  # 解析并更新配置
        
        # 处理include文件
        config_file_dict = config_dict.copy()  # 复制原始配置
        for include in config_dict.get("includes", []):  # 遍历includes列表
            # 拼接include文件路径
            with open(os.path.join("./config/", include), "r", encoding="utf-8") as fin:
                config_dict.update(yaml.load(fin.read(), Loader=loader))  # 加载并合并include文件
        
        if config_dict.get("includes") is not None:  # 如果存在includes键
            config_dict.pop("includes")  # 移除includes键，因为它只是用于引用
        
        config_dict.update(config_file_dict)  # 确保原始配置优先级高于include文件
        return config_dict

    @staticmethod
    def _load_variable_dict(variable_dict):
        """加载运行时变量字典
        
        Args:
            variable_dict: 运行时变量字典
            
        Returns:
            变量字典或空字典
        """
        config_dict = dict()  # 初始化空字典
        # 如果提供了变量字典则更新，否则使用空字典
        config_dict.update(variable_dict if variable_dict is not None else {})
        return config_dict

    @staticmethod
    def _load_console_dict():
        """解析命令行参数
        
        Returns:
            包含命令行参数的字典(不包含值为None的参数)
        """
        parser = argparse.ArgumentParser()  # 创建参数解析器
        
        # 添加各种命令行参数定义
        parser.add_argument("-w", "--way_num", type=int, help="way num")
        parser.add_argument("-s", "--shot_num", type=int, help="shot num")
        parser.add_argument("-q", "--query_num", type=int, help="query num")
        parser.add_argument("-bs", "--batch_size", type=int, help="batch_size")
        parser.add_argument("-es", "--episode_size", type=int, help="episode_size")
        parser.add_argument("-data", "--data_root", help="dataset path")
        parser.add_argument("-log_name", "--log_name", help="specific log dir name")
        parser.add_argument("-image_size", type=int, help="image size")
        parser.add_argument("-aug", "--augment", type=bool, help="use augment or not")
        parser.add_argument("-aug_times", "--augment_times", type=int, 
                          help="augment times (for support in few-shot)")
        parser.add_argument("-aug_times_query", "--augment_times_query", type=int,
                          help="augment times for query in few-shot")
        parser.add_argument("-train_episode", type=int, help="train episode num")
        parser.add_argument("-test_episode", type=int, help="test episode num")
        parser.add_argument("-epochs", type=int, help="epoch num")
        parser.add_argument("-result", "--result_root", help="result path")
        parser.add_argument("-save_interval", type=int, help="checkpoint save interval")
        parser.add_argument("-log_level", help="log level in: debug, info, warning, error, critical")
        parser.add_argument("-log_interval", type=int, help="log interval")
        parser.add_argument("-gpus", "--device_ids", help="device ids")
        parser.add_argument("-n_gpu", type=int, help="gpu num")
        parser.add_argument("-seed", type=int, help="seed")
        parser.add_argument("-deterministic", type=bool, help="deterministic or not")
        parser.add_argument("-tag", "--tag", type=str, help="experiment tag")
        
        args = parser.parse_args()  # 解析命令行参数
        # 返回非空参数的字典
        return {k: v for k, v in vars(args).items() if v is not None}

    def _recur_update(self, dic1, dic2):
        """递归合并两个字典
        
        dic2中的值会覆盖dic1中的相同键的值
        
        Args:
            dic1: 被更新的字典(低优先级)
            dic2: 更新字典(高优先级)
            
        Returns:
            合并后的字典
        """
        if dic1 is None:  # 如果dic1为空
            dic1 = dict()  # 初始化为空字典
            
        for k in dic2.keys():  # 遍历dic2的所有键
            if isinstance(dic2[k], dict):  # 如果值是字典
                # 递归合并子字典
                dic1[k] = self._recur_update(
                    dic1[k] if k in dic1.keys() else None, dic2[k]
                )
            else:  # 如果不是字典
                dic1[k] = dic2[k]  # 直接更新值
        return dic1

    def _update(self, dic1, dic2):
        """简单合并两个字典
        
        dic2中的值会覆盖dic1中的相同键的值
        
        Args:
            dic1: 被更新的字典(低优先级)
            dic2: 更新字典(高优先级)
            
        Returns:
            更新后的字典
        """
        if dic1 is None:  # 如果dic1为空
            dic1 = dict()  # 初始化为空字典
            
        for k in dic2.keys():  # 遍历dic2的所有键
            dic1[k] = dic2[k]  # 更新/添加键值对
        return dic1

    def _merge_config_dict(self):
        """合并所有来源的配置
        
        按照优先级合并: 命令行参数 > 变量字典 > 用户配置文件 > 默认配置
        
        Returns:
            最终合并后的配置字典
        """
        config_dict = dict()  # 初始化空配置字典
        
        # 按照优先级从低到高合并配置
        config_dict = self._update(config_dict, self.default_dict)  # 合并默认配置
        config_dict = self._update(config_dict, self.file_dict)    # 合并用户配置
        config_dict = self._update(config_dict, self.variable_dict) # 合并变量字典
        config_dict = self._update(config_dict, self.console_dict) # 合并命令行参数(最高优先级)
        
        # 处理测试参数的默认值
        if config_dict["test_way"] is None:
            config_dict["test_way"] = config_dict["way_num"]
        if config_dict["test_shot"] is None:
            config_dict["test_shot"] = config_dict["shot_num"]
        if config_dict["test_query"] is None:
            config_dict["test_query"] = config_dict["query_num"]
            
        # 如果未指定端口，随机选择一个可用端口
        if config_dict["port"] is None:
            port = random.randint(25000, 55000)  # 随机端口范围
            while self.is_port_in_use("127.0.0.1", port):  # 检查端口是否被占用
                old_port = port
                port = str(int(port) + 1)  # 端口号+1
                print(f"Warning: Port {old_port} is already in use, switch to port {port}")
            config_dict["port"] = port  # 设置最终端口号
            
        # 添加额外配置信息
        config_dict["resume"] = self.is_resume  # 设置恢复训练标志
        if self.is_resume:  # 如果是恢复训练
            # 设置恢复路径(去掉文件名部分)
            config_dict["resume_path"] = self.config_file[: -1 * len("/config.yaml")]
        # 计算tensorboard缩放比例
        config_dict["tb_scale"] = float(config_dict["train_episode"]) / config_dict["test_episode"]
        
        return config_dict  # 返回最终配置字典

    def is_port_in_use(self, host, port):
        """检查端口是否被占用
        
        Args:
            host: 主机地址
            port: 端口号
            
        Returns:
            bool: 端口是否被占用
        """
        import socket  # 导入socket模块
        
        # 创建socket并尝试连接
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex((host, int(port))) == 0  # 返回连接结果