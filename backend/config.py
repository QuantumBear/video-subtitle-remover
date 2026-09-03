
import os
import json
import configparser
from enum import Enum
from pathlib import Path
from backend.tools.constant import InpaintMode, SubtitleDetectMode

# 项目版本号
VERSION = "1.4.0"
PROJECT_HOME_URL = "https://github.com/YaoFANGUK/video-subtitle-remover"
PROJECT_ISSUES_URL = PROJECT_HOME_URL + "/issues"
PROJECT_RELEASES_URL = PROJECT_HOME_URL + "/releases"
PROJECT_UPDATE_URLS = [
    "https://api.github.com/repos/YaoFANGUK/video-subtitle-remover/releases/latest",
    "https://accelerate.xdow.net/api/repos/YaoFANGUK/video-subtitle-remover/releases/latest",
]

# 硬件加速选项开关
HARDWARD_ACCELERATION_OPTION = True


# ============================================================
# 迷你配置系统:替代 qfluentwidgets 的配置机制(去 GUI 依赖)。
# 对外接口与原实现兼容:config.<item>.value 读、config.set(item, value) 写、
# json 持久化、枚举序列化、范围/选项/布尔校验(非法值回退默认)。
# ============================================================
class ConfigValidator:
    """校验器基类:validate(value) 返回 bool。"""

    def validate(self, value):
        return True


class BoolValidator(ConfigValidator):
    def validate(self, value):
        return isinstance(value, bool)


class RangeValidator(ConfigValidator):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def validate(self, value):
        return isinstance(value, (int, float)) and self.left <= value <= self.right


class OptionsValidator(ConfigValidator):
    def __init__(self, options):
        self.options = list(options)

    def validate(self, value):
        if isinstance(value, Enum):
            return value in self.options
        return value in self.options


class EnumSerializer:
    """枚举成员 ↔ 原始值(字符串)的双向转换,用于 json 持久化。"""

    def __init__(self, enum_class):
        self.enum_class = enum_class

    def serialize(self, value):
        return value.value if isinstance(value, Enum) else value

    def deserialize(self, value):
        try:
            return self.enum_class(value)
        except ValueError:
            # 兼容历史遗留值(如旧版中文枚举值),保留原值交由上层迁移逻辑处理
            return value


class ConfigItem:
    """单个配置项:group/key 定位,validator 校验,serializer 负责 json 序列化。"""

    def __init__(self, group, key, default, validator=None, serializer=None, restart=False):
        self.group = group
        self.key = key
        self.validator = validator
        self.serializer = serializer
        self.defaultValue = default
        self.value = default

    def serialize(self):
        return self.serializer.serialize(self.value) if self.serializer else self.value


def _items(cls):
    """收集配置类上的全部 ConfigItem 实例。"""
    return {name: attr for name, attr in vars(cls).items() if isinstance(attr, ConfigItem)}


class Config:
    # 界面语言设置
    intefaceTexts = {
        '简体中文': 'ch',
        '繁體中文': 'chinese_cht',
        'English': 'en',
        '한국어': 'ko',
        '日本語': 'japan',
        'Tiếng Việt': 'vi',
        'Español': 'es'
    }
    interface = ConfigItem("Window", "Interface", "ChineseSimplified",
                           OptionsValidator(intefaceTexts.values()), restart=True)

    # 窗口位置和大小(GUI 时期遗留,保留配置键以兼容已有 config.json)
    windowX = ConfigItem("Window", "X", None)
    windowY = ConfigItem("Window", "Y", None)
    windowW = ConfigItem("Window", "Width", 1200)
    windowH = ConfigItem("Window", "Height", 1200)

    # 使用一个配置项存储所有选区
    # 默认值为一个选区，格式为："ymin,ymax,xmin,xmax;ymin,ymax,xmin,xmax;..."，分号分隔不同选区
    subtitleSelectionAreas = ConfigItem("Main", "SubtitleSelectionAreas", "0.88,0.99,0.15,0.85")

    """
    MODE可选算法类型
    - InpaintMode.STTN_AUTO 智能擦除版
    - InpaintMode.STTN_DET 带字幕检测版, 无智能擦除
    - InpaintMode.LAMA 算法：对于动画类视频效果好，速度一般，不可以跳过字幕检测
    - InpaintMode.PROPAINTER 算法： 需要消耗大量显存，速度较慢，对运动非常剧烈的视频效果较好
    """
    # 【设置inpaint算法】
    inpaintMode = ConfigItem("Main", "InpaintMode", InpaintMode.STTN_AUTO,
                             OptionsValidator(InpaintMode), EnumSerializer(InpaintMode))

    subtitleDetectMode = ConfigItem("Main", "SubtitleDetectMode", SubtitleDetectMode.PP_OCRv5_SERVER,
                                    OptionsValidator(SubtitleDetectMode), EnumSerializer(SubtitleDetectMode))

    # 【设置像素点偏差】
    # 用于判断是不是非字幕区域(一般认为字幕文本框的长度是要大于宽度的，如果字幕框的高大于宽，且大于的幅度超过指定像素点大小，则认为是错误检测)
    subtitleYXAxisDifferencePixel = ConfigItem("Main", "SubtitleYXAxisDifferencePixel", 10, RangeValidator(0, 300))
    # 用于放大mask大小，防止自动检测的文本框过小，inpaint阶段出现文字边，有残留
    subtitleAreaDeviationPixel = ConfigItem("Main", "SubtitleAreaDeviationPixel", 10, RangeValidator(1, 300))
    # 同于判断两个文本框是否为同一行字幕，高度差距指定像素点以内认为是同一行
    subtitleAreaYAxisDifferencePixel = ConfigItem("Main", "SubtitleAreaYAxisDifferencePixel", 20, RangeValidator(0, 300))
    # 用于判断两个字幕文本的矩形框是否相似，如果X轴和Y轴偏差都在指定阈值内，则认为时同一个文本框
    subtitleAreaPixelToleranceYPixel = ConfigItem("Main", "SubtitleAreaPixelToleranceYPixel", 20, RangeValidator(0, 300))
    subtitleAreaPixelToleranceXPixel = ConfigItem("Main", "SubtitleAreaPixelToleranceXPixel", 20, RangeValidator(0, 300))
    subtitleTimelineBackwardFrameCount = ConfigItem("Main", "SubtitleTimelineBackwardFrameCount", 3, RangeValidator(0, 300))
    subtitleTimelineForwardFrameCount = ConfigItem("Main", "subtitleTimelineForwardFrameCount", 3, RangeValidator(0, 300))

    # 以下参数仅适用STTN算法时，才生效
    # 参考帧步长
    sttnNeighborStride = ConfigItem("Sttn", "NeighborStride", 5, RangeValidator(1, 100))
    # 参考帧数量
    sttnReferenceLength = ConfigItem("Sttn", "ReferenceLength", 10, RangeValidator(1, 100))
    # 设置STTN算法最大同时处理的帧数量
    sttnMaxLoadNum = ConfigItem("Sttn", "MaxLoadNum", 50, RangeValidator(1, 300))

    def getSttnMaxLoadNum(self):
        return max(self.sttnMaxLoadNum.value,
                   self.sttnNeighborStride.value * self.sttnReferenceLength.value)

    # 以下参数仅适用PROPAINTER算法时，才生效
    # 【根据自己的GPU显存大小设置】最大同时处理的图片数量，设置越大处理效果越好，但是要求显存越高
    propainterMaxLoadNum = ConfigItem("ProPainter", "MaxLoadNum", 70, RangeValidator(1, 300))

    # 是否使用硬件加速
    hardwareAcceleration = ConfigItem("Main", "HardwareAcceleration", HARDWARD_ACCELERATION_OPTION, BoolValidator())

    # 启动时检查应用更新
    checkUpdateOnStartup = ConfigItem("Main", "CheckUpdateOnStartup", True, BoolValidator())

    # 视频保存目录
    saveDirectory = ConfigItem("Main", "SaveDirectory", "")

    def __init__(self):
        self._items = _items(Config)
        self._config_file = None

    def set(self, item, value):
        """设置配置项:校验通过则生效并写盘;非法值回退默认值。"""
        if not self.validator_ok(item, value):
            item.value = item.defaultValue
        else:
            item.value = value
        if self._config_file:
            self.save(self._config_file)

    @staticmethod
    def validator_ok(item, value):
        return item.validator is None or item.validator.validate(value)

    def load(self, path):
        """从 json 加载配置;键缺失或值非法时保持默认值。"""
        self._config_file = path
        if not os.path.exists(path):
            return
        with open(path, encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                return
        if not isinstance(data, dict):
            return
        for item in self._items.values():
            raw = data.get(item.group, {}).get(item.key)
            if raw is None:
                continue
            value = item.serializer.deserialize(raw) if item.serializer else raw
            if self.validator_ok(item, value):
                item.value = value

    def save(self, path):
        """全部配置项序列化写盘(按 group 分组)。"""
        data = {}
        for item in self._items.values():
            data.setdefault(item.group, {})[item.key] = item.serialize()
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)


CONFIG_FILE = 'config/config.json'
config = Config()
config.load(CONFIG_FILE)

# 向后兼容：旧的 SubtitleDetectMode 枚举值为中文，迁移为新值
_detect_mode_value = config.subtitleDetectMode.value
if isinstance(_detect_mode_value, str) and _detect_mode_value in ("快速", "Fast"):
    config.set(config.subtitleDetectMode, SubtitleDetectMode.PP_OCRv5_MOBILE)
elif isinstance(_detect_mode_value, str) and _detect_mode_value in ("精准", "Precise"):
    config.set(config.subtitleDetectMode, SubtitleDetectMode.PP_OCRv5_SERVER)

# 读取界面语言配置
tr = configparser.ConfigParser()

TRANSLATION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'interface', f"{config.interface.value}.ini")
tr.read(TRANSLATION_FILE, encoding='utf-8')

# 项目的base目录
BASE_DIR = str(Path(os.path.abspath(__file__)).parent)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
