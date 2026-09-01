"""
功能：组织 CardioResp 4D MRI 的 DICOM 数据接口。
论文来源：Necessary adaptation: image-domain input handling for the local acquisition.
输入：DICOM 文件、manifest 和配置。
输出：扫描、清单和数据集模块。
主要步骤：提供数据子包命名空间。
是否属于原论文直接实现 / 必要适配 / 可选实验：Necessary adaptation.
命令行使用示例：python -m cardioresp4d.data.inspect_dataset --help
"""
