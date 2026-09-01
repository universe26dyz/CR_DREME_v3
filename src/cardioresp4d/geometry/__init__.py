"""
功能：提供 CardioResp 4D MRI Phase 1 的 DICOM 病人世界坐标工具。
论文来源：Necessary adaptation: DICOM patient-coordinate handling for local image-domain MRI.
输入：DICOM manifest 中的平面几何元数据。
输出：物理平面、受试者归一化和几何质控 API。
主要步骤：按 DICOM PS3.3 解释像素坐标，并暴露可复现的世界坐标工具。
是否属于原论文直接实现 / 必要适配 / 可选实验：Necessary adaptation.
命令行使用示例：python -m cardioresp4d.geometry.geometry_qc --help
"""
