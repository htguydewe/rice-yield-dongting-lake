# Code

代码按用途分为四类，论文最终建模主线为 `modeling/run_bilstm_comparison_numbered.py`，对应运行轮次 `run_111`：

- `data_preparation/`：从原始遥感、气象、地形和统计资料构建模型输入表。
- `modeling/`：训练和比较论文最终使用的 RF、LSTM、Bi-LSTM 三类模型。
- `visualization/`：生成论文图表、空间分布图、Taylor 图、特征重要性图等。
- `legacy/`：早期实验脚本，用于追溯方法演进，不作为首选运行入口。

部分脚本保留了原始工作目录绝对路径。如果在新机器上运行，需要将路径改为本仓库的相对路径，或通过脚本参数传入数据目录。
