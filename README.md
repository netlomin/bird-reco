# finetune-Alexnet-VGGnet
实现了一个鸟类识别的finetune操作，利用了两种深度网络 alexnet和vggnet
# 实验环境：
linux+pytorch

# 使用规范
NetWork.py：存储模型结构//
finetune_v2.py:  规范化化后的finetune版本，直接运行//
k-flod-finetune.py： 使用k折交叉验证的finetune，其中修改transorm的参数就可以进行数据增强操作，直接运行//

//
pl.py:   loss曲线的刻画
