import matplotlib.pyplot as plt
def plloss(accgroup,lossgroup):
    x1 = range(len(accgroup))
    plt.title('training process')
    plt.plot(x1, accgroup, color='green', label='training accuracy')
    plt.plot(x1, lossgroup, color='skyblue', label='loss')
    plt.legend()  # 显示图例
    plt.xlabel('iteration times')
    plt.ylabel('rate')
    plt.show()
