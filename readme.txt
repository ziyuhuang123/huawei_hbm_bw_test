在GPU的HBM访问中存在合并访存与非合并访存，这里测试了strided reading模式，单个线程访问one_step byte的数据，下一个线程跨步one_step的整数倍，再访问one_step byte长度的数据。这里我们测试one_step=8， 32， 64byte，跨步长度当然是one_step的整数倍。在A100和H100上做了测试。发现带宽的下降明显存在平台现象。为防止nvcc过度优化，在寄存器获取global值之后又写入smem。（似乎不写入smem好像也可以）

运行的命令为bash test.sh。注意修改test.sh里面架构号。然后得到results.csv，然后运行python plot.py得到折线图。