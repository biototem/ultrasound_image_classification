# 超声影像良恶性分析
前言（无）  

projectA_seg 原理，输入图像，输出类别图，得到良性像素数量和恶性像素数量，
如果恶性像素总数大于恶性像素与良性像素之和的百分之50，该图判断为恶性，否则为良性  

projectA_simple 原理，输入图像到几个不同输入尺度的网络，
分别得到各个网络的预测结果，再联合三个网络的结果得到最终的结果  

### 简写解释
u 普超  
eu 彩超  
cm 混淆矩阵  
lr 学习率  
prefix 命名前缀  

# 依赖
pytorch >= 1.1  
numpy  
tensorboardX  
scikit-image  
opencv-python >= 4.0  
seaborn  
matplotlib  
imageio  

# 快速上手
1. 指定原始数据集文件夹和标签文件的位置  
    打开 a_config.py  
    找到 ori_dataset_path 和 ori_label_path 变量位置，进行修改  
    
    示例数据集格式如下  
    观察 breastSWE 目录  
    数据集根目录/医生名/良恶性类别/图像名-超声类别.tif  
    
    示例标签格式如下  
    观察 ultrasound_grouping.csv 文件格式  
    第一列为 是否属于测试数据  
    第二列为 文件名  
    第三列为 是否属于恶性  

2. 构造数据集
    依次运行以下程序  
    make_dataset_step_1.py  
    make_dataset_step_2.py  
    make_dataset_step_3_crop_img.py  
    构造数据集时可能需要3倍于原数据集的硬盘空间  
    不过 out_dir1 和 out_dir2 在数据集构造完成后可以直接删除  
    
    你可以修改 a_config.py 中 out_dir1，out_dir2，out_dir3 变量  
    可以修改中间文件储存的目录  

3. 修改训练超参数  
    打开 a_config.py，找到以下变量并修改，不过一般只需要修改batch_size就行  
    device：选择再哪张显卡或CPU上运行模型  
    train_lr：设定训练时的学习率  
    epoch：设定训练轮次  
    batch_size：设定训练批大小  

4. 开始训练  
    进入 projectA_seg 文件夹，运行 _train.py 则直接训练模型  
    进入 projectA_simple 文件夹，运行 _train.py 则直接训练模型  
    
    训练日志存档分别储存在以 logs_ 为前缀的文件夹中，你可以使用 tensorboard 来实时观察训练情况  
    projectA_seg 的 save 文件夹中储存着当前验证F1分数最高的模型，相关信息和混淆矩阵  
    projectA_simple 因为有三个不同尺度的模型，不同尺度的模型的存档分别储存在 save.32 save.64 save.128 文件夹中  

5. 测试模型  
    训练完成后，直接运行 _test.py 开始使用测试集对模型进行测试    
    测试结果会同时输出到控制台上和储存到 save 文件夹中的 test.txt 文件中  

6. 输出每个样本的分数（可选）  
    直接运行对应的 _test_out_csv.py 程序，在 projectA_seg 或 projectA_simple   
    会生成以 prob_ 开头的文件，格式如下  
    projectA_seg 中，每列意义：数据集类型，文件名，预测的恶性概率，是否为恶性的标签  
    projectA_simple 中，每列意义：数据集类型，文件名，尺度为32时预测的恶性概率，尺度为64时预测的恶性概率，尺度为128时预测的恶性概率，是否为恶性的标签  

7. 对新的图像进行预测  
    将新的图像放入以 test_in 开头的文件夹中，注意 eu 代表彩超，u 代表普超。  
    projectA_seg 对图像要求：尽可能除去多余的白边，除去多余白边的图像若大于300x300，建议先对图像分成多个小于300x300的小块，再进行预测    
    projectA_simple 对图像要求：尽可能除去多余的白边, 对图像大小无要求  
    运行 _pred_with_new_img.py  
    输出结果会储存在以 test_out 开头的文件夹中  
    projectA_seg 可以额外输出热图  
    
    输出结果格式  
    
    projectA_seg 格式  
    benign_pixel_num 判断为良性的像素数量  
    bg_pixel_num 判断为背景的像素数量  
    malignant_pixel_num 判断为恶性的像素数量  
    pred 预测结果，0代表预测为良性，1代表预测为恶性  
    
    projectA_simple 格式  
    pred 预测结果，0代表预测为良性，1代表预测为恶性  
    pred_x32 尺度为32的网络的预测结果  
    pred_x64 尺度为64的网络的预测结果  
    pred_x128 尺度为128的网络的预测结果  
    prob_x32 尺度为32的网络的预测为恶性的概率  
    prob_x64 尺度为64的网络的预测为恶性的概率  
    prob_x128 尺度为128的网络的预测为恶性的概率  

8. 没了
