import numpy as np
import tensorflow as tf
import pandas as pd

# 读训练数据
data = pd.read_csv('data/train.csv')

# 数据预处理
data['Sex'] = data['Sex'].apply(lambda s: 1 if s == "male" else 0) # male:1, female:0
# 缺失字段添0
data = data.fillna(0)
# 选取特征[性别，年龄，客舱等级，兄弟姐妹和配偶在船数量，父母孩子在船数量，船票价格]
dataSet_X = data[['Sex','Age','Pclass','SibSp','Parch','Fare']].as_matrix()

# 建立标签数据集
data['Deceased'] = data['Survived'].apply(lambda s: 1 - s)
dataSet_Y = data[['Deceased', 'Survived']].as_matrix()

# 定义计算图

# 定义占位符
x = tf.placeholder(tf.float32, shape=[None, 6]) # None代表可以输入任意条6元数据
y = tf.placeholder(tf.float32, shape=[None, 2]) # None代表可以输入任意条2元数据

# 使用逻辑回归算法

# 参数权重
weights = tf.Variable(tf.random_normal([6, 2]), name="weights")
# 偏移量
bias = tf.Variable(tf.zeros([2]), name='bias')

y_pred = tf.nn.softmax(tf.matmul(x,weights) + bias)

# 定义交叉熵
cross_entropy = - tf.reduce_sum(y * tf.log(y_pred + 1e-10), reduction_indices=1)
# 定义损失函数
cost = tf.reduce_mean(cross_entropy)

# 使用梯度下降优化算法最小化损失函数
train_op = tf.train.GradientDescentOptimizer(0.001).minimize(cost)

with tf.Session() as sess:
    # 变量初始化
    sess.run(tf.global_variables_initializer())

    # 训练模型
    for epoch in range(50):
        total_loss = 0.
        for i in range(len(dataSet_X)):
            # prepare feed data and run
            feed_dict = {x: [dataSet_X[i]],y: [dataSet_Y[i]]}
            _, loss = sess.run([train_op, cost], feed_dict=feed_dict)
            total_loss += loss
        # display loss per epoch
        print('Epoch: %04d, total loss=%.9f' % (epoch + 1, total_loss))
    print("Train Complete")

    # 读测试数据
    testdata = pd.read_csv('data/test.csv')
    # 数据清洗，数据预处理
    testdata = testdata.fillna(0)
    testdata['Sex'] = testdata['Sex'].apply(lambda s: 1 if s == 'male' else 0)
    # 特征选择
    X_test = testdata[['Sex', 'Age', 'Pclass', 'SibSp', 'Parch', 'Fare']]
    # 评估模型
    predictions = np.argmax(sess.run(y_pred, feed_dict={x: X_test}), 1)
    # 保存结果
    submission = pd.DataFrame({
        'PassengerId': testdata['PassengerId'],
        'Survived': predictions
    })
    submission.to_csv("submission",index=False)



