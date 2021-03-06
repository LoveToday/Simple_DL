# Simple_DL
实现最简单的深度学习,二分类，判断一张图片是否含有猫


实现步骤
1.读取 

    a.引入h5py

        pip3 install h5py

    b.开始读取

        train_data_set = h5py.File('train_catvnoncat.h5', 'r')
        test_data_set = h5py.File('test_catvnoncat.h5', 'r')
        
    c.处理成自己所需要的
        for key in train_data_set.keys():
        print(key)

        train_set_x = train_data_set['train_set_x']
        train_labels_y = train_data_set['train_set_y']

        test_set_x = test_data_set['test_set_x']
        test_labels_y = test_data_set['test_set_y']
        
        # 维度处理
        m_train = train_set_x.shape[0]
        m_test = test_set_x.shape[0]

        # 取出数据集,训练集
        train_data_org = train_set_x[:]
        test_data_org = test_set_x[:]

        train_labels_org = train_labels_y[:]
        test_labels_org = test_labels_y[:]

        # 重新设置维度
        train_data_tran = train_data_org.reshape(m_train,-1).T
        test_data_tran = test_data_org.reshape(m_test,-1).T


        train_labels_tran = train_labels_org[np.newaxis, :]
        test_labels_tran = test_labels_org[np.newaxis, :]

        # 标准化数据
        train_data_sta = train_data_tran/255
        test_data_sta = test_data_tran/255
        
2.定义sigmoid函数

    def sigmoid(z):
        a = 1/(1 + np.exp(-z))
        return a
        
初始化参数
    
    # 初始化参数
    n_dim = train_data_sta.shape[0]
    w = np.zeros((n_dim, 1))
    b = 0
    
3.定义前向传播函数，代价函数以及梯度下降

    def propagate(w, b, x, y):
        # 前向传播函数
        z = np.dot(w.T, x) + b
        A = sigmoid(z)

        m = x.shape[1]

        # 代价函数
        J = (-1/m) * np.sum(y * np.log(A) + (1 - y) * np.log(1 - A))

        # 梯度下降
        dw = 1/m * np.dot(x, (A-y).T)
        db = 1/m * np.sum(A - y)

        grands = { 'dw': dw, 'db': db }
        return grands, J
        
 4.拟合优化w与b的值
 
    def optimize(w, b, x, y, alpha, n_iters):  
        costs = [] 
        for i in range(n_iters):
            grands, J = propagate(w, b, x, y)
            dw = grands['dw']
            db = grands['db']

            w = w - alpha * dw
            b = b - alpha *db
            if i % 100 == 0:
                costs.append(J)

        grands = { 'dw': dw, 'db': db }
        parms = { 'w': w, 'b': b }
        return grands, parms, costs
    
 5.根据w与b的值预测一张或多张图片是否是一只猫
 
    def predict(w,b,x_test):
        z = np.dot(w.T, x_test) + b
        A = sigmoid(z)

        m = x_test.shape[1]
        y_pred = np.zeros((1, m))

        for i in range(m):
            if A[:,i] > 0.5:
                y_pred[:,i] = 1
            else:
                y_pred[:,i] = 0

        return y_pred
        
 6.模型整合
 
    def model(w,b,x_train,y_train,x_test,y_test,alpha,n_iters):
        grands, parms, costs = optimize(w, b, x_train, y_train, alpha, n_iters)

        w = parms['w']
        b = parms['b']

        y_pred_train = predict(w,b,x_train)
        y_pred_test = predict(w,b,x_test)

        print('the train acc is', np.mean(y_pred_train == y_train) * 100, '%')
        print('the test acc is', np.mean(y_pred_test == y_test) * 100, '%')

        b = {
            'w': w,
            'b': b,
            'y_pred_train': y_pred_train,
            'y_pred_test': y_pred_test,
            'alpha': alpha,
            'costs': costs
        }
        return b
 
7.调用模型

    b = model(w,b,train_data_sta,train_labels_tran,test_data_sta,test_labels_tran,alpha=0.005,n_iters=400)



8.测试数据
改变图片的大小，导入skimage, from skimage import transform

    python3 -m pip install scikit-image
    
    print(image.shape)
    image_tran = transform.resize(image, (64,64,3)).reshape(64*64*3,1)
    print(image_tran.shape)

    print('++++++w',b['w'], b['b'])

    # 直接预测实验
    y_pred = predict(b['w'],b['b'],image_tran)

    print(y_pred[:,0])
    
    
