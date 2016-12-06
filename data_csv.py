import pickle

train_x = pickle.load(open('data/train_x.pkl', 'rb'))
train_y = pickle.load(open('data/train_y.pkl', 'rb'))
with open('data/train_csv.csv', 'w') as f:
    f.write('x1,x2,y' +"\n")
    for x,y in zip(train_x, train_y):
        f.write(str(x[0]) + "," + str(x[1]) + "," +str(y) +"\n")



train_x = pickle.load(open('data/test_x.pkl', 'rb'))
train_y = pickle.load(open('data/test_y.pkl', 'rb'))
with open('data/test_csv.csv', 'w') as f:
    f.write('x1,x2,y' +"\n")
    for x,y in zip(train_x, train_y):
        f.write(str(x[0]) + "," + str(x[1]) + "," +str(y) +"\n")