'''
Created on Aug 8, 2016

@author: he8819197
'''
import scipy.sparse as sp
import numpy as np
import pandas as pd

class Dataset(object):
    '''
    classdocs
    '''

    def __init__(self, path):
        '''
        Constructor
        '''
        #self.trainMatrix = self.load_rating_file_as_matrix(path + ".train.rating")
        #self.testRatings = self.load_rating_file_as_list(path + ".test.rating")
        self.path = path
        """path_file_train = path + '/train.csv'
        f = open(path_file_train, 'r')
        line = f.readline()
        parse = line.strip().split(',')
        self.i_uid = np.where(parse == 'user_id')[0][0]
        self.i_order_num = np.where(parse == 'order_number')[0][0]
        self.i_dow = np.where(parse == 'order_dow')[0][0]
        self.i_hod = np.where(parse == 'order_hour_of_day')[0][0]
        self.i_days_prior = np.where(parse == 'days_since_prior_order')[0][0]
        self.i_pid = np.where(parse == 'product_id')[0][0]
        self.i_aid = np.where(parse == 'aisle_id')[0][0]
        self.i_did = np.where(parse == 'department_id')[0][0]
        f.close()"""

        #
        #self.testNegatives = self.load_negative_file(path + ".test.negative")
        #assert len(self.testRatings) == len(self.testNegatives)
        
        #self.num_users, self.num_items = self.trainMatrix.shape

    def get_user_item_features_test(self, samples=None):


        df = pd.read_csv(self.path + '/products.csv')
        df = df.drop('product_name', 1)
        # print list(df)
        arr = df.values
        if samples is not None:
            sample_inter = np.random.choice(np.arange(0,samples,1), 100, replace=False)
            items_inter= arr[sample_inter, :]
            items_features = arr[:samples]
        else:
            items_features = arr
        del df
        del arr

        f = open(self.path+'/train.csv', 'r')
        line = f.readline()
        parse = np.array(line.strip().split(','))
        i_uid = np.where(parse == 'user_id')[0][0]
        i_order_num = np.where(parse == 'order_number')[0][0]
        i_dow = np.where(parse == 'order_dow')[0][0]
        i_hod = np.where(parse == 'order_hour_of_day')[0][0]
        i_days = np.where(parse == 'days_since_prior_order')[0][0]
        i_pid = np.where(parse == 'product_id')[0][0]
        i_aid = np.where(parse == 'aisle_id')[0][0]
        i_did = np.where(parse == 'department_id')[0][0]
        f.close()

        df = pd.read_csv(self.path+'/train.csv')
        #print list(df)
        #df.drop('order_id', 1)
        #df.drop('add_to_cart_order', 1)
        #df.drop('reordered', 1)
        arr = df.values
        test_set = arr[:, [i_uid,i_order_num,i_dow,i_hod,i_days,i_pid,i_aid,i_did]] #[[arr[:,i_uid], arr[:, [i_order_num,i_dow,i_hod,i_days]]], [arr[:,i_pid], arr[:,[i_aid,i_did]]]]
        del df
        del arr
        if samples is not None:
            check = True
            for i in items_inter:
                arr_index = np.where(test_set[:,5]==i[0])[0]
                if len(arr_index) > 0:
                    #print len(arr_index)
                    if check:
                        samples_test = test_set[arr_index, :]
                        check = False
                    else:
                        samples_test = np.concatenate((samples_test, test_set[arr_index, :]), axis=0)
            test_set = samples_test
        #print test_set.shape

        path_file_user_test = self.path + '/orders.csv'
        df = pd.read_csv(path_file_user_test)
        df = df.loc[(df['eval_set'] == 'train')]
        df.days_since_prior_order = df.days_since_prior_order.fillna(-1.)
        df = df.drop('order_id', 1)
        df = df.drop('eval_set', 1)
        arr = df.values
        if samples is not None:
            #sample_user = samples[0]
            test_users = test_set[:, [0,1,2,3,4]]
            users_features = np.array([test_users[0]])
            for u in test_users:
                find = np.flatnonzero((users_features==u).all(1))
                if len(find) == 0:
                    users_features = np.concatenate((users_features, np.array([u])), axis=0)
        else:
            users_features = arr  # [arr[:,0], arr[:, 1:]]
        # print list(df)
        """indexs = list(df.index)
        users = [[],[]]
        for index in indexs:
            user_id = df.user_id.values[index]
            order_num = df.order_number.values[index]
            dow = df.order_dow.values[index]
            hod = df.order_hour_of_day.values[index]
            days = df.days_since_prior_order.values[index]
            users[0].append(user_id)
            users[1].append([order_num, dow, hod, days])"""
        del df

        return (test_set, users_features, items_features)

    def get_num_users(self):
        path_file = self.path+'/orders.csv'
        df = pd.read_csv(path_file)
        users = df.user_id.values
        return len(list(set(users)))

    def get_num_items(self):
        path_file = self.path + '/products.csv'
        df = pd.read_csv(path_file)
        items = df.product_id.values
        return len(items)

    def get_order_products_as_list(self):
        path_file = self.path + '/order_products__prior.csv'
        fp = open(path_file, 'r')
        line = fp.readline()
        parse = np.array(line.strip().split(','))
        i_oid = np.where(parse=='order_id')[0][0]
        i_pid = np.where(parse=='product_id')[0][0]
        line = fp.readline()
        list_order_products = [[],[]]
        while line:
            parse = line.strip().split(',')
            order_id, product_id = int(parse[i_oid]), int(parse[i_pid])
            if order_id not in list_order_products[0]:
                list_order_products[0].append(order_id)
                list_order_products[1].append([product_id])
            else:
                index = list_order_products[0].index(order_id)
                list_order_products[1][index].append(product_id)
            line = fp.readline()
        fp.close()
        return list_order_products


    def load_rating_file_as_list(self, filename):
        ratingList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item = int(arr[0]), int(arr[1])
                ratingList.append([user, item])
                line = f.readline()
        return ratingList

    def load_negative_file(self, filename):
        negativeList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                negatives = []
                for x in arr[1: ]:
                    negatives.append(int(x))
                negativeList.append(negatives)
                line = f.readline()
        return negativeList

    def load_rating_file_as_matrix(self, filename):
        '''
        Read .rating file and Return dok matrix.
        The first line of .rating file is: num_users\t num_items
        '''
        # Get number of users and items
        num_users, num_items = 0, 0
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u)
                num_items = max(num_items, i)
                line = f.readline()
        # Construct matrix
        mat = sp.dok_matrix((num_users, num_items), dtype=np.float32)
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
                if (rating > 0):
                    mat[user, item] = 1.0
                line = f.readline()
        return mat

    def load_mini_batch(self, f, batch_size_user):
        #end_file = False
        list_user_ids = list()
        mini_batch = list()
        while len(list_user_ids) != batch_size_user:
            line = f.readline()
            if (line is None) or (line==''):
                #end_file = True
                break
            parse = line.strip().split(',')
            user_id = int(parse[1])
            if user_id not in list_user_ids:
                list_user_ids.append(user_id)
            # add label
            parse.append('1')
            mini_batch.append(np.array(parse, dtype=float))
        return np.array(mini_batch) #, end_file


    """def load_mini_batch(self, f, batch_size_users):
        user_ids = list()
        end_file = False
        for i in range(batch_size_users):
            line = f.readline()
            if (line == None) or (line == ''):
                end_file = True
                break
            user_ids.append(int(line.strip()))

        # load features of user
        orders = pd.read_csv(self.path+'/orders.csv')
        orders = orders.loc[(orders['eval_set'] == 'prior')]
        check = True
        for user_id in user_ids:
            if check:
                orders_users = orders.loc[(orders['user_id'] == user_id)]
                check = False
            else:
                useri = orders.loc[(orders['user_id'] == user_id)]
                orders_users = orders_users.append(useri, ignore_index=True)

        del orders
        orders_users.drop('eval_set', 1)
        orders_users.days_since_prior_order = orders_users.days_since_prior_order.fillna(-1.)

        df = pd.read_csv(self.path+'/order_products__prior.csv')
        indexs = list(orders_users.index)
        list_order_products = list()
        user_input = list()
        item_input = list()
        for index in indexs:
            order_id = orders_users.order_id.values[index]
            user_id = orders_users.user_id.values[index]
            order_number = orders_users.order_number.values[index]
            dow = orders_users.order_dow.values[index]
            hod = orders_users.order_hour_of_day.values[index]
            days = orders_users.days_since_prior_order.values[index]
            user_features = np.array([user_id, order_number, dow, hod, days])

            df_order_id = df.loc[(df['order_id'] == order_id)]
            list_products = df_order_id.product_id.values
            list_order_products.append([user_features, list_products])
            for i in range(len(list_products)):
                user_input.append(user_features)
                aid = df_order_id.aisle_id.values[i]
                did = df_order_id.department_id.values[i]
                item_features = np.array([list_products[i], aid, did])
                item_input.append(item_features)
        #print len(user_input)
        return (np.array(user_input), np.array(item_input), list_order_products, end_file)"""


if __name__ == '__main__':
    dataset = Dataset('dataset')
    test_set, users, items = dataset.get_user_item_features_test()
    print test_set.shape
    print users.shape
    print items.shape
    """fp = open('dataset/users.txt')
    i = 0
    while True:
        i += 1
        print '__mini__batch__: ', i
        user_input, item_input, orders_products, end_file = dataset.load_mini_batch(fp, 200)
        print user_input.shape
        print item_input.shape
        print len(orders_products)
        if end_file:
            break
    fp.close()"""
