from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS, Rating
import sys, time, math


def correct_rating(value):
    if value < 1:
        return 1.0
    elif value < 5:
        return value
    else:
        return 5.0


start = time.time()
sc = SparkContext("local[*]", "inf553_hw3")
train = sc.textFile(sys.argv[1]).cache()
test = sc.textFile(sys.argv[2]).cache()
case_id = int(sys.argv[3])


# train = sc.textFile("/Users/frank/Desktop/553/hw_data/hw3/yelp_train.csv").cache()
# test = sc.textFile("/Users/frank/Desktop/553/hw_data/hw3/yelp_val.csv")
train_header = train.first()
train_data = train.filter(lambda record: record != train_header).map(lambda record: record.split(",")).cache()
test_header = test.first()
test_data = test.filter(lambda record: record != test_header).map(lambda record: record.split(",")).cache()

if case_id == 1:
    t = test_data.map(lambda x: (x[0], x[1]))
    user_business = train_data.map(lambda x: (x[0], x[1])).union(t)
    user = user_business.map(lambda x: (x[0], 1)).reduceByKey(lambda x, y: x + y).keys()
    business = user_business.map(lambda x: (x[1], 1)).reduceByKey(lambda x, y: x + y).keys()
    user_map = {}
    id_user = {}
    business_map = {}
    id_business = {}

    for user_id, user_name in enumerate(user.collect()):
        user_map[user_name] = user_id
        id_user[user_id] = user_name

    for business_id, business_name in enumerate(business.collect()):
        business_map[business_name] = business_id
        id_business[business_id] = business_name

    test_pair = test_data.map(lambda record: (user_map[record[0]], business_map[record[1]]))
    train_rating = train_data.map(lambda record: Rating(user_map[record[0]], business_map[record[1]], float(record[2])))

    rank = 5
    numIterations = 10
    lambda_num = 0.3
    model = ALS.train(train_rating, rank, numIterations, lambda_num)

    predictions = model.predictAll(test_pair).map(lambda r: ((id_user[r[0]], id_business[r[1]]), r[2]))

    output = open(sys.argv[4], "w")
    # output = open("output.csv", "w")
    predict_result = sorted(predictions.collect())

    print("user_id, business_id, prediction", file=output)
    for ele in predict_result:
        print(ele[0][0] + "," + ele[0][1] + "," + str(correct_rating(ele[1])), file=output)

    output.close()
    rate_system = test_data.map(lambda record: ((record[0], record[1]), float(record[2]))).join(predictions).map(lambda record: abs(record[1][0]-record[1][1])).map(lambda x: x*x).mean()
    rmse = math.sqrt(rate_system)


if case_id == 2:
    user_ave_value = train_data.map(lambda record: (record[0], (float(record[2]), 1))).reduceByKey(lambda x, y: (x[0]+y[0], x[1]+y[1])).map(lambda x: (x[0], (x[1][0]/x[1][1]))).collectAsMap()
    business_ave_value = train_data.map(lambda x: (x[1], (float(x[2]), 1))).reduceByKey(lambda x, y: (x[0]+y[0], x[1]+y[1])).map(lambda x: (x[0], x[1][0]/x[1][1])).collectAsMap()
    user_value = train_data.map(lambda x: (x[0], (x[1], float(x[2]) - user_ave_value[x[0]]))).groupByKey().mapValues(dict).collectAsMap()

    business_alluser = train_data.map(lambda x: (x[1], x[0])).groupByKey().mapValues(set).collectAsMap()
    user_allbusiness = train_data.map(lambda x: (x[0], x[1])).groupByKey().mapValues(set).collectAsMap()

    test_pair = test_data.map(lambda x: (x[0], x[1])).collect()

    predictions = []
    for pairs in test_pair:
        if pairs[1] not in business_alluser:
            predictions.append((pairs[0], pairs[1], user_ave_value[pairs[0]]))
        elif pairs[0] not in user_allbusiness:
            predictions.append((pairs[0], pairs[1], business_ave_value[pairs[1]]))
        else:
            if pairs[1] in user_value[pairs[0]]:
                predictions.append((pairs[0], pairs[1], user_value[pairs[0]][pairs[1]] + user_ave_value[pairs[0]]))
            else:
                correlation = []
                for ele in business_alluser[pairs[1]]:
                    co_items = user_allbusiness[pairs[0]].intersection(user_allbusiness[ele])
                    sum_num = 0
                    de1 = 0
                    de2 = 0
                    for item in co_items:
                        sum_num += user_value[pairs[0]][item] * user_value[ele][item]
                        de1 += user_value[pairs[0]][item] * user_value[pairs[0]][item]
                        de2 += user_value[ele][item] * user_value[ele][item]
                    co_num = 0
                    if de1 != 0 and de2 != 0:
                        co_num = sum_num / math.sqrt(de1 * de2)

                    correlation.append((co_num, user_value[ele][pairs[1]]))

                correlation = sorted(correlation)
                N = 30
                if len(correlation) > N:
                    for i in range(len(correlation) -N):
                        del correlation[0]

                upper_sum = 0
                lower_sum = 0
                for num in correlation:
                    upper_sum += float(num[0]) * num[1]
                    lower_sum += abs(float(num[0]))

                pred = user_ave_value[pairs[0]]
                if lower_sum != 0:
                    pred = upper_sum/lower_sum + user_ave_value[pairs[0]]

                predictions.append((pairs[0], pairs[1], correct_rating(pred)))

    output = open(sys.argv[4], "w")
    # output = open("output.csv", "w")
    print("user_id, business_id, prediction", file=output)
    for ele in predictions:
        print(ele[0], "," + ele[1], "," + str(ele[2]), file=output)
    output.close()

    ddddd = sc.parallelize(predictions).map(lambda x: ((x[0], x[1]), float(x[2])))
    rate_system = test_data.map(lambda record: ((record[0], record[1]), float(record[2]))).join(ddddd).map(lambda record: abs(record[1][0]-record[1][1])).map(lambda x: x*x).mean()
    rmse = math.sqrt(rate_system)


if case_id == 3:
    business_alluser = train_data.map(lambda x: (x[1], x[0])).groupByKey().mapValues(set).collectAsMap()
    user_allbusiness = train_data.map(lambda x: (x[0], x[1])).groupByKey().mapValues(set).collectAsMap()
    get_value = train_data.map(lambda x: ((x[0], x[1]), float(x[2]))).collectAsMap()

    def cold_start(x):
        if x[0] not in user_allbusiness or x[1] not in business_alluser:
            return ()
        else:
            return (x[0] ,x[1])


    def item_weight(a, b, c):
        set_a = business_alluser[a]
        set_b = business_alluser[b]
        coreted_item = (set_a & set_b) - {c}
        if len(coreted_item) == 0:
            return 0
        else:
            a_list = []
            b_list = []
            for x in coreted_item:
                a_list.append(get_value[(x, a)])
                b_list.append(get_value[(x, b)])
            A_ave = sum(a_list) / len(a_list)
            B_ave = sum(b_list) / len(b_list)

            numerator = 0
            A_square = 0
            B_square = 0
            for i in range(len(coreted_item)):
                numerator += (a_list[i] - A_ave) * (b_list[i] - B_ave)
                A_square += (a_list[i] - A_ave) ** 2
                B_square += (b_list[i] - B_ave) ** 2

            denominator = math.sqrt(A_square) * math.sqrt(B_square)
            if denominator == 0:
                return None
            return numerator / denominator


    def item(x):
        sum_all = 0
        count = 0
        for item in business_alluser[x[1]]:
            if item == x[0]:
                continue
            sum_all += get_value[(item, x[1])]
            count += 1

        ave = 0.0
        if count != 0:
            ave = float(sum_all / count)
        co_rated = []
        for ele in user_allbusiness[x[0]]:
            if ele == x[1]:
                continue
            co = item_weight(x[1], ele, x[0])
            if co == None:
                return (x, ave)
            co_rated.append([x[0], ele, co])
        co_rated.sort(key=lambda x: x[2], reverse=True)

        length = 5
        if length > len(co_rated):
            length = len(co_rated)

        up_list = []
        for i in range(length):
            up_list.append(get_value[(co_rated[i][0], co_rated[i][1])] * co_rated[i][2])
        up_num = sum(up_list)
        lo_list = []
        for i in range(length):
            lo_list.append(abs(co_rated[i][2]))
        lo_num = sum(lo_list)

        pre_value = ave
        if lo_num != 0:
            pre_value = ave + up_num / lo_num
        return (x, pre_value)


    process_data = test_data.map(cold_start).filter(lambda x: len(x) != 0)
    not_exist_data = test_data.map(lambda x: ((x[0], x[1]), 3)).subtractByKey(process_data.map(lambda x: (x, 1)))
    predictions = process_data.map(lambda x: item(x))
    result = sc.union([predictions, not_exist_data])

    output = open(sys.argv[4], "w")
    print("user_id, business_id, prediction", file=output)
    for ele in result.collect():
        print(ele[0][0], "," + ele[0][1], "," + str(ele[1]), file=output)
    output.close()

    rate_system = test_data.map(lambda x: ((x[0], x[1]), float(x[2]))).join(result).map(lambda record: abs(record[1][0]-record[1][1])).map(lambda x: x*x).mean()
    rmse = math.sqrt(rate_system)


duration = time.time() - start
print("Duration: ", duration)
print("RMSE: ", str(rmse))


