from pyspark import SparkContext
import sys, json

sc = SparkContext("local[*]", "inf553_hw1")
review = sc.textFile(sys.argv[1]).cache()

A = review.filter(lambda record: json.loads(record)["useful"] > 0).map(lambda record: (json.loads(record)["useful"], 1)).reduceByKey(lambda x, y: x + y).values().sum()
B = review.filter(lambda record: json.loads(record)["stars"] == 5).map(lambda record: (json.loads(record)["stars"], 1)).reduceByKey(lambda x, y: x + y).first()[1]
C = review.map(lambda record: (len(json.loads(record)["text"]), 1)).reduceByKey(lambda x, y: x + y).keys().max()
user = review.map(lambda record: (json.loads(record)["user_id"], 1)).reduceByKey(lambda x, y: x + y).sortByKey().sortBy(lambda x: x[1], ascending=False)
D = user.count()
E = user.take(20)
business = review.map(lambda record: (json.loads(record)["business_id"], 1)).reduceByKey(lambda x, y: x + y).sortByKey().sortBy(lambda x: x[1], ascending=False)
F = business.count()
G = business.take(20)

jsonStr = {}
jsonStr["n_review_useful"] = A
jsonStr["n_review_5_star"] = B
jsonStr["n_characters"] = C
jsonStr["n_user"] = D
u = []
for ele in E:
    u.append([ele[0], ele[1]])
jsonStr["top20_user"] = u

jsonStr["n_business"] = F
b = []
for ele in G:
    b.append([ele[0], ele[1]])
jsonStr["top20_business"] = b

output = open(sys.argv[2], "w")
print(json.dumps(jsonStr), file=output)
output.close()
