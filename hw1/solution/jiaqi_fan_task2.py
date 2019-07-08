from pyspark import SparkContext
import sys, json, time

sc = SparkContext("local[*]", "inf553_hw1")

review = sc.textFile(sys.argv[1]).cache()
business = sc.textFile(sys.argv[2]).cache()
questionA = open(sys.argv[3], "w")
questionB = open(sys.argv[4], "w")

rev = review.map(lambda record: (json.loads(record)["business_id"], (json.loads(record)["stars"], 1))).reduceByKey(lambda x, y: (x[0]+y[0], x[1]+y[1]))
bus = business.map(lambda record: (json.loads(record)["business_id"], json.loads(record)["state"]))
state = bus.join(rev).map(lambda record: record[1]).reduceByKey(lambda x, y: (x[0]+y[0], x[1]+y[1])).map(lambda x: (x[0], x[1][0]/x[1][1])).sortByKey().sortBy(lambda x: x[1], ascending=False)

startM1 = time.time()
m1 = state.collect()
print(m1[0], m1[1], m1[2], m1[3], m1[4])
duringM1 = time.time() - startM1

startM2 = time.time()
m2 = state.take(5)
print(m2)
duringM2 = time.time() - startM2

print("state,stars", file=questionA)
for i in m1:
    print("{0},{1}".format(i[0], i[1]), file=questionA)

q2Str = {}
q2Str["m1"] = duringM1
q2Str["m2"] = duringM2
q2Str["explanation"] = "The collect() function needs to convert the entire dataset into an array. However, take() function gets and converts first n data of the dataset into an array. which has smaller computation."

print(json.dumps(q2Str), file=questionB)

questionA.close()
questionB.close()
