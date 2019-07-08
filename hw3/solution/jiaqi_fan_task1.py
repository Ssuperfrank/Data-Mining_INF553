from pyspark import SparkContext, SparkConf
import sys, time, random, itertools

start_time = time.time()

conf = SparkConf().setAppName("inf553_hw3").setMaster('local[*]').set('spark.executor.memory', '4g').set('spark.driver.memory', '4g')
sc = SparkContext(conf=conf)


# sc = SparkContext("local[*]", "inf553_hw3","4g")
data = sc.textFile(sys.argv[1]).cache()
header = data.first()
new_data = data.filter(lambda record: record != header).map(lambda record: record.split(",")).cache()

user = new_data.map(lambda x: (x[0], 1)).reduceByKey(lambda x, y: x + y).keys()
business = new_data.map(lambda x: (x[1], 1)).reduceByKey(lambda x, y: x + y).keys()

user_num = user.count()
business_num = business.count()

user_map = {}
business_map = {}


for user_id, user_name in enumerate(user.collect()):
    user_map[user_name] = user_id

for business_id, business_name in enumerate(business.collect()):
    business_map[business_name] = business_id


band = 20
row = 2
n = band * row

singnature_matrix = []

ran = [71, 277, 691, 1699, 2087, 4211, 6329, 9043, 12613]

while n != 0:
    a = random.randint(2, 100)
    b = random.randint(2, 10)
    p = random.choice(ran)
    single_line = new_data.map(lambda x: (x[1],  (a * user_map[x[0]] + b) % p)).groupByKey().mapValues(list).map(lambda x: (x[0], min(x[1])))
    singnature_matrix.append(single_line.collectAsMap())
    n -= 1


hash_array = [1, 23, 133, 231, 591, 891, 1331, 2937, 3473, 5751, 8717, 13321]
candidate = set([])
for index in range(band):
    st = index * row
    bandBucket = {}

    for ele in business_map:
        val = 0
        for i in range(row):
            val += singnature_matrix[st + i][ele] * hash_array[i]
        hashval = val % 1699
        if hashval not in bandBucket:
            bandBucket[hashval] = [ele]
        else:
            bandBucket[hashval].append(ele)

    for ele in bandBucket.values():
        if len(ele) > 1:
            for i in (itertools.combinations(ele, 2)):
                candidate.add(tuple(sorted(i)))


business_user = new_data.map(lambda x: (x[1], x[0])).groupByKey().mapValues(set).collectAsMap()

similar_pairs = []
for pair in candidate:
    s1 = business_user[pair[0]]
    s2 = business_user[pair[1]]
    sim = float(len(s1.intersection(s2)))/ float(len(s1.union(s2)))
    if sim >= 0.5:
        similar_pairs.append((pair[0], pair[1], str(sim)))

out_pairs = sorted(similar_pairs)

output = open(sys.argv[2], "w")
print("business_id_1,business_id_2,similarity", file=output)

for ele in out_pairs:
    print(ele[0] + "," + ele[1] + "," + ele[2], file=output)

output.close()
Duration = time.time() - start_time
print("Duration: " + str(Duration))