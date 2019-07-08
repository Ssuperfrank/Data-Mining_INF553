from pyspark import SparkContext
import sys, time, itertools


def removeDuplicate(item_set, size):
    newSet = set([])
    if size == 2:
        for item in item_set:
            aset = set([])
            for ele in item:
                aset.add(ele)
            newSet.add(tuple(sorted(aset)))
    else:
        for items in item_set:
            # items (()()())
            aset = set([])
            for item in items:
                for ele in item:
                    aset.add(ele)
            if len(aset) == size:
                newSet.add(tuple(sorted(aset)))
    return newSet


def aPriori(chunk, ps):
    num_basket_chunk = 0
    baskets = []
    candidates = set([])
    single_set = set([])

    for basket in chunk:
        num_basket_chunk += 1
        baskets.append(basket)
        for single in basket:
            single_set.add(single)

    threshold = ps * num_basket_chunk
    for i in single_set:
        count = 0
        for basket in baskets:
            if i in basket:
                count += 1
        if count >= threshold:
            candidates.add(i)
            yield ((i,), 1)

    size = 2
    preItemSets = candidates.copy()
    curItemSets = set([])
    while True:
        curItemSets = removeDuplicate(set(itertools.combinations(preItemSets, 2)), size).copy()
        if len(curItemSets) == 0:
            break
        preItemSets.clear()
        for pairs in curItemSets:
            count = 0
            for basket in baskets:
                if set(pairs).intersection(basket) == set(pairs):
                    count += 1
            if count >= threshold:
                preItemSets.add(pairs)
                yield (pairs, 1)
        curItemSets.clear()
        size += 1


def countFre(file, baskets):
    for itemCan in file:
        count = 0
        for basket in baskets:
            if set(itemCan).intersection(basket) == set(itemCan):
                count += 1
        yield (itemCan, count)


startTime = time.time()

sc = SparkContext("local[*]", "inf553_hw2")


threshold = int(sys.argv[1])
support = int(sys.argv[2])
data = sc.textFile(sys.argv[3]).cache()


header = data.first()
preData = data.filter(lambda record: record != header).map(lambda record: record.split(",")).groupByKey().mapValues(set).filter(lambda x: len(x[1]) > threshold).values().cache()

numBaskets = preData.count()
ps = support/numBaskets
baskets = preData.collect()

pass1 = preData.mapPartitions(lambda chunk: aPriori(chunk, ps)).reduceByKey(lambda x, y: x + y).keys()
pass2 = pass1.mapPartitions(lambda x: countFre(x, baskets)).filter(lambda x: x[1] >= support).keys()

intermediate_result = pass1.map(lambda x: (len(x), x)).sortBy(lambda x: x[1]).sortByKey().groupByKey().mapValues(list).sortByKey().collect()
final_result = pass2.map(lambda x: (len(x), x)).sortBy(lambda x: x[1]).sortByKey().groupByKey().mapValues(list).sortByKey().collect()

output = open(sys.argv[4],"w")

print("Intermediate Itemsets:", file=output)
strI = ""
for ele in intermediate_result[0][1]:
    strI = strI + ele.__str__().replace(",", "") + ", "
print(strI.rstrip(", "), file=output)
for i in range(1, len(intermediate_result)):
    print(intermediate_result[i][1].__str__().lstrip("[").rstrip("]"), file=output)

print("", file=output)

print("Frequent Itemsets:", file=output)
strF = ""
for ele in final_result[0][1]:
    strF = strF + ele.__str__().replace(",", "") + ", "
print(strF.rstrip(", "), file=output)
for i in range(1, len(final_result)):
    print(final_result[i][1].__str__().lstrip("[").rstrip("]"), file=output)

Duration = time.time() - startTime
print("Duration: " + Duration.__str__())
