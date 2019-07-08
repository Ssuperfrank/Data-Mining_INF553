from pyspark import SparkContext
from pyspark.conf import SparkConf
import sys
import time

start_time=time.time()
threshold = float(sys.argv[1])
support = float(sys.argv[2])
input_file = sys.argv[3]
output_file = sys.argv[4]
conf = SparkConf()
conf.setMaster("local[*]").setAppName("My app")
sc = SparkContext(conf = conf)
lines = sc.textFile(input_file)
header = lines.first()
process_lines = lines.filter(lambda s: s != header).map(lambda s: s.split(","))


baskets = process_lines.map(lambda x: (x[0], x[1])).distinct().groupByKey().mapValues(list).filter(lambda x: len(x[1])>threshold).map(lambda x:(set(x[1])))
print(baskets.collect())



Total_length = len(baskets.collect())
partitions = baskets.getNumPartitions()
print(partitions)

def aPriori(iterator, s):

	baskets= list(iterator)
	length = len(baskets)
	pass_support = s / partitions
	pass_support = s * (length / Total_length)
	single = {}
	frequItems = []
	checkItem = set()
	for basket in baskets:
		for item in basket:
			if item not in single:
				single[item] = 1
			else:
				single[item] += 1
	for index in single:
		if single[index] >= pass_support:
			frequItems.append({index})
	temp = frequItems
	size = 0
	while(temp!=[]):
		size += 1
		candidates = []
		checkCand = set()
		for x in temp:
			for y in temp:
				items = x | y
				if len(items) == size:
					sorted_items = tuple(sorted(tuple(items)))
					if sorted_items not in checkCand:
						candidates.append(items)
						checkCand.add(sorted_items)
		temp=[]
		for cand in candidates:
			if tuple(sorted(list(cand))) not in checkItem:
				count=0
				for basket in baskets:
					if cand.issubset(basket):
						count += 1
					if count >= pass_support:
						checkItem.add(tuple(sorted(list(cand))))
						frequItems.append(cand)
						temp.append(cand)
						break
	yield frequItems

aprior =baskets.mapPartitions(lambda x :aPriori(x,support)).flatMap(lambda x:x)
sorted_aprior = aprior.map(lambda x:(tuple(sorted(list(x))),1)).reduceByKey(lambda x,y:x+y,numPartitions=1)
candidate = sorted(sorted(sorted_aprior.map(lambda x:x[0]).collect()),key=lambda x:len(x),reverse=False)
print(candidate)
def Count(iterator,candidate):
	baskets = list(iterator)
	cout_dic = {}
	count = []

	for cand in candidate:
		for basket in baskets:
			if (set(cand)).issubset(basket):
				if cand not in cout_dic:
					cout_dic[cand] = 1
				else:
					cout_dic[cand] += 1
	for index in cout_dic:
		count.append((index , cout_dic[index]))
	yield count

count = baskets.mapPartitions(lambda x:Count(x,candidate)).flatMap(lambda x:x).reduceByKey(lambda x,y:x+y,numPartitions=1)
frequentItem = sorted(sorted(count.filter(lambda x:x[1]>=support).map(lambda x:x[0]).collect()),key=lambda x:len(x),reverse=False)
print(frequentItem)

output=open(output_file,'w')
output.write("Candidates:\n")
size = 0
for cand in candidate:
	if len(cand) == size:
		output.write(",")
	elif size == 0:
		t = 0
	else:
		output.write("\n\n")
	if len(cand) == 1:
		output.write("('" + str(cand[0]) + "')")
	else:
		output.write(str(cand))
	size = len(cand)
output.write("\n\nFrequent Itemsets:\n")
size = 0
for cand in frequentItem:
	if len(cand) == size:
		output.write(",")
	elif size == 0:
		t = 0
	else:
		output.write("\n\n")
	if len(cand) == 1:
		output.write("('" + str(cand[0]) + "')")
	else:
		output.write(str(cand))
	size = len(cand)
end_time=time.time()
print(end_time - start_time)
output.close()