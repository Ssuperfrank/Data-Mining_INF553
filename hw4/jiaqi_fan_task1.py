from pyspark import SparkContext, SparkConf
import sys, time, itertools


def order_node(a, b):
    return (min(a, b), max(a, b))


def Girvan_Newman(root, adjacent_nodes):
    leveldic = {}
    parents = {}
    childs = {}
    visited = set([])

    head = root
    childs[head] = adjacent_nodes[head]
    parents[head] = set([])
    leveldic[0] = {head}
    visited.add(head)

    leveldic[1] = set([])
    for i in childs[head]:
        leveldic[1].add(i)
        visited.add(i)

    level_num = 1
    while len(leveldic[level_num]) != 0:
        leveldic[level_num+1] = set([])
        for i in leveldic[level_num]:
            parents[i] = leveldic[level_num-1].intersection(adjacent_nodes[i])
            # childs[i] = set([])
            # for node in adjacent_nodes[i]:
            #     if node not in visited:
            #         childs[i].add(node)
            childs[i] = adjacent_nodes[i] - visited
            # childs[i] = adjacent_nodes[i] - leveldic[level_num] - leveldic[level_num-1]
            leveldic[level_num+1] = leveldic[level_num+1].union(childs[i])
        visited = visited.union(leveldic[level_num+1])
        level_num += 1

    weight_nodes = {}
    weight_edges = {}
    for num in range(level_num, -1, -1):
        for node in leveldic[num]:
            if len(childs[node]) != 0 or len(parents[node]) != 0:
                if len(childs[node]) == 0:
                    weight_nodes[node] = 1
                    up_weight = float(weight_nodes[node])/ len(parents[node])
                    for i in parents[node]:
                        weight_edges[order_node(node, i)] = up_weight
                else:
                    weight = 0
                    for i in childs[node]:
                        weight += weight_edges[order_node(i, node)]
                    weight_nodes[node] = weight + 1
                    if len(parents[node]) == 0:
                        continue
                    up_weight = float(weight_nodes[node]) / len(parents[node])
                    for i in parents[node]:
                        weight_edges[order_node(node, i)] = up_weight

    for pairs in weight_edges:
        yield (pairs, weight_edges[pairs])


start_time = time.time()
sc = SparkContext("local[*]", "inf553_hw4")

threshold = int(sys.argv[1])
data = sc.textFile(sys.argv[2])

header = data.first()
data = data.filter(lambda record: record != header).map(lambda record: record.split(",")).cache()
user_all_business = data.map(lambda x: (x[0], x[1])).groupByKey().mapValues(set).collectAsMap()

graph_con = []
for con in itertools.combinations(user_all_business.keys(), 2):
    if len(user_all_business[con[0]] & user_all_business[con[1]]) >= threshold:
        graph_con.append(con)
        graph_con.append((con[1], con[0]))

edges = sc.parallelize(graph_con).map(lambda x: (x[0], x[1])).groupByKey().mapValues(set)
user_onodes = edges.collectAsMap()

betweenness = edges.flatMap(lambda x: Girvan_Newman(x[0], user_onodes)).reduceByKey(lambda x, y: x+y).map(lambda x: (x[0], float(x[1]/2))).sortByKey().sortBy(lambda x: x[1], ascending = False)
ori_bet = betweenness.collect()

between_output = open(sys.argv[3], "w")
for i in ori_bet:
    print(i[0], ",", i[1], file=between_output)
between_output.close()

user = edges.keys()
edges_num = len(ori_bet)   #m
adjacent_matrix = {}        #A
for i in ori_bet:
    adjacent_matrix[i[0]] = 1
nodes_degree = {}
for i in user_onodes:
    nodes_degree[i] = len(user_onodes[i])


def combine_set(node):
    parents = {}
    childs = {}
    leveldic = {}

    leveldic[0] = {node}
    leveldic[1] = user_onodes[node]

    level_num = 1
    while len(leveldic[level_num]) != 0:
        leveldic[level_num+1] = set([])
        for i in leveldic[level_num]:
            parents[i] = leveldic[level_num-1].intersection(user_onodes[i])
            childs[i] = user_onodes[i] - leveldic[level_num] - leveldic[level_num-1]
            leveldic[level_num+1] = leveldic[level_num+1].union(childs[i])
        level_num += 1
    community = {node}
    for i in leveldic:
        community = community.union(leveldic[i])
    yield (tuple(sorted(community)), 1)


def cal_modularity(community):
    modularity = 0
    for i in community:
        for j in community:
            A = 0
            if order_node(i, j) in adjacent_matrix:
                A = 1
            modularity += A-(nodes_degree[i]*nodes_degree[j])/(2*edges_num)
    # modularity= modularity/(2*edges_num)
    return (community, modularity)


communities = user.flatMap(lambda x: combine_set(x)).reduceByKey(lambda x, y: x+y).map(lambda x: x[0])
modularity = communities.map(lambda x: cal_modularity(x)).values().sum() / (2*edges_num)
max_modularity = modularity
max_communities = communities


links = edges_num
while links != 0:
    high_bet = betweenness.first()[1]
    remove_edges = betweenness.filter(lambda x: x[1] == high_bet).map(lambda x: x[0]).collect()
    for i in remove_edges:
        links -= 1
        user_onodes[i[0]] = user_onodes[i[0]] - {i[1]}
        user_onodes[i[1]] = user_onodes[i[1]] - {i[0]}

    betweenness = user.flatMap(lambda x: Girvan_Newman(x, user_onodes)).reduceByKey(lambda x, y: x+y).map(lambda x: (x[0], x[1]/2)).sortBy(lambda x: x[1], ascending = False)
    communities = user.flatMap(lambda x: combine_set(x)).reduceByKey(lambda x, y: x+y).map(lambda x: x[0])
    modularity = communities.map(lambda x: cal_modularity(x)).values().sum() / (2*edges_num)
    if modularity > max_modularity:
        max_modularity = modularity
        max_communities = communities

communities_output = max_communities.map(lambda x: (x, len(x))).sortByKey().sortBy(lambda x: x[1]).map(lambda x: x[0]).collect()
community_output = open(sys.argv[4], "w")
for community in communities_output:
    st = str(community).rstrip(")").lstrip("(")
    print(st, file=community_output)
community_output.close()

duration = time.time() - start_time
print("Duration: ", duration)

