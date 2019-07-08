import java.io._
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf

object jiaqi_fan_task2 {

  def combinerPairs(input: List[String]): Set[List[String]] = {
    var newPairs = Set[List[String]]()
    val len = input.length
    for(i <- 0 until len ){
      for (j <- i+1 until len){
        newPairs += List(input(i),input(j)).sorted
      }
    }
    return newPairs
  }

  def combineTriple(input: List[List[String]], size: Int): Set[List[String]] ={
    var newTriple = Set[List[String]]()
    val len = input.length
    for (i <- 0 until(len)){
      for(j <- i+1 until(len)) {
        var aset = Set[String]()

        for (ele <- input(i))
          aset += ele
        for (ele <- input(j))
          aset += ele

        if(aset.size == size)
          newTriple += aset.toList.sorted
      }
    }

    return newTriple
  }

  def aPriori(iterator: Iterator[Set[String]], ps: Float): Iterator[(List[String], Int)] ={
    var candidates = scala.collection.mutable.ListBuffer.empty[(List[String], Int)]
    var baskets = scala.collection.mutable.ListBuffer.empty[Set[String]]
    var num_basket = 0
    var single_set = Set[String]()

    while(iterator.hasNext) {
      var basket = iterator.next()
      baskets += basket
      num_basket += 1
      for (ele <- basket) {
        single_set += ele
      }
    }
    val threshold = ps * num_basket

    var fre_single = Set[String]()
    for (i <- single_set){
      var count = 0
      for(j <- baskets){
        if(Set(i).intersect(j) == Set(i)){
          count += 1
        }
      }
      if (count >= threshold){
        fre_single += i
        candidates += Tuple2(List(i), 1)
      }
    }

    var size = 2
    var pairSet = combinerPairs(fre_single.toList)
    var fre_pair = Set[List[String]]()
    for (i <- pairSet){
      var count = 0
      for (j <- baskets){
        if ( j.intersect(i.toSet) == i.toSet){
          count += 1
        }
      }
      if(count >= threshold){
        fre_pair += i
        candidates += Tuple2(i, 1)
      }
    }

    while(!fre_pair.isEmpty){
      size += 1
      var triple_set = combineTriple(fre_pair.toList, size)
      fre_pair = Set[List[String]]()
      for (i <- triple_set){
        var count = 0
        for (j <- baskets){
          if ( j.intersect(i.toSet) ==  i.toSet){
            count += 1
          }
        }
        if(count >= threshold){
          fre_pair += i
          candidates += Tuple2(i, 1)
        }
      }
    }
    return candidates.iterator
  }

  def countFre(iterator: Iterator[List[String]] , baskets: Array[Set[String]]):  Iterator[(List[String], Int)] = {
    var fre = scala.collection.mutable.ListBuffer.empty[(List[String], Int)]
    while(iterator.hasNext){
      val pair = iterator.next()
      var count = 0
      for(i <- baskets){
        if(i.intersect(pair.toSet) == pair.toSet){
          count += 1
        }
      }
      fre += Tuple2(pair, count)
    }
    return fre.iterator
  }

  def main(args: Array[String]): Unit = {
    val start_time = System.currentTimeMillis()

    val conf = new SparkConf()
      .setAppName("INF553_hw2")
      .setMaster("local[*]")
    val sc =  new SparkContext(conf)
    //    val case_number = args(0)
    //    val support = args(1)
    //    val data = sc.textFile(args(2))
    //    val output = new PrintWriter(args(3))
    val filter_threshold = 70
    val support = 50
    val data = sc.textFile("/Users/frank/Desktop/553/data/AZ_yelp.csv").cache()
    val output = new PrintWriter("output2.txt")

    val header = data.first()
    val preData = data.filter(record => record != header).map(record => record.split(","))
    val newData = preData.map(x=>(x(0),x(1))).groupByKey().mapValues(x=> x.toSet).filter(x => x._2.size > filter_threshold).values.cache()

    val baskets = newData.collect()
    val num_baskets = newData.count()
    var ps = support.toFloat / num_baskets

    val pass1 = newData.mapPartitions(chunk => aPriori(chunk, ps)).reduceByKey((x,y) => x +y).keys
    val pass2 = pass1.mapPartitions(fre => countFre(fre,baskets)).filter(x=> x._2 >= support).keys

    val intermediate_result = pass1.map(x => (x.length, x)).groupByKey().mapValues(x=> x.toList).sortByKey().collect()
    val final_result = pass2.map(x => (x.length, x)).groupByKey().mapValues(x=> x.toList).sortByKey().collect()

    var intermediate_result_output = scala.collection.mutable.ListBuffer.empty[List[String]]
    var final_result_output = scala.collection.mutable.ListBuffer.empty[List[String]]

    for(i <- intermediate_result){
      val value = i._2
      var set = Set[String]()
      for (list_pair <- value){
        var string = ""
        string += "("
        val last_one = list_pair.last
        for (ele <- list_pair){
          if (ele == last_one){
            string += "\'"+ele+"\'"
          }else{
            string += "\'"+ele+"\', "
          }
        }
        string += ")"
        set += string
      }
      intermediate_result_output += set.toList.sorted
    }

    for(i <- final_result){
      val value = i._2
      var set = Set[String]()
      for (list_pair <- value){
        var string = ""
        string += "("
        val last_one = list_pair.last
        for (ele <- list_pair){
          if (ele == last_one){
            string += "\'"+ele+"\'"
          }else{
            string += "\'"+ele+"\', "
          }
        }
        string += ")"
        set += string
      }
      final_result_output += set.toList.sorted
    }

    output.write("Intermediate Itemsets:\n")
    for(list <- final_result_output){
      val last_one = list.last
      for (ele <- list){
        if(last_one == ele){
          output.write(ele + "\n")
        }else{
          output.write(ele + ", ")
        }
      }
      output.write("\n")
    }

    output.write("Frequent Itemsets:\n")
    for(list <- final_result_output){
      val last_one = list.last
      for (ele <- list){
        if(last_one == ele){
          output.write(ele + "\n")
        }else{
          output.write(ele + ", ")
        }
      }
      output.write("\n")
    }
    output.close()

    val end_time = System.currentTimeMillis()
    println("Duration: " + (end_time-start_time)/1000)

    output.close()
  }
}
