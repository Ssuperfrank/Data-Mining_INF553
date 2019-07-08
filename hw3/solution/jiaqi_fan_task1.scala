import java.io._
import java.util.Random
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf


object jiaqi_fan_task1 {



  def main(args: Array[String]): Unit = {
    val start = System.currentTimeMillis()
    val conf = new SparkConf()
      .setAppName("INF553_hw3")
      .setMaster("local[*]")
    val sc =  new SparkContext(conf)
    val train_data = sc.textFile(args(0))
//    val train_data = sc.textFile("/Users/frank/Desktop/553/hw_data/hw3/yelp_train.csv")

    val header = train_data.first()
    val data = train_data.filter(record => record!= header).map(record=> record.split(",")).cache()

    val user = data.map(x => (x(0), 1)).reduceByKey((x, y)=>  x+y).keys
    val business = data.map(x => (x(1), 1)).reduceByKey((x, y)=>  x+y).keys

    val user_num = user.count()
    val business_num = business.count()

    var user_map = scala.collection.mutable.Map[String,Int]()
    var business_map = scala.collection.mutable.Map[String,Int]()
    var num = 0
    for(i <- user.collect()){
      user_map += (i -> num)
      num += 1
    }
    num = 0
    for(i <- business.collect()){
      business_map += (i -> num)
      num += 1
    }

    val band = 15
    val row = 2
    var n = band * row

    var singnature_matrix = scala.collection.mutable.ListBuffer[scala.collection.Map[String, Int]]()
    val ran = List[Int](71, 277, 691, 1699, 2087, 4211, 6329, 9043, 12613)
    val r = new Random()
    while(n != 0){
      n -= 1
      val a = r.nextInt(100)
      val b = r.nextInt(100)
      val p = ran(r.nextInt(ran.length))
      val single_line = data.map( x=> ( x(1),  (a * user_map(x(0)) + b) % p )).groupByKey().mapValues(x=> x.toList).map(x => (x._1, x._2.min)).collectAsMap()
      singnature_matrix += single_line
    }

    val hash_array = List[Int](1, 23, 133, 231, 591, 891, 1331, 2937, 3473, 5751, 8717, 13321)
    var candidates = scala.collection.mutable.Set[(String,String)]()

    for(index <- 0 until(band)){
      val start = index * row
      var bandBucket = scala.collection.mutable.Map[Int,scala.collection.mutable.ListBuffer[String]]()

      for (ele <- business_map){
        var value = 0
        for(i <- 0 until(row)){
          value += singnature_matrix(start + i)(ele._1) * hash_array(i)
        }
        val hash_value = value % 16999
        if(!bandBucket.contains(hash_value)){
          bandBucket(hash_value) =  scala.collection.mutable.ListBuffer[String](ele._1)
        }else{
          bandBucket(hash_value) += ele._1
        }
      }

      for(ele <- bandBucket.values){
        if (ele.toList.length > 1){
          for(i <- ele){
            for(j <- ele){
              if(i != j) {
                val x = List(i,j).sorted
                candidates += Tuple2(x(0), x(1))
              }
            }
          }
        }
      }
    }

    val business_user = data.map(x => (x(1), x(0))).groupByKey().mapValues(x => x.toSet).collectAsMap()

    var similar_pairs = scala.collection.mutable.ListBuffer[(String,String,String)]()
    for (pair <- candidates){
      var s1 = business_user(pair._1)
      var s2 = business_user(pair._2)
      var sim = s1.intersect(s2).size.toFloat / s1.union(s2).size.toFloat
      if (sim >= 0.5)
        similar_pairs += Tuple3(pair._1, pair._2, sim.toString)
    }

    val output_pairs = similar_pairs.toList.sorted
    val output = new PrintWriter(args(1))
//    val output = new PrintWriter("output.csv")

    output.write("business_id_1,business_id_2,similarity\n")
    for (ele <- output_pairs){
      output.write( ele._1 + "," + ele._2 + "," +ele._3 + "\n")
    }
    output.close()
    val Duration = (System.currentTimeMillis() - start)/1000
    println("Duration: " + Duration.toString)
    println(output_pairs.length)


  }
}