import java.io.PrintWriter

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.recommendation.{ALS, Rating}

object jiaqi_fan_task2 {



  def main(args: Array[String]): Unit = {
    val start = System.currentTimeMillis()
    val conf = new SparkConf()
      .setAppName("INF553_hw3")
      .setMaster("local[*]")
    val sc =  new SparkContext(conf)
    val train = sc.textFile(args(0))
    val test = sc.textFile(args(1))
    val case_num = args(2).toInt
//
//    val train = sc.textFile("/Users/frank/Desktop/553/hw_data/hw3/yelp_train.csv")
//    val test = sc.textFile("/Users/frank/Desktop/553/hw_data/hw3/yelp_val.csv")
//    val case_num = 2


    val train_header = train.first()
    val train_data = train.filter(x=> x!= train_header).map(x => x.split(",")).map(x => (x(0), x(1), x(2))).cache()
    val test_header = test.first()
    val test_data = test.filter(x=> x!= test_header).map(x => x.split(",")).map(x => (x(0), x(1), x(2))).cache()


    if(case_num == 1) {

      val user = train_data.map(x => x._1).union(test_data.map(x => x._1)).distinct().collect()
      val business = train_data.map(x => x._2).union(test_data.map(x => x._2)).distinct().collect()

      var user_id_map = scala.collection.mutable.Map[String, Int]()
      var business_id_map = scala.collection.mutable.Map[String, Int]()
      var id_user_map = scala.collection.mutable.Map[Int, String]()
      var id_business_map = scala.collection.mutable.Map[Int, String]()
      var num = 0
      for (i <- user) {
        user_id_map += (i -> num)
        id_user_map += (num -> i)
        num += 1
      }
      num = 0
      for (i <- business) {
        business_id_map += (i -> num)
        id_business_map += (num -> i)
        num += 1
      }

      val test = test_data.map(x => (user_id_map(x._1), business_id_map(x._2)))
      val training = train_data.map(x => (x._1, x._2, x._3) match {case (user, business, rating) => Rating( user_id_map(user), business_id_map(business),rating.toDouble )})

      val rank = 5
      val numIterations = 10
      val model = ALS.train(training, rank, numIterations, 0.2)

      val predictions = model.predict(test).map{case Rating(user, business, ratings) => ((id_user_map(user), id_business_map(business)),ratings)}

      val output = new PrintWriter(args(3))
//      val output = new PrintWriter("output.csv")
      output.write("user_id, business_id, prediction\n")
      for(ele <- predictions.collect()){
        output.write( ele._1._1 + "," + ele._1._2 + "," +ele._2 + "\n")
      }

      val rmse = test_data.map(x => ((x._1, x._2), x._3.toDouble)).join(predictions).map(x => Math.abs(x._2._1 - x._2._2)).map(x => x*x).mean()
      println(rmse)

    }


    if (case_num == 2){
        val get_value = train_data.map(x => ((x._1, x._2), x._3.toDouble)).collectAsMap()
        val user_ave_value = train_data.map(x=> (x._1, (x._3.toDouble, 1))).reduceByKey((x,y) => (x._1 + y._1, x._2 + y._2)).map(x => (x._1,x._2._1/x._2._2)).collectAsMap()

        val user_all_business = train_data.map(x=> (x._1,x._2)).groupByKey().mapValues(x=> x.toSet).collectAsMap()
        val business_all_user = train_data.map(x=> (x._2,x._1)).groupByKey().mapValues(x=> x.toSet).collectAsMap()


        val test_pair = test_data.map(x => (x._1, x._2)).collect()

        var predictions = scala.collection.mutable.ListBuffer[((String,String),Double)]()
        for (pair <- test_pair){

          if(!user_all_business.contains(pair._1)) {
                predictions += Tuple2((pair._1, pair._2), 3.0)
          }else if (!business_all_user.contains(pair._2)){
                predictions += Tuple2((pair._1, pair._2), 3.0)
          }else{
              var user_weight = scala.collection.mutable.ListBuffer[(Double,Double)]()
              for (user <- business_all_user(pair._2)){

                val a_set = user_all_business(pair._1)
                val u_set = user_all_business(user)
                val co_set = a_set.intersect(u_set)

                val ave_u = user_ave_value(user)
                val ave_a = user_ave_value(pair._1)

                var up_num = 0.0
                var de1 = 0.0
                var de2 = 0.0

                for(item <- co_set){
                  up_num += (get_value((user,item)) - ave_u) * (get_value((pair._1,item)) - ave_a)
                  de1 += (get_value((user,item)) - ave_u) * (get_value((user,item)) - ave_u)
                  de2 += (get_value((pair._1,item)) - ave_a) * (get_value((pair._1,item)) - ave_a)
                }

                val lo_num = math.sqrt(de1)* math.sqrt(de2)
                var co_rated = 0.0
                if (lo_num!=0) {
                  co_rated = up_num / lo_num
                }

                user_weight += Tuple2(co_rated, get_value((user,pair._2)) - user_ave_value(user))
              }

            var up_sum = 0.0
            var lo_sum = 0.0

            for(x <- user_weight){
              up_sum += x._1 * x._2
              lo_sum += math.abs(x._1)
            }
            var pre = user_ave_value(pair._1)
            if(lo_sum != 0)
              pre  += up_sum/lo_sum

            predictions += Tuple2((pair._1, pair._2), pre)

          }

          val output = new PrintWriter(args(3))
          //      val output = new PrintWriter("output.csv")
          output.write("user_id, business_id, prediction\n")
          for(ele <- predictions.toList){
            output.write( ele._1._1 + "," + ele._1._2 + "," +ele._2 + "\n")
          }

          val p = sc.parallelize(predictions)
          val rmse = test_data.map(x => ((x._1, x._2), x._3.toDouble)).join(p).map(x => Math.abs(x._2._1 - x._2._2)).map(x => x*x).mean()
          println(rmse)

        }
    }


    val end = System.currentTimeMillis()
    val During = (end - start) /1000
    println("Duration: " + During)

    }
}
