import java.io.PrintWriter

import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.json4s._
import org.json4s.jackson.JsonMethods._


object jiaqi_fan_task2 {

  case class review(review_id:String,user_id:String,business_id:String,stars:Int,useful:Int,funny:Int,cool:Int,text:String,date:String)
  case class business(business_id:String,name:String,address:String,city:String,state:String,postal_code:String,latitude:Float,longitude:Float,stars:Float,review_count:Int,is_open:Int,attributes:Any,categories:Any,hours:Any)


  def main(args: Array[String]): Unit = {

    val conf = new SparkConf()
      .setAppName("INF553_hw1")
      .setMaster("local[*]")
    val sc =  new SparkContext(conf)
    val review = sc.textFile(args(0)).cache()
    val business = sc.textFile(args(1)).cache()

    val bus = business.map(record => {implicit val formats = DefaultFormats ; (parse(record).extract[business].business_id, parse(record).extract[business].state)})
    val rev = review.map(record => {implicit val formats = DefaultFormats ; (parse(record).extract[review].business_id, (parse(record).extract[review]stars, 1))}).reduceByKey((x,y) => (x._1+y._1, x._2+y._2))

    val state = bus.join(rev).map(record => record._2).reduceByKey((x,y) => (x._1+y._1, y._2+x._2)).map(record => (record._1, record._2._1.toFloat/record._2._2.toFloat)).sortByKey().sortBy(x => x._2, ascending = false)


    val start1 = System.currentTimeMillis()
    val m1 = state.collect()
    for (i <- 0 until 5)
      println(m1(i))
    val during1 = System.currentTimeMillis() - start1

    val start2 = System.currentTimeMillis()
    val m2 = state.take(5)
    for (i <- 0 until m2.length)
      println(m2(i))
    val during2 = System.currentTimeMillis() - start2

    val q1 = new PrintWriter(args(2))
    val q2 = new PrintWriter(args(3))


    q1.write("state,stars\n")
    for (i <- 0 until m1.length)
      q1.write(m1(i)._1+","+m1(i)._2+"\n")
    q1.close()

    q2.write("{\n")
    q2.write("\t\"m1\": "+during1+",\n")
    q2.write("\t\"m2\": "+during2+",\n")
    q2.write("\t\"explanation\": \"The collect() function needs to convert the entire dataset into an array. However, take() " +
      "function gets and converts first n data of the dataset into an array. which has smaller computation.\"\n")
    q2.write("}")
    q2.close()

  }


}