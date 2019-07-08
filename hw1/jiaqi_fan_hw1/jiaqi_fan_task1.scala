import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.json4s._
import org.json4s.jackson.JsonMethods._
import java.io._

object jiaqi_fan_task1 {

  case class review(review_id:String,user_id:String,business_id:String,stars:Int,useful:Int,funny:Int,cool:Int,text:String,date:String)

  def main(args: Array[String]): Unit ={

    val conf = new SparkConf()
      .setAppName("INF553_hw1")
      .setMaster("local[*]")
    val sc =  new SparkContext(conf)
    val review = sc.textFile(args(0)).cache()
    val output = new PrintWriter(args(1))

    val A = review.filter(record => { implicit val formats = DefaultFormats ; parse(record).extract[review].useful > 0}).count()
    val B = review.filter(record => { implicit val formats = DefaultFormats ; parse(record).extract[review].stars == 5}).count()
    val C = review.map(record => { implicit val formats = DefaultFormats ; parse(record).extract[review].text.length}).max()
    val user = review.map(record => ({ implicit val formats = DefaultFormats ; parse(record).extract[review].user_id}, 1)).reduceByKey((x,y)=> x+y).sortByKey().sortBy(x => x._2, ascending = false)
    val D = user.count()
    val E = user.take(20)
    val business = review.map(record => ({ implicit val formats = DefaultFormats ; parse(record).extract[review].business_id}, 1)).reduceByKey((x,y)=> x+y).sortByKey().sortBy(x => x._2, ascending = false)
    val F = business.count()
    val G = business.take(20)


    output.write("{\n")
    output.write("\t\"n_review_useful\": " + A + ",\n")
    output.write("\t\"n_review_5_star\": " + B + ",\n")
    output.write("\t\"n_characters\": " + C + ",\n")
    output.write("\t\"n_user\": " + D + ",\n")

    output.write("\t\"top20_user\": [")

    for (i <- 0 until E.length){
      output.write("[\""+E(i)._1+"\", "+E(i)._2+"]" + ", ")
      if(i == E.length-1)
        output.write("[\""+E(i)._1+"\", "+E(i)._2+"]"+"],\n")
    }


    output.write("\t\"n_business\": " + F + ",\n")
    output.write("\t\"top20_business\": [")
    for (i <- 0 until G.length){
      output.write("[\""+G(i)._1+"\", "+G(i)._2+"]" + ", ")
      if(i == G.length-1)
        output.write("[\""+G(i)._1+"\", "+G(i)._2+"]" + "]\n")
    }
    output.write("}")

    output.close()
  }
}
