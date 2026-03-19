import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{LinearSVC, OneVsRest}
import org.apache.spark.ml.feature._
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions.Window

// 1. Setup & Load
System.setProperty("hadoop.home.dir", "C:\\hadoop")
val rawData = spark.read
  .option("header", "true")
  .option("multiLine", "true")
  .option("escape", "\"")
  .csv("data/Merged_dataset.csv")

// 2. Clean & Remove Duplicates first
val cleanedData = rawData
  .withColumn("lyrics_clean", lower(regexp_replace(col("lyrics"), "single space", " ")))
  .withColumn("lyrics_clean", regexp_replace(col("lyrics_clean"), "[^a-z\\s]", ""))
  .filter(length(col("lyrics_clean")) > 150)
  .dropDuplicates("lyrics")

// 3. 🔥 THE "UNDERSAMPLING" STEP: Get exactly 500 from each
// We use a Window function to pick the first 500 random rows for every genre.
val windowSpec = Window.partitionBy("genre").orderBy(rand(42L))

val balancedData = cleanedData
  .withColumn("row_num", row_number().over(windowSpec))
  .filter(col("row_num") <= 500)
  .drop("row_num")

println(s"Balanced Dataset Count: ${balancedData.count()} rows (500 per genre)")

// 4. ML PIPELINE (High Feature Count for SVM)
val labelIndexer = new StringIndexer()
  .setInputCol("genre")
  .setOutputCol("label")
  .setHandleInvalid("skip")

val tokenizer = new RegexTokenizer()
  .setInputCol("lyrics_clean")
  .setOutputCol("words")
  .setPattern("\\s+")

val remover = new StopWordsRemover()
  .setInputCol("words")
  .setOutputCol("filtered")

// Using 20,000 features to capture unique genre vocabulary
val cv = new CountVectorizer()
  .setInputCol("filtered")
  .setOutputCol("rawFeatures")
  .setVocabSize(20000)

val idf = new IDF()
  .setInputCol("rawFeatures")
  .setOutputCol("features")

// 5. THE MODEL: Linear SVM (One-Vs-Rest)
// SVM is the gold standard for small, balanced text datasets.
val lsvc = new LinearSVC()
  .setMaxIter(25)
  .setRegParam(0.1)

val ovr = new OneVsRest()
  .setClassifier(lsvc)
  .setLabelCol("label")
  .setFeaturesCol("features")

val pipeline = new Pipeline().setStages(Array(
    labelIndexer, tokenizer, remover, cv, idf, ovr
))

// 6. Train/Test Split (80/20)
// Since we have 4,000 rows, this gives us 3,200 for training and 800 for testing.
val Array(train, test) = balancedData.randomSplit(Array(0.8, 0.2), seed = 42L)

println("Training Balanced SVM (Undersampled)...")
val model = pipeline.fit(train)

// 7. Evaluation
val predictions = model.transform(test)
val evaluator = new MulticlassClassificationEvaluator()
  .setLabelCol("label")
  .setMetricName("accuracy")

val accuracy = evaluator.evaluate(predictions)

println(s"==========================================")
println(s"UNDERSAMPLED (REAL) ACCURACY: ${accuracy * 100}%")
println(s"==========================================")

// 8. Save the trained Scala PipelineModel for the web app
//    (separate path so it doesn't conflict with any PySpark models)
val scalaModelPath = "music_genre_model_scala"
println(s"Saving Scala PipelineModel to '$scalaModelPath' ...")
model.write.overwrite().save(scalaModelPath)
println(s"Saved Scala model to '$scalaModelPath'.")