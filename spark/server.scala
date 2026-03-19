import java.io.{File, InputStream}
import java.net.InetSocketAddress
import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path, Paths}

import com.sun.net.httpserver.{HttpExchange, HttpHandler, HttpServer}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.feature.StringIndexerModel
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.sql.functions._

// Spark shell provides an existing SparkSession named `spark`.
// This script starts a small local HTTP server that:
// - serves the frontend from ./web
// - exposes POST /api/predict (text/plain body: lyrics) returning JSON with scores for each label

val Host = "127.0.0.1"
val Port = 8080
// Use the Scala-trained model saved by spark/train_model.scala
val ModelPath = "music_genre_model_scala"
val WebRoot = Paths.get("web").toAbsolutePath.normalize()

// Windows Spark often requires winutils via hadoop.home.dir (matches your training script).
System.setProperty("hadoop.home.dir", "C:\\hadoop")
System.setProperty("HADOOP_HOME", "C:\\hadoop")

// If present, load the Hadoop native DLL to avoid NativeIO$Windows.access0 errors.
try {
  val hadoopDll = Paths.get("C:\\hadoop\\bin\\hadoop.dll")
  if (Files.exists(hadoopDll)) {
    System.load(hadoopDll.toString)
    println(s"Loaded native Hadoop DLL: $hadoopDll")
  } else {
    println(s"Note: $hadoopDll not found (if you hit NativeIO errors, install matching Hadoop winutils + hadoop.dll).")
  }
} catch {
  case e: Throwable =>
    println(s"Warning: failed to load native Hadoop DLL: ${e.getMessage}")
}

def readAll(is: InputStream): Array[Byte] = {
  val buf = new Array[Byte](8192)
  val baos = new java.io.ByteArrayOutputStream()
  var n = is.read(buf)
  while (n != -1) {
    baos.write(buf, 0, n)
    n = is.read(buf)
  }
  baos.toByteArray
}

def send(exchange: HttpExchange, status: Int, contentType: String, bytes: Array[Byte]): Unit = {
  exchange.getResponseHeaders.set("Content-Type", contentType)
  exchange.getResponseHeaders.set("Cache-Control", "no-store")
  exchange.getResponseHeaders.set("Access-Control-Allow-Origin", "*")
  exchange.getResponseHeaders.set("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
  exchange.getResponseHeaders.set("Access-Control-Allow-Headers", "Content-Type")
  exchange.sendResponseHeaders(status, bytes.length.toLong)
  val os = exchange.getResponseBody
  os.write(bytes)
  os.close()
}

def sendText(exchange: HttpExchange, status: Int, text: String, contentType: String = "text/plain; charset=utf-8"): Unit = {
  send(exchange, status, contentType, text.getBytes(StandardCharsets.UTF_8))
}

def guessContentType(p: Path): String = {
  val name = p.getFileName.toString.toLowerCase
  if (name.endsWith(".html")) "text/html; charset=utf-8"
  else if (name.endsWith(".css")) "text/css; charset=utf-8"
  else if (name.endsWith(".js")) "application/javascript; charset=utf-8"
  else if (name.endsWith(".json")) "application/json; charset=utf-8"
  else if (name.endsWith(".png")) "image/png"
  else if (name.endsWith(".jpg") || name.endsWith(".jpeg")) "image/jpeg"
  else if (name.endsWith(".svg")) "image/svg+xml; charset=utf-8"
  else "application/octet-stream"
}

def safeResolve(webRoot: Path, requestedPath: String): Option[Path] = {
  val raw = requestedPath.takeWhile(_ != '?')
  val cleaned =
    if (raw == null || raw.isEmpty || raw == "/") "index.html"
    else raw.stripPrefix("/").replace('\\', '/')

  val resolved = webRoot.resolve(cleaned).normalize()
  if (!resolved.startsWith(webRoot)) None
  else Some(resolved)
}

def jsonEscape(s: String): String =
  s.flatMap {
    case '"'  => "\\\""
    case '\\' => "\\\\"
    case '\b' => "\\b"
    case '\f' => "\\f"
    case '\n' => "\\n"
    case '\r' => "\\r"
    case '\t' => "\\t"
    case c if c.isControl => f"\\u${c.toInt}%04x"
    case c => c.toString
  }

def softmax(xs: Array[Double]): Array[Double] = {
  if (xs.isEmpty) xs
  else {
    val max = xs.max
    val exps = xs.map(x => math.exp(x - max))
    val sum = exps.sum
    if (sum == 0.0) xs.map(_ => 0.0) else exps.map(_ / sum)
  }
}

println(s"Loading PipelineModel from '$ModelPath' ...")
val pipelineModel = PipelineModel.load(ModelPath)
val labelStage = pipelineModel.stages.headOption match {
  case Some(sim: StringIndexerModel) => sim
  case Some(other) =>
    throw new IllegalStateException(s"Expected first stage to be StringIndexerModel, got: ${other.getClass.getName}")
  case None =>
    throw new IllegalStateException("PipelineModel has no stages.")
}

val labels: Array[String] = labelStage.labels
println(s"Loaded model with ${labels.length} labels: ${labels.mkString(", ")}")
println(s"Serving web from: $WebRoot")

def predictScores(lyrics: String): (String, Array[(String, Double)]) = {
  import spark.implicits._
  val df = Seq(lyrics).toDF("lyrics")
  val cleaned = df
    .withColumn("lyrics_clean", lower(regexp_replace(col("lyrics"), "single space", " ")))
    .withColumn("lyrics_clean", regexp_replace(col("lyrics_clean"), "[^a-z\\s]", ""))

  val out = pipelineModel.transform(cleaned)
  // LinearSVC -> OneVsRest gives rawPrediction margins but no probability.
  val rawVec = out.select(col("rawPrediction")).head().getAs[org.apache.spark.ml.linalg.Vector](0)
  val scores = softmax(rawVec.toArray)
  val pairs = labels.zip(scores).sortBy { case (_, p) => -p }
  val predicted = pairs.headOption.map(_._1).getOrElse("unknown")
  (predicted, pairs)
}

val server = HttpServer.create(new InetSocketAddress(Host, Port), 0)

server.createContext("/api/health", new HttpHandler {
  override def handle(exchange: HttpExchange): Unit = {
    if (exchange.getRequestMethod.equalsIgnoreCase("OPTIONS")) {
      send(exchange, 204, "text/plain; charset=utf-8", Array.emptyByteArray)
      return
    }
    sendText(exchange, 200, "ok")
  }
})

server.createContext("/api/predict", new HttpHandler {
  override def handle(exchange: HttpExchange): Unit = {
    if (exchange.getRequestMethod.equalsIgnoreCase("OPTIONS")) {
      send(exchange, 204, "text/plain; charset=utf-8", Array.emptyByteArray)
      return
    }
    if (!exchange.getRequestMethod.equalsIgnoreCase("POST")) {
      sendText(exchange, 405, "Method Not Allowed")
      return
    }
    try {
      val body = new String(readAll(exchange.getRequestBody), StandardCharsets.UTF_8).trim
      if (body.isEmpty) {
        sendText(exchange, 400, """{"error":"Empty request body"}""", "application/json; charset=utf-8")
        return
      }
      val (predicted, pairs) = predictScores(body)
      val itemsJson = pairs
        .map { case (label, p) => s"""{"label":"${jsonEscape(label)}","score":${"%.10f".formatLocal(java.util.Locale.US, p)}}""" }
        .mkString("[", ",", "]")
      val json =
        s"""{"predicted":"${jsonEscape(predicted)}","labels":${labels.map(l => s""""${jsonEscape(l)}"""").mkString("[", ",", "]")},"scores":$itemsJson}"""
      sendText(exchange, 200, json, "application/json; charset=utf-8")
    } catch {
      case e: Throwable =>
        val msg = jsonEscape(Option(e.getMessage).getOrElse(e.toString))
        sendText(exchange, 500, s"""{"error":"$msg"}""", "application/json; charset=utf-8")
    }
  }
})

server.createContext("/", new HttpHandler {
  override def handle(exchange: HttpExchange): Unit = {
    if (exchange.getRequestMethod.equalsIgnoreCase("OPTIONS")) {
      send(exchange, 204, "text/plain; charset=utf-8", Array.emptyByteArray)
      return
    }
    if (!exchange.getRequestMethod.equalsIgnoreCase("GET")) {
      sendText(exchange, 405, "Method Not Allowed")
      return
    }
    val reqPath = exchange.getRequestURI.getPath
    safeResolve(WebRoot, reqPath) match {
      case None =>
        sendText(exchange, 400, "Bad Request")
      case Some(p) if Files.exists(p) && Files.isRegularFile(p) =>
        val bytes = Files.readAllBytes(p)
        send(exchange, 200, guessContentType(p), bytes)
      case Some(_) =>
        sendText(exchange, 404, "Not Found")
    }
  }
})

server.setExecutor(java.util.concurrent.Executors.newFixedThreadPool(8))
server.start()

println(s"✅ Server running at http://$Host:$Port (press Ctrl+C to stop spark-shell)")
