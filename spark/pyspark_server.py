import json
import os
import posixpath
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import unquote, urlparse

from pyspark.ml import PipelineModel
from pyspark.ml.feature import StringIndexerModel
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace, lit


HOST = "127.0.0.1"
PORT = 8081  # avoid clash with Jenkins on 8080
MODEL_PATH = "music_genre_model"  # existing PySpark model directory
WEB_ROOT = (Path(__file__).resolve().parent.parent / "web").resolve()


def softmax(xs):
    if not xs:
        return xs
    m = max(xs)
    import math

    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [e / s for e in exps] if s != 0.0 else [0.0 for _ in xs]


def guess_type(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".html"):
        return "text/html; charset=utf-8"
    if name.endswith(".css"):
        return "text/css; charset=utf-8"
    if name.endswith(".js"):
        return "application/javascript; charset=utf-8"
    if name.endswith(".json"):
        return "application/json; charset=utf-8"
    if name.endswith(".png"):
        return "image/png"
    if name.endswith(".jpg") or name.endswith(".jpeg"):
        return "image/jpeg"
    if name.endswith(".svg"):
        return "image/svg+xml; charset=utf-8"
    return "application/octet-stream"


print("Starting SparkSession for PySpark server...")
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
spark = (
    SparkSession.builder.appName("LyricsGenreWebServerPySpark")
    .config("spark.python.worker.faulthandler.enabled", "true")
    .config("spark.sql.execution.pyspark.udf.faulthandler.enabled", "true")
    .config("spark.sql.execution.arrow.pyspark.enabled", "false")
    .config("spark.python.use.daemon", "false")
    .config("spark.python.worker.reuse", "false")
    .getOrCreate()
)

try:
    sc = spark.sparkContext
    print("Spark configs (python-related):")
    for k in [
        "spark.pyspark.driver.python",
        "spark.pyspark.python",
        "spark.python.worker.faulthandler.enabled",
        "spark.sql.execution.pyspark.udf.faulthandler.enabled",
        "spark.sql.execution.arrow.pyspark.enabled",
        "spark.python.use.daemon",
        "spark.python.worker.reuse",
    ]:
        try:
            print(f"  {k} = {sc.getConf().get(k)}")
        except Exception:
            pass
except Exception:
    pass

print(f"Loading PySpark PipelineModel from '{MODEL_PATH}' ...")
pipeline_model = PipelineModel.load(MODEL_PATH)

# First stage should be StringIndexerModel (like your Scala pipeline)
first_stage = pipeline_model.stages[0]
if not isinstance(first_stage, StringIndexerModel):
    raise RuntimeError(
        f"Expected first stage to be StringIndexerModel, got {type(first_stage)}"
    )

LABELS = list(first_stage.labels)
print(f"Loaded model with labels: {', '.join(LABELS)}")
print(f"Serving static files from: {WEB_ROOT}")


class Handler(BaseHTTPRequestHandler):
    server_version = "PySparkGenreServer/1.0"

    def _set_common_headers(self, status=200, content_type="text/plain; charset=utf-8"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_OPTIONS(self):
        self._set_common_headers(204)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            self._set_common_headers(200)
            self.wfile.write(b"ok")
            return

        # Static file serving
        rel = parsed.path
        if not rel or rel == "/":
            rel = "/index.html"

        rel = posixpath.normpath(unquote(rel)).lstrip("/")
        target = (WEB_ROOT / rel).resolve()

        if not str(target).startswith(str(WEB_ROOT)):
            self._set_common_headers(400)
            self.wfile.write(b"Bad Request")
            return

        if not target.exists() or not target.is_file():
            self._set_common_headers(404)
            self.wfile.write(b"Not Found")
            return

        try:
            data = target.read_bytes()
            self._set_common_headers(200, guess_type(target))
            self.wfile.write(data)
        except Exception as e:
            self._set_common_headers(500)
            self.wfile.write(f"Error reading file: {e}".encode("utf-8"))

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/predict":
            self._set_common_headers(404)
            self.wfile.write(b"Not Found")
            return

        length = int(self.headers.get("Content-Length", "0") or "0")
        body = self.rfile.read(length).decode("utf-8").strip()
        if not body:
            self._set_common_headers(
                400, "application/json; charset=utf-8"
            )
            self.wfile.write(b'{"error":"Empty request body"}')
            return

        try:
            from pyspark.ml.linalg import Vector

            # Build DataFrame and apply same cleaning as training
            df = spark.createDataFrame([(body,)], ["lyrics"])
            cleaned = (
                df.withColumn(
                    "lyrics_clean",
                    lower(
                        regexp_replace(
                            col("lyrics"),
                            "single space",
                            " ",
                        )
                    ),
                )
                .withColumn(
                    "lyrics_clean",
                    regexp_replace(col("lyrics_clean"), "[^a-z\\s]", ""),
                )
                # The fitted pipeline's first stage is a StringIndexer on "genre".
                # For inference we don't have a true genre label, but the transformer
                # still expects the column to exist. Provide a dummy value so that
                # downstream stages can run; the classifier itself only uses "features".
                .withColumn("genre", lit(LABELS[0]))
            )

            out = pipeline_model.transform(cleaned)
            raw_row = out.select("rawPrediction").first()
            vec = raw_row["rawPrediction"]
            if isinstance(vec, Vector):
                raw = list(vec.toArray())
            else:
                raw = list(vec)

            scores = softmax(raw)
            pairs = sorted(
                zip(LABELS, scores),
                key=lambda t: t[1],
                reverse=True,
            )
            predicted = pairs[0][0] if pairs else "unknown"

            payload = {
                "predicted": predicted,
                "labels": LABELS,
                "scores": [
                    {"label": label, "score": float(score)}
                    for label, score in pairs
                ],
            }

            self._set_common_headers(200, "application/json; charset=utf-8")
            self.wfile.write(json.dumps(payload).encode("utf-8"))
        except Exception as e:
            self._set_common_headers(500, "application/json; charset=utf-8")
            msg = str(e) or e.__class__.__name__
            self.wfile.write(
                json.dumps({"error": msg}).encode("utf-8")
            )


def main():
    server_address = (HOST, PORT)
    httpd = HTTPServer(server_address, Handler)
    # Avoid non-ASCII symbols here because Windows cmd.exe default codepage
    # (cp1252) can throw UnicodeEncodeError.
    print(
        f"PySpark HTTP server running at http://{HOST}:{PORT} "
        f"(Ctrl+C to stop)"
    )
    try:
        httpd.serve_forever()
    finally:
        httpd.server_close()
        spark.stop()


if __name__ == "__main__":
    main()

