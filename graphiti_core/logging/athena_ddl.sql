-- Athena DDL for Graphiti invocation logs
-- Partitioned by year/month/day/hour for efficient querying
-- Logs are stored as gzip-compressed JSON Lines in S3

CREATE EXTERNAL TABLE IF NOT EXISTS graphiti_llm_invocation_logs (
    `timestamp`     STRING,
    operation       STRING,
    model_id        STRING,
    prompt_name     STRING,
    group_id        STRING,
    input_tokens    INT,
    output_tokens   INT,
    latency_ms      DOUBLE,
    status          STRING,
    error_message   STRING,
    input_preview   STRING,
    metadata        MAP<STRING, STRING>
)
PARTITIONED BY (
    `year`  INT,
    `month` INT,
    `day`   INT,
    `hour`  INT
)
ROW FORMAT SERDE 'org.openx.data.jsonserde.JsonSerDe'
STORED AS INPUTFORMAT 'org.apache.hadoop.mapred.TextInputFormat'
OUTPUTFORMAT 'org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat'
LOCATION 's3://spike-graphiti-logs/GraphitiLogs/'
TBLPROPERTIES (
    'has_encrypted_data' = 'false',
    'serialization.encoding' = 'utf-8'
);

-- Auto-discover partitions
MSCK REPAIR TABLE graphiti_llm_invocation_logs;

-- Example queries:

-- Token usage by prompt type (last 7 days)
-- SELECT prompt_name,
--        COUNT(*) AS calls,
--        SUM(input_tokens) AS total_input,
--        SUM(output_tokens) AS total_output,
--        AVG(latency_ms) AS avg_latency_ms
-- FROM graphiti_llm_invocation_logs
-- WHERE year = 2026 AND month = 3
-- GROUP BY prompt_name
-- ORDER BY total_input DESC;

-- Error rate by model
-- SELECT model_id, status, COUNT(*) AS cnt
-- FROM graphiti_llm_invocation_logs
-- GROUP BY model_id, status;

-- Embedding call latency distribution
-- SELECT APPROX_PERCENTILE(latency_ms, ARRAY[0.5, 0.9, 0.99]) AS p50_p90_p99
-- FROM graphiti_llm_invocation_logs
-- WHERE operation = 'embedding.create';
