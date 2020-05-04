#!/usr/bin/env python
# coding: utf-8

import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import StructType, StructField
from pyspark.sql.types import StringType, IntegerType, DoubleType, TimestampType


config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID']=config['AWS']['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY']=config['AWS']['AWS_SECRET_ACCESS_KEY']

# "com.amazonaws:aws-java-sdk-1.7.4",

def create_spark_session():
    spark = (
        SparkSession
        .builder
        .config(
            "spark.jars.packages",
            "org.apache.hadoop:hadoop-aws:2.7.3",
        )
        .getOrCreate()
        )
    return spark


def read_song_data(path, spark):
    song_data_schema = StructType([
        StructField("artist_id", StringType(), False),
        StructField("artist_latitude", StringType(), True),
        StructField("artist_longitude", StringType(), True),
        StructField("artist_location", StringType(), True),
        StructField("artist_name", StringType(), False),
        StructField("song_id", StringType(), False),
        StructField("title", StringType(), False),
        StructField("duration", DoubleType(), False),
        StructField("year", IntegerType(), False)
    ])
    song_data = spark.read.json(path, schema=song_data_schema)
    song_data.createOrReplaceTempView("song_data")

    return song_data


def read_log_data(path, spark):
    log_data_schema = StructType([
        StructField("artist", StringType(), True),
        StructField("auth", StringType(), False),
        StructField("firstName", StringType(), True),
        StructField("gender", StringType(), True),
        StructField("itemInSession", IntegerType(), False),
        StructField("lastName", StringType(), True),
        StructField("length", DoubleType(), True),
        StructField("level", StringType(), False),
        StructField("location", StringType(), True),
        StructField("method", StringType(), False),
        StructField("page", StringType(), False),
        StructField("registration", DoubleType(), True),
        StructField("sessionId", IntegerType(), False),
        StructField("song", StringType(), True),
        StructField("status", IntegerType(), False),
        StructField("ts", DoubleType(), False),
        StructField("userAgent", StringType(), True),
        StructField("userId", StringType(), True)
    ])
    log_data = spark.read.json(path, schema=log_data_schema)

    # filter by actions for song plays
    log_data = log_data.filter(log_data.page == 'NextSong')
    log_data.createOrReplaceTempView("log_data")

    return log_data


def extract_songs_table(spark):
    """Extract columns to create songs table.

    Parameters
    ----------
    spark : spark session

    Returns
    -------
    songs_table : dataframe
    """

    query = """
    select
      distinct song_id, title, artist_id, year, duration
    from song_data
    """
    songs_table = spark.sql(query)

    return songs_table


def extract_artists_table(spark):
    """Extract columns to create artists table.

    Parameters
    ----------
    spark : spark session

    Returns
    -------
    artists_table : dataframe
    """

    query = """
    select
      distinct
      artist_id,
      artist_name as name,
      artist_location as location,
      artist_latitude as latitude,
      artist_longitude as longitude
    from song_data
    """
    artists_table = spark.sql(query)

    return artists_table


def extract_users_table(spark):
    """Extract columns to create users table.

    Parameters
    ----------
    spark : spark session

    Returns
    -------
    users_table : dataframe
    """

    user_query = """
    select
        user_id,
        first_name,
        last_name,
        gender,
        level
    from (
        select
          ts,
          max(ts) over (partition by userId) as last_ts,
          userId as user_id,
          firstName as first_name,
          lastName as last_name,
          gender,
          level
        from log_data
        where userId is not null
          and userId != ''
          and page = 'NextSong'
    ) as user
    where user.ts = user.last_ts
    """
    users_table = spark.sql(user_query)

    return users_table


def extract_time_table(spark):
    """Extract columns to create time table.

    Parameters
    ----------
    spark : spark session

    Returns
    -------
    time_table : dataframe
    """

    time_query = """
    select
      start_time,
      date_format(start_time, 'H') as hour,
      date_format(start_time, 'd') as day,
      date_format(start_time, 'w') as week,
      date_format(start_time, 'M') as month,
      date_format(start_time, 'YYYY') as year,
      date_format(start_time, 'u') as weekday
    from (
    select
      distinct get_datetime(ts) as start_time
      from log_data
      where page = 'NextSong') as t
    """
    get_datetime = F.udf(
        lambda ts: datetime.fromtimestamp(ts // 1000),
        TimestampType()
        )
    _ = spark.udf.register('get_datetime', get_datetime)

    time_table = spark.sql(time_query)
    time_table.createOrReplaceTempView('time')

    return time_table


def extract_songplays_table(spark):
    """Extract columns to create songplays table.

    Parameters
    ----------
    spark : spark session

    Returns
    -------
    songplays_table : dataframe
    """

    song_plays_query = """
    SELECT event.start_time,
           event.userId    AS user_id,
           event.level,
           song.song_id,
           song.artist_id,
           event.sessionId AS session_id,
           event.location,
           event.userAgent AS user_agent,
           time.year,
           time.month
    FROM log_data AS event
             JOIN (
        SELECT s.song_id,
               s.title,
               s.duration,
               s.artist_id,
               artist.name as artist
        FROM song AS s
                 left JOIN artist ON s.artist_id = artist.artist_id
    ) AS song
              ON event.song = song.title
                  AND event.artist = song.artist
                  AND event.length = song.duration
         JOIN time
              ON event.start_time = time.start_time
    WHERE event.page = 'NextSong'
    """
    songplays_table = spark.sql(song_plays_query)
    songplays_table = (
        songplays_table
        .withColumn("songplay_id", F.monotonically_increasing_id())
        )

    return songplays_table


def process_song_data(spark, input_data, output_data):
    # get filepath to song data file
    # song_data_path = os.path.join(input_data, "song_data/*/*/*/*.json")
    song_data_path = input_data + "song_data/*/*/*/*.json"

    # read song data file
    df = read_song_data(song_data_path, spark)

    # extract columns to create songs table
    songs_table = extract_songs_table(spark)

    # write songs table to parquet files partitioned by year and artist
    songs_table_path = os.path.join(output_data, "songs_table.parquet")
    songs_table.write.parquet(
        path=songs_table_path,
        mode="overwrite",
        partitionBy=["year", "artist_id"],
        compression='snappy'
    )

    # extract columns to create artists table
    artists_table = extract_artists_table(spark)

    # write artists table to parquet files
    artists_table_path = os.path.join(output_data, "artists_table.parquet")
    artists_table.write.parquet(
        path=artists_table_path,
        mode="overwrite",
        compression='snappy'
    )


def process_log_data(spark, input_data, output_data):
    # get filepath to log data file
    # log_data_path = os.path.join(input_data, "log_data/*/*/*.json")
    log_data_path = input_data + "log_data/*/*/*.json"

    # read log data file
    df = read_log_data(log_data_path, spark)

    # extract columns for users table
    users_table = extract_users_table(spark)

    # write users table to parquet files
    users_table_path = os.path.join(output_data, 'users_table.parquet')
    users_table.write.parquet(
        path=users_table_path,
        mode='overwrite',
        compression='snappy'
    )

    # create datetime column from original timestamp column
    get_datetime = F.udf(lambda ts: datetime.fromtimestamp(ts // 1000), TimestampType())
    _ = spark.udf.register('get_datetime', get_datetime)
    df = df.withColumn('start_time', get_datetime('ts'))
    df.createOrReplaceTempView('log_data')

    # extract columns to create time table
    time_table = extract_time_table(spark)

    # write time table to parquet files partitioned by year and month
    time_table_path = os.path.join(output_data, 'time_table.parquet')
    time_table.write.parquet(
        path=time_table_path,
        mode='overwrite',
        partitionBy=['year', 'month'],
        compression='snappy'
    )

    # read in song data to use for songplays table
    songs_table_path = os.path.join(output_data, "songs_table.parquet")
    song_df = spark.read.parquet(songs_table_path)
    song_df.createOrReplaceTempView('song')

    # read in artist data to use for songplays table
    artists_table_path = os.path.join(output_data, "artists_table.parquet")
    artist_df = spark.read.parquet(artists_table_path)
    artist_df.createOrReplaceTempView('artist')

    # extract columns from joined song and log datasets to create songplays table
    songplays_table = extract_songplays_table(spark)

    # write songplays table to parquet files partitioned by year and month
    songplays_table_path = os.path.join(output_data, 'songplays_table.parquet')
    songplays_table.write.parquet(
        path=songplays_table_path,
        mode='overwrite',
        partitionBy=['year', 'month'],
        compression='snappy'
    )


def main():
    input_data = config['S3']['INPUT_DATA']
    output_data = config['S3']['OUTPUT_DATA']

    spark = create_spark_session()
    process_song_data(spark, input_data, output_data)
    process_log_data(spark, input_data, output_data)

    spark.stop()


if __name__ == "__main__":
    main()
