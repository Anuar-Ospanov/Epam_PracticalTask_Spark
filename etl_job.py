from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType
import geohash2  # Switching to geohash2 for geohash encoding

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Join Weather and Restaurant Data") \
    .getOrCreate()

# Load the restaurant data (as CSV)
restaurant_df = spark.read.csv('restaurant_csv/*.csv', header=True, inferSchema=True)

# Load the weather data (as Parquet)
weather_df = spark.read.parquet('weather_parquet/*.parquet')

# Rename the 'lat' and 'lng' columns to avoid conflict
restaurant_df = restaurant_df.withColumnRenamed('lat', 'restaurant_lat') \
                             .withColumnRenamed('lng', 'restaurant_lng')

weather_df = weather_df.withColumnRenamed('lat', 'weather_lat') \
                       .withColumnRenamed('lng', 'weather_lng')

# Function to generate a 4-character geohash from latitude and longitude
def generate_geohash(lat, lon):
    if lat is not None and lon is not None:
        return geohash2.encode(lat, lon)[:4]  # Geohash truncated to 4 characters
    else:
        return None  # Return None if lat or lon is missing

# UDF to apply geohashing to each row in the DataFrame
geohash_udf = udf(lambda lat, lon: generate_geohash(lat, lon), StringType())

# Add geohash column to the weather DataFrame
weather_df = weather_df.withColumn(
    "geohash", geohash_udf(col("weather_lat").cast("double"), col("weather_lng").cast("double"))
)

# Add geohash column to the restaurant DataFrame
restaurant_df = restaurant_df.withColumn(
    "geohash", geohash_udf(col("restaurant_lat").cast("double"), col("restaurant_lng").cast("double"))
)

# Perform a left join on the geohash column
df_joined = restaurant_df.join(weather_df, on="geohash", how="left")

# Remove duplicates to prevent data multiplication (optional)
df_joined = df_joined.dropDuplicates()

# Define output path for Parquet files
output_path = 'output/joined_weather_restaurant_data'

# Write the result to Parquet, partitioned by 'geohash' and using overwrite mode to handle re-runs
df_joined.write.mode("overwrite").partitionBy("geohash").parquet(output_path)

# Optionally, you can check if the data is stored properly by reading back
stored_df = spark.read.parquet(output_path)
stored_df.show()

# Stop the Spark session
spark.stop()
