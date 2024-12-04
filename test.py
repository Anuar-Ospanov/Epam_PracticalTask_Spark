import unittest
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import FloatType, StructType, StructField, StringType
import requests

# Function to mock geocoding since we don't want to hit the actual API during testing
def mock_geocode_address(address):
    mock_data = {
        "Milan, Italy": (45.4642, 9.1900),
        "Paris, France": (48.8566, 2.3522)
    }
    return mock_data.get(address, (None, None))

# UDF to fetch latitude (same as before but using mock function)
@udf(returnType=FloatType())
def get_lat(city, country, lat):
    if lat is None:  # If latitude is missing
        address = f"{city}, {country}"
        lat, _ = mock_geocode_address(address)
        return lat
    return lat

# UDF to fetch longitude (same as before but using mock function)
@udf(returnType=FloatType())
def get_lng(city, country, lng):
    if lng is None:  # If longitude is missing
        address = f"{city}, {country}"
        _, lng = mock_geocode_address(address)
        return lng
    return lng

# Test case for the geocoding and DataFrame transformation logic
class TestGeocodeFunctionality(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Initialize Spark session before running the tests
        cls.spark = SparkSession.builder.master("local").appName("TestGeocodeFunctionality").getOrCreate()

    def test_geocode_with_missing_coordinates(self):
        # Example data with missing coordinates
        data = [("Milan", "Italy", None, None), ("Paris", "France", 48.8566, 2.3522)]
        columns = ["city", "country", "lat", "lng"]

        # Define schema explicitly
        schema = StructType([
            StructField("city", StringType(), True),
            StructField("country", StringType(), True),
            StructField("lat", FloatType(), True),
            StructField("lng", FloatType(), True)
        ])

        # Create DataFrame
        df = self.spark.createDataFrame(data, schema)

        # Apply the UDFs to update lat/lng
        df_updated = df.withColumn("lat", get_lat(col("city"), col("country"), col("lat"))) \
                       .withColumn("lng", get_lng(col("city"), col("country"), col("lng")))

        # Collect the results into a list of rows for validation
        updated_data = df_updated.collect()

        # Check the updated data and assert the results
        self.assertEqual(updated_data[0]['lat'], 45.4642)  # Milan, Italy latitude
        self.assertEqual(updated_data[0]['lng'], 9.1900)   # Milan, Italy longitude
        self.assertEqual(updated_data[1]['lat'], 48.8566)  # Paris latitude (already present)
        self.assertEqual(updated_data[1]['lng'], 2.3522)   # Paris longitude (already present)

    @classmethod
    def tearDownClass(cls):
        # Stop the Spark session after tests
        cls.spark.stop()

if __name__ == "__main__":
    unittest.main()
