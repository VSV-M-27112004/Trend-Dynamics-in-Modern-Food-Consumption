from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, split
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import pandas as pd
from pyspark.sql.functions import split, size, length, col, explode


spark = SparkSession.builder.appName("Foodcom Visualizations").getOrCreate()

recipes_df = spark.read.option("header", True).csv("hdfs:///user/jagadeesh/datasets/foodcom/recipes.csv")

recipes_df = recipes_df.withColumn("ingredients_list", split(col("ingredients"), ","))
recipes_df = recipes_df.withColumn("num_ingredients", size(col("ingredients_list")))
recipes_df = recipes_df.withColumn("steps_list", split(col("steps"), ","))
recipes_df = recipes_df.withColumn("num_steps", size(col("steps_list")))

recipes_df = recipes_df.withColumn("description_length", length(col("description")))

tags_df = recipes_df.select(explode(split(col("tags"), ",")).alias("tag"))
tags_count = tags_df.groupBy("tag").count().orderBy(col("count").desc()).limit(20)
tags_pd = tags_count.toPandas()
plt.figure(figsize=(10, 6))
sns.barplot(y="tag", x="count", data=tags_pd, palette="coolwarm")
plt.title("Top 20 Most Common Tags")
plt.xlabel("Count")
plt.ylabel("Tag")
plt.savefig("/home/jagadeesh/top_tags.png")
plt.close()

ingredients_df = recipes_df.select(explode(split(col("ingredients"), ",")).alias("ingredient"))
ingredients_count = ingredients_df.groupBy("ingredient").count().orderBy(col("count").desc()).limit(20)
ingredients_pd = ingredients_count.toPandas()
plt.figure(figsize=(10, 6))
sns.barplot(y="ingredient", x="count", data=ingredients_pd, palette="viridis")
plt.title("Top 20 Most Common Ingredients")
plt.xlabel("Count")
plt.ylabel("Ingredient")
plt.savefig("/home/jagadeesh/top_ingredients.png")
plt.close()

num_df = recipes_df.select("num_ingredients", "description_length", "num_steps").toPandas()

plt.figure(figsize=(10, 6))
sns.histplot(num_df["num_ingredients"], bins=20, kde=True)
plt.title("Distribution of Number of Ingredients")
plt.xlabel("Number of Ingredients")
plt.savefig("/home/jagadeesh/hist_num_ingredients.png")
plt.close()

plt.figure(figsize=(10, 6))
sns.histplot(num_df["description_length"], bins=20, kde=True)
plt.title("Distribution of Description Length")
plt.xlabel("Description Length")
plt.savefig("/home/jagadeesh/hist_description_length.png")
plt.close()

plt.figure(figsize=(8, 6))
sns.scatterplot(x="num_ingredients", y="num_steps", data=num_df)
plt.title("Ingredients vs Steps")
plt.savefig("/home/jagadeesh/scatter_ingredients_steps.png")
plt.close()

plt.figure(figsize=(8, 6))
sns.scatterplot(x="description_length", y="num_ingredients", data=num_df)
plt.title("Description Length vs Number of Ingredients")
plt.savefig("/home/jagadeesh/scatter_desc_ingredients.png")
plt.close()

print("All visualizations done")
spark.stop()
