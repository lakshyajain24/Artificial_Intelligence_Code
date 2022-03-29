# Sentiment-Analysis

Since there was a lack of enough data for training the model, we took a transfer learning approach to solve it. We took the Jewelry data from Amazon Reviews data until 2018. It was 14 GB of JSON data.



1. Process a large dataset and convert it into a smaller dataset after cleaning and preprocessing the data to convert into a balanced_reviews.csv.
2. Read the balanced_reviews dataset and use the Bag of Words Model to train a model and save it into a pickle file.
3. Using Pre-trained models to achieve the same.
4. Create a Web Scraper to scrape the data from Etsy to test the newly created model by saving the reviews/feedback into a database.
5. Create a Dashboard using the Dash framework to integrate the picked file and predict the sentiments of the stored data from the database.

 
![Screenshot 2021-08-27 093027](https://user-images.githubusercontent.com/45160091/131069807-d31ed4ce-a9ba-4129-aa63-e7fb03679cc2.png)

