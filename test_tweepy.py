import tweepy


consumer_key = "5odJPJM3DongS04XAklivg2bu"
consumer_secret = "kXcyZgVXjSeDodHF0ZZhvvQTkVib5jSn9Mke7o1GyjkV5F38DF"
access_token = "1184317041924136961-Pp2pNmon5sZakMRNoZpninAy7aLMdj"
access_token_secret = "syMcq0Cp7cVtS3iuFhGXWlk8HtVjXVb4PFTNCYX5B8JR5"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

MAX_TWEETS = 200
for tweet in tweepy.Cursor(api.search, q='#holidayshopping').items(MAX_TWEETS):
    # Do something
    print(tweet.text.split("\t")[0])
    #import pdb; pdb.set_trace()
