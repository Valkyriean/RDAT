from tweepy import Client


BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAIHFbAEAAAAA0oBVG5orLLErnyAqw2po3fOae5w%3D4lgZWoMOyGG496F2aNACoKOdDCaDnxqret6oFPLToE244O6Tx6"


def topic():


    id = 1243807211815649285
    client = Client(BEARER_TOKEN)

    tw = client.get_tweet(id, tweet_fields=["context_annotations", "created_at","entities","public_metrics"])
    # uid = tw.data.author_id
    print(tw)
    print(tw.data.context_annotations)
    print(tw.data.created_at)
    print(tw.data.entities)
    print(tw.data.public_metrics)

    for i in tw.data:
        print(i)
    # print(uid)
    # print(tw.includes["users"][0].public_metrics)
    
topic()