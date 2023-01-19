from dataprep.connector import Connector

# You can get ”yelp_access_token“ by following https://www.yelp.com/developers/documentation/v3/authentication
conn_yelp = Connector("yelp", _auth={"access_token": yelp_access_token}, _concurrency = 5)

df = await conn_yelp.query("businesses", term = "Capilano Suspension Bridge Park", location = "Vancouver", _count = 1)

df[["name","display_phone"]]