#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import requests # library to handle requests
from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe
# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors
# import k-means from clustering stage
from sklearn.cluster import KMeans
get_ipython().system(' pip install geocoder')
import geocoder


# In[2]:


data=pd.read_html("https://en.wikipedia.org/wiki/List_of_districts_of_Istanbul")[0]
data.head()


# In[3]:


data=data.drop([39,40])


# In[4]:


data=data.drop(columns=['Mensual household income TL(USD)'],axis=1)


# In[5]:


data.head()


# In[6]:


from geopy.exc import GeocoderTimedOut 
from geopy.geocoders import Nominatim    
longitude = [] 
latitude = [] 
def findGeocode(District): 
        
    try: 
          
       
        geolocator = Nominatim(user_agent="https://en.wikipedia.org/wiki/List_of_districts_of_Istanbul") 
          
        return geolocator.geocode(District) 
      
    except GeocoderTimedOut: 
        return findGeocode(District)     
  
  
for i in (data["District"]): 
      
    if findGeocode(i) != None: 
           
        loc = findGeocode(i) 
          
         
        latitude.append(loc.latitude) 
        longitude.append(loc.longitude) 
       
   
    else: 
        latitude.append(np.nan) 
        longitude.append(np.nan) 


# In[7]:


data["Longitude"] = longitude 
data["Latitude"] = latitude 


# In[8]:


data.head()


# In[9]:


data=data.drop([41,42])
data


# In[10]:


data.shape


# In[11]:


get_ipython().run_line_magic('pip', 'install folium')


# In[12]:


import folium


# In[13]:


address = 'Istanbul'

geolocator = Nominatim(user_agent="https://en.wikipedia.org/wiki/List_of_districts_of_Istanbul")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of Istanbul are {}, {}.'.format(latitude, longitude))


# In[14]:


map_Istanbul = folium.Map(location=[latitude, longitude], zoom_start=11)

 # add markers to map
for lat, lng, label in zip(data['Latitude'], data['Longitude'], data['District']):
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_Istanbul) 
    
map_Istanbul


# In[15]:


CLIENT_ID = '1BALN0HBRJ5OIF4NSILGCIZMLRTMW5A4HD1NWCXEQVYIA5WV' # your Foursquare ID
CLIENT_SECRET = 'RCRNLG3EFFQDXZ4QZ3YJEX20ZHTGKHJCYBTBWGXWX4PRZOT1' # your Foursquare Secret
VERSION = '20180604'
LIMIT = 30
print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)


# In[16]:


data.loc[0, 'District']


# In[17]:


neighborhood_latitude = data.loc[0, 'Latitude']
neighborhood_longitude = data.loc[0, 'Longitude'] 

neighborhood_name = data.loc[0, 'District']

print('Latitude and longitude values of {} are {}, {}.'.format(neighborhood_name, 
                                                               neighborhood_latitude, 
                                                               neighborhood_longitude))


# In[18]:


LIMIT = 100

radius = 500

url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
    CLIENT_ID, 
    CLIENT_SECRET, 
    VERSION, 
    neighborhood_latitude, 
    neighborhood_longitude, 
    radius, 
    LIMIT)
url 


# In[19]:


import json
results = requests.get(url).json()
print('json')


# In[20]:


def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']
        
    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']


# In[21]:


venues = results['response']['groups'][0]['items']
    
nearby_venues = json_normalize(venues) # flatten JSON

# filter columns
filtered_columns = ['venue.name', 'venue.categories', 'venue.location.lat', 'venue.location.lng']
nearby_venues =nearby_venues.loc[:, filtered_columns]

# filter the category for each row
nearby_venues['venue.categories'] = nearby_venues.apply(get_category_type, axis=1)

# clean columns
nearby_venues.columns = [col.split(".")[-1] for col in nearby_venues.columns]

nearby_venues.head()


# In[22]:


print('{} venues were returned by Foursquare.'.format(nearby_venues.shape[0]))


# In[23]:


print ('{} unique categories in Adalar'.format(nearby_venues['categories'].value_counts().shape[0]))


# In[24]:


print (nearby_venues['categories'].value_counts()[0:15])


# Explore Neighborhoods in Istanbul¶
# 

# In[28]:


def getNearbyVenues(names, latitudes, longitudes, radius=500):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']

    
    return(nearby_venues)


# In[29]:


Istanbul_venues = getNearbyVenues(names=data['District'],
                                   latitudes=data['Latitude'],
                                   longitudes=data['Longitude']
                                  )


# In[30]:


Istanbul_Venue_Category=(Istanbul_venues['Venue Category'].value_counts().index.tolist())
print(Istanbul_Venue_Category)


# In[31]:


Istanbul_Venues_only_restaurant = Istanbul_venues[Istanbul_venues['Venue Category'].str.contains( 'Restaurant')].reset_index(drop=True)
Istanbul_Venues_only_restaurant.index = np.arange(1, len(Istanbul_Venues_only_restaurant )+1)


# In[32]:


print (Istanbul_Venues_only_restaurant['Venue Category'].value_counts())


# In[33]:


print('There are {} uniques categories.'.format(len(Istanbul_Venues_only_restaurant['Venue Category'].unique())))


# In[34]:


Istanbul_Dist_Venues_Top12 = Istanbul_Venues_only_restaurant['Venue Category'].value_counts()[0:12].to_frame(name='frequency')
Istanbul_Dist_Venues_Top12=Istanbul_Dist_Venues_Top12.reset_index()

Istanbul_Dist_Venues_Top12.rename(index=str, columns={"index": "Venue_Category", "frequency": "Frequency"}, inplace=True)
Istanbul_Dist_Venues_Top12


# In[35]:


import seaborn as sns
from matplotlib import pyplot as plt

s=sns.barplot(x="Venue_Category", y="Frequency", data=Istanbul_Dist_Venues_Top12)
s.set_xticklabels(s.get_xticklabels(), rotation=45, horizontalalignment='right')

plt.title('12 Most Frequently Occuring Venues in  Major Districts of Istanbul', fontsize=15)
plt.xlabel("Venue Category", fontsize=15)
plt.ylabel ("Frequency", fontsize=15)
plt.savefig("Most_Freq_Venues1.png", dpi=300)
fig = plt.figure(figsize=(18,7))
plt.show()


# In[36]:


print ("Shape of the Data-Frame with Venue Category only Restaurant: ", Istanbul_Venues_only_restaurant.shape)
Istanbul_Venues_only_restaurant.head(10)


# In[37]:


Istanbul_Venues_restaurant = Istanbul_Venues_only_restaurant.groupby(['Neighborhood'])['Venue Category'].apply(lambda x: x[x.str.contains('Restaurant')].count())
Istanbul_Venues_restaurant


# In[54]:


Istanbul_Venues_restaurant_df  = Istanbul_Venues_restaurant.to_frame().reset_index()
Istanbul_Venues_restaurant_df.columns = ['Neighborhood', 'Number of Restaurant']
Istanbul_Venues_restaurant_df.index = np.arange(1, len(Istanbul_Venues_restaurant_df)+1)
list_rest_no =Istanbul_Venues_restaurant_df['Number of Restaurant'].to_list()
list_dist =Istanbul_Venues_restaurant_df['Neighborhood'].to_list()
print(list_rest_no)
print(list_dist)


# In[62]:


Istanbul_onehot = pd.get_dummies(Istanbul_Venues_only_restaurant[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
Istanbul_onehot['Neighborhood'] = Istanbul_Venues_only_restaurant['Neighborhood'] 
Istanbul_onehot=Istanbul_onehot[['Neighborhood','Afghan Restaurant', 'American Restaurant', 'Arepa Restaurant',
       'Argentinian Restaurant', 'Asian Restaurant', 'Chinese Restaurant',
       'Comfort Food Restaurant', 'Doner Restaurant', 'Dumpling Restaurant',
       'Eastern European Restaurant', 'English Restaurant',
       'Falafel Restaurant', 'Fast Food Restaurant', 'Halal Restaurant',
       'Italian Restaurant', 'Kebab Restaurant', 'Kokoreç Restaurant',
       'Kumpir Restaurant', 'Mediterranean Restaurant',
       'Middle Eastern Restaurant', 'Restaurant', 'Seafood Restaurant',
       'Sushi Restaurant', 'Syrian Restaurant', 'Thai Restaurant',
       'Theme Restaurant', 'Turkish Home Cooking Restaurant',
       'Turkish Restaurant', 'Vegetarian / Vegan Restaurant']]
Istanbul_onehot.head()


# In[63]:


Istanbul_onehot.shape


# In[64]:


Istanbul_grouped = Istanbul_onehot.groupby('Neighborhood').mean().reset_index()
Istanbul_grouped


# In[65]:


top_venues = 5

for hood in Istanbul_grouped['Neighborhood']:
    print("----"+hood+"----")
    temp = Istanbul_grouped[Istanbul_grouped['Neighborhood'] == hood].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(top_venues))
    print('\n')


# In[66]:


def return_most_common_venues(row, top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:top_venues]


# In[67]:


top_venues = 10

indicators = ['st', 'nd', 'rd']

columns = ['Neighborhood']
for ind in np.arange(top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighborhood'] = Istanbul_grouped['Neighborhood']

for ind in np.arange(Istanbul_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(Istanbul_grouped.iloc[ind, :], top_venues)

neighborhoods_venues_sorted.head()


# Cluster Neighborhoods
# 

# Run k-means to cluster the neighborhood into 5 clusters.

# In[68]:


kclusters = 5
Istanbul_grouped_clustering = Istanbul_grouped.drop('Neighborhood', 1)
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(Istanbul_grouped_clustering)
kmeans.labels_[0:10]


# In[97]:


Istanbul_merged = data
Istanbul_merged.rename(columns={'District':'Neighborhood'}, inplace=True)
Istanbul_merged = Istanbul_merged.join(neighborhoods_venues_sorted.set_index('Neighborhood'), on='Neighborhood')
Istanbul_merged.head() 


# In[98]:


Istanbul_merged=Istanbul_merged.drop([0,5])


# In[99]:


Istanbul_merged.head(10) 


# In[106]:


Istanbul_merged.loc[Istanbul_merged['Cluster Labels'] == 0, Istanbul_merged.columns[[0] + list(range(5, Istanbul_merged.shape[1]))]]


# In[87]:


Istanbul_merged.loc[Istanbul_merged['Cluster Labels'] == 1, Istanbul_merged.columns[[0] + list(range(5, Istanbul_merged.shape[1]))]]


# In[88]:


Istanbul_merged.loc[Istanbul_merged['Cluster Labels'] == 2, Istanbul_merged.columns[[0] + list(range(5, Istanbul_merged.shape[1]))]]


# In[89]:


Istanbul_merged.loc[Istanbul_merged['Cluster Labels'] == 3, Istanbul_merged.columns[[0] + list(range(5, Istanbul_merged.shape[1]))]]


# In[90]:


Istanbul_merged.loc[Istanbul_merged['Cluster Labels'] == 4, Istanbul_merged.columns[[0] + list(range(5, Istanbul_merged.shape[1]))]]


# In[ ]:




