import requests
import csv
import os
import glob
import pandas as pd

# Delete features file and audio files if already exist.
if os.path.exists("features.csv"):
    os.remove("features.csv")

if os.path.exists("audio"):
    files = glob.glob('audio/*')
    for f in files:
        print(f"Deleting {f}")
        os.remove(f)
else:
    os.makedirs('audio')


# API url must include query parameters. See https://www.xeno-canto.org/help/search
# Query takes all birds from the United States or Canada with quality A or B.
base_urls = ['https://www.xeno-canto.org/api/2/recordings?query=cnt:united_states+q_gt:C',
             'https://www.xeno-canto.org/api/2/recordings?query=cnt:canada+q_gt:C']

for url in base_urls:
    print(url)
    r = requests.get(url, allow_redirects=True)

    pages = r.json()['numPages']
    for page in range(1, pages+1):
        print(f'page {page}')
        page_url = f'{url}&page={page}'
        r_page = requests.get(page_url, allow_redirects=True)
        print(r_page.text[:100])

        for result in r_page.json()['recordings']:
            # Download features to csv for filtering
            fields = [result['id'], result['gen'], result['sp'], result['ssp'], result['en'], result['cnt'], result['loc'],
                      result['type'], result['q'], result['length'], result['bird-seen'], result['file']]
            with open(r'features.csv', 'a', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(fields)

print('United States and Canada features downloaded')



# Read the features csv file and add the column names
features = pd.read_csv('features.csv', header=None, encoding='latin')
features.columns= ['id', 'gen', 'sp', 'ssp', 'en', 'cnt', 'loc', 'type', 'q', 'length', 'bird-seen', 'file']

# Select birds from our list - birds were selected as having most recordings and being present in the state of Michigan
selected_birds = ['Red-winged Blackbird', 'Common Yellowthroat', 'Northern Cardinal',
                  'Carolina Wren', 'Red Crossbill', 'Spotted Towhee']
features = features[features.en.isin(selected_birds)]


# Filter type feature to song only or call only
def song_or_call(row):
    if 'call' in row.type and 'song' not in row.type:
        return 'call'
    if 'song' in row.type and 'call' not in row.type:
        return 'song'

features['type'] = features['type'].str.lower()
features['category'] = features.apply(lambda row: song_or_call(row), axis=1)
features = features[(features.category == 'call')| (features.category == 'song')]

# Save filtered features file to retrieve labels easily when training classification models
features.to_csv('features_filtered.csv', index=False)
print('Audio files selected')


# Download audio files
if os.path.exists("audio"):
    files = glob.glob('audio/*')
    for f in files:
        print(f"Deleting {f}")
        os.remove(f)
else:
    os.makedirs('audio')

for idx, row in features.iterrows():
    # Download audio file to audio folder
    f = requests.get(f"http:{row.file}", allow_redirects=True)
    open(f"audio/{row.id}.mp3", 'wb').write(f.content)
    print(row.id)

print('Audio files downloaded')