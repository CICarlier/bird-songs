import requests
import csv
import os
import glob

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
url = 'https://www.xeno-canto.org/api/2/recordings?query=cnt:brazil&page=5'
r = requests.get(url, allow_redirects=True)



for result in r.json()['recordings'][:10]:
    # Download audio file to audio folder
    f = requests.get(f"http:{result['file']}", allow_redirects=True)
    open(f"audio/{result['id']}.mp3", 'wb').write(f.content)

    # Download features to csv for classification
    fields = [result['id'], result['gen'], result['sp'], result['ssp'], result['en'], result['cnt'], result['type'],
              result['q'], result['length'], result['bird-seen']]
    with open(r'features.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(fields)