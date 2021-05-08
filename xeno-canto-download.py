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
# Query takes all birds from Michigan with quality A, B or C, lenght <= to 120 seconds, where bird was seen and no playback was used to lure the bird.
url = "https://www.xeno-canto.org/api/2/recordings?query=loc:michigan&q_gt:D&len_lt:120&bird-seen=yes&playback-used=no"
r = requests.get(url, allow_redirects=True)



for result in r.json()['recordings']:
    # Download audio file to audio folder
    f = requests.get(f"http:{result['file']}", allow_redirects=True)
    open(f"audio/{result['id']}.mp3", 'wb').write(f.content)

    # Download features to csv for classification
    fields = [result['id'], result['gen'], result['sp'], result['ssp'], result['en'], result['cnt'], result['type'],
              result['q'], result['length'], result['bird-seen']]
    with open(r'features.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(fields)