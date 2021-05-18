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


# # API url must include query parameters. See https://www.xeno-canto.org/help/search
# # Query takes all birds from Michigan with quality A or B, length <= to 120 seconds.
base_urls = ['https://www.xeno-canto.org/api/2/recordings?query=cnt:united_states+q_gt:C+len_lt:120',
             'https://www.xeno-canto.org/api/2/recordings?query=cnt:canada+q_gt:C+len_lt:120']

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
            # Download features to csv for classification
            fields = [result['id'], result['gen'], result['sp'], result['ssp'], result['en'], result['cnt'], result['loc'],
                      result['type'], result['q'], result['length'], result['bird-seen'], result['file']]
            with open(r'features.csv', 'a', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(fields)
print('downloaded all features')