import requests
from datetime import datetime
import time
import json
import os

from bs4 import BeautifulSoup, SoupStrainer
from urlextract import URLExtract as urlextract
extractor = urlextract()

import sys
# Add functions path
sys.path.append('../')

from drive import get_id, get_content, save_to_file, export_json

"""
Get conent from CNN url
"""
# Get content from Url
def get_content_url(url, category=None):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        # Get title
        title = soup.title.string
        # Get article
        article = soup.find('article').text
        # Get publish date
        date_published = soup.find('meta', property='og:pubdate')['content']
        # Save to content
        content = {
            'title': title,
            'text': article,
            'date_published': date_published,
            'url': url,
            'scrape_date': str(datetime.now()),
            'category': category}
    except:
        print(url + '\n could not be loaded')

        return None, None

    # Return content and html
    return content, response.text


"""
Get urls from feed
"""
def get_urls_from_feed():

    # Settings
    # on hard drive
    path_urls_used = 'urls_used.txt'
    path_count = 'count.txt'
    path_html = 'html_'
    path_content = 'content_'
    # On google drive
    path_content_drive = 'NewsFeed/CNN/Data/content_'
    path_html_drive = 'NewsFeed/CNN/Data/html_'
    path_urls_used_drive = 'NewsFeed/CNN/urls_used.txt'
    path_count_drive = 'NewsFeed/CNN/count.txt'
    sleep_time = 5
    drive = True

    # Get count of files saved already
    if drive:
        file_count = int(get_content(path_count_drive))
    else:
        with open(path_count, 'r') as file:
            file_count = int(file.read())

    # Load urls already used
    empty=False
    if drive:
        content = str(get_content(path_urls_used_drive))
        # Check if file is empty
        if content == '':
            empty = True
            urls_used, time_urls = [], []
        else:
            urls_used, time_urls = zip(*[x.strip().split(' ', 1) for x in content.split('\n')])
    else:
        with open(path_urls_used, 'r') as txt:
            # Check if file is empty
            if not txt.read(1):
                empty = True
                urls_used, time_urls = [], []
            else:
                urls_used, time_urls = zip(*[x.strip().split(' ', 1) for x in txt.readlines()])
    
    # Remove old urls if they are older than 2 days
    if not empty:
        for i in reversed(range(len(urls_used))):
            # Get time delta
            date_url = datetime.strptime(time_urls[i].split('.')[0], '%Y-%m-%d %H:%M:%S')
            if (datetime.now() - date_url).seconds/(60*60*24) > 2:
                del date_url[i]
                del urls_used[i]

    urls_used, time_urls = list(urls_used), list(time_urls)

    # Categorical new feeds
    top_stories_url = 'http://rss.cnn.com/rss/edition.rss'
    latest_url = 'http://rss.cnn.com/rss/cnn_latest.rss'
    world_url = 'http://rss.cnn.com/rss/edition_world.rss'
    business_url = 'http://rss.cnn.com/rss/money_news_international.rss'
    technology_url = 'http://rss.cnn.com/rss/edition_technology.rss'
    science_url = 'http://rss.cnn.com/rss/edition_space.rss'
    entertainment_url = 'http://rss.cnn.com/rss/edition_entertainment.rss'
    sports_url = 'http://rss.cnn.com/rss/edition_sport.rss'
    # List of categorical feeds
    feeds = {
        'world': world_url,
        'business': business_url,
        'science': science_url,
        'technology': technology_url,
        'entertainment': entertainment_url,
        'sports': sports_url,
        'top_stories': top_stories_url,
        'latest': latest_url
    }

    # Loop over all feeds
    for category, url in feeds.items():
        print('Get content of category {}'.format(category))

        # Function to flatten list of lists
        flatten = lambda t: [item for sublist in t for item in sublist]

        # Get feed html
        response = requests.get(url)

        # Get items
        soup = BeautifulSoup(response.text, "html.parser")
        items = soup.findAll('item')

        urls = []
        for item in items:
            # Get all links from item
            links = extractor.find_urls(item.text)
            # Remove feedburner pages
            links = [l for l in links if 'www.cnn.com' in l]
            # Check if link in item
            if len(links) > 0:
                # Sometimes two links are together, split them
                if links[0].count('https://') + links[0].count('http://') > 1:
                    splitted = [('http://' + k) for k in flatten([j.split('http://') for j in flatten([i.split('https://') for i in links])]) if 'www.' in k]
                    if len(splitted) > 0:
                        # Remove weekdays from some urls (like Mon, Tue...)
                        if ',' in splitted[0]:
                            splitted[0] = splitted[0][:-4]
                        urls.append(splitted[0])
                else:
                    urls.append(links[0])

        # Loop through urls
        for url in urls:
            if url not in urls_used:
                time.sleep(sleep_time)
                # Get content, html
                print('Get content...')
                content, html = get_content_url(url, category)

                # Save content and url
                if content != None:
                    with open(path_content + str(file_count), 'w') as f:
                        json.dump(content, f)
                    # Load to drive
                    if drive:
                        export_json(path_content_drive +  str(file_count) + '.json', path_content + str(file_count))
                        os.remove(path_content + str(file_count))
                        save_to_file(path_html_drive + str(file_count) + '.txt', html)
                    else:
                        with open(path_html + str(file_count), 'w') as txt:
                            txt.write(html)

                    file_count += 1

                # Add url to urls used
                urls_used.append(url)
                time_urls.append(str(datetime.now()))

                # Save updated urls used
                tmp = '\n'.join(['{} {}'.format(u,t) for u, t in zip(urls_used, time_urls)])
                if drive:
                    save_to_file(path_urls_used_drive, tmp)
                else:
                    with open(path_urls_used, 'w') as txt:
                        txt.write(tmp)

                # Save updated count
                if drive:
                    save_to_file(path_count_drive, str(file_count))
                else:
                    with open(path_count, 'w') as txt:
                        txt.write(file_count)



get_urls_from_feed()