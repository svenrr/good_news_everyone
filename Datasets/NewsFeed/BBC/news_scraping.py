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
Get conent from BBC url
"""
# Get content from Url
def get_content_url(url, category=None):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        # Get title
        title = soup.title.string
        # Get article
        article = soup.find('article')
        # PromoLink if strings are other linked websites
        article = ' '.join([text.text for text in article.findAll('p') if 'PromoLink' not in str(text)])
        # Get publish date
        date_published = str(response.text).split('datePublished', 1)[1].split(',')[0].split('.')[0].split('"')[-1]
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
    path_content_drive = 'NewsFeed/BBC/Data/content_'
    path_html_drive = 'NewsFeed/BBC/Data/html_'
    path_urls_used_drive = 'NewsFeed/BBC/urls_used.txt'
    path_count_drive = 'NewsFeed/BBC/count.txt'
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

    # Overview news page
    #news_feed_url = 'https://www.bbc.com/news/10628494'
    # Categorical new feeds
    # Top stories
    top_stories_url = 'http://feeds.bbci.co.uk/news/rss.xml'
    # World
    world_url = 'http://feeds.bbci.co.uk/news/world/rss.xml'
    # Business
    business_url = 'http://feeds.bbci.co.uk/news/business/rss.xml'
    # Politics
    politics_url = 'http://feeds.bbci.co.uk/news/politics/rss.xml'
    # Health
    health_url = 'http://feeds.bbci.co.uk/news/health/rss.xml'
    # Education & Family
    education_url = 'http://feeds.bbci.co.uk/news/education/rss.xml'
    # Science & Environment
    science_url = 'http://feeds.bbci.co.uk/news/science_and_environment/rss.xml'
    # Technology
    technology_url = 'http://feeds.bbci.co.uk/news/technology/rss.xml'
    # Entertainment & Arts
    entertainment_url = 'http://feeds.bbci.co.uk/news/entertainment_and_arts/rss.xml'
    # List of categorical feeds
    feeds = {
        'world': world_url,
        'business': business_url,
        'politics': politics_url,
        'health': health_url,
        'education': education_url,
        'science': science_url,
        'technology': technology_url,
        'entertainment': entertainment_url,
        'top_stories': top_stories_url
    }

    # Loop over all feeds
    for category, url in feeds.items():
        print('Category: {}'.format(category))
        # Load url
        response = requests.get(top_stories_url)
        soup = BeautifulSoup(response.text, "html.parser")
        # Get news items
        items = soup.findAll('item')
        # Get urls
        urls = [extractor.find_urls(item.text)[0] for item in items]

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