from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import time
import datetime
import tqdm
import ssl
import requests
from pytube import YouTube as YT


# SSL for proxied internet access
ssl._create_default_https_context = ssl._create_unverified_context

# SELENIUM options to not close the browser
options = Options()
#options.add_experimental_option("detach", True)
options.add_experimental_option("detach", False)
options.add_argument("--mute-audio")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()),
                          options=options)

# video URLs in LIST format
url_list = [
    "https://youtu.be/hS5CfP8n_js",
    "https://youtu.be/a4LICTpN24s",
    "https://youtu.be/2j3p_aDMTNg",
    "https://youtube.com/shorts/eaMDDseIL7E?feature=share",
    "https://youtube.com/shorts/ZlUj2WQ7_GE?feature=share"
]

#code to watch the youtube video for the entire lenght
initial_start = datetime.datetime.now() #not used now but useful
for url in tqdm.tqdm(url_list,desc="Progress:"):
    myvideo = YT(url, use_oauth=True, allow_oauth_cache=True)
    duration = myvideo.length
    driver.get(url)
    start = datetime.datetime.now() #not used now but useful
    #Window size: width = 1200px, height = 762px.
    # size = driver.get_window_size()760
    driver.set_window_size(900, 760)
    time.sleep(duration)
    # TAKE FINAL TIME
    end = datetime.datetime.now() #not used now but useful
    elapsed = end - start #not used now but useful
#total time
final_elapsed = end - initial_start #not used now but useful
print(f"completed in {final_elapsed}")
print("-----------------------------------")
