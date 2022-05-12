import re

from regex import R, W

rumour = {}
nonrumour = {}
with open("output/hashtags.csv", "r") as f:
    f.readline()
    for line in f.readlines():
        line = line.strip().split(',')
        tag = line[0]
        nonrumour_count = int(line[-2])
        rumour_count = int(line[-1])
        tag = tag.lower()
        # tag = re.sub(r'[^\w\s]', '', tag)
        tag = ''.join(ch for ch in tag if ch.isalnum())
        tag = tag.replace('ー','')
        nonrumour[tag] = nonrumour.get(tag, 0) + nonrumour_count
        rumour[tag] = rumour.get(tag, 0) + rumour_count
    nonrumour = {k: v for k, v in sorted(nonrumour.items(), key=lambda item: item[1],reverse=True)}
    rumour = {k: v for k, v in sorted(rumour.items(), key=lambda item: item[1],reverse=True)}
    print(nonrumour)
    f = open("output/hashtags_nonrumour.csv", 'w')
    f.write("")
    f.close()
    f = open("output/hashtags_rumour.csv", 'w')
    f.write("")
    f.close()
    with open("output/hashtags_nonrumour.csv", 'a') as w:
        for k,v in nonrumour.items():
            w.write(f"{k},{v}\n")
    with open("output/hashtags_rumour.csv", 'a') as w:
        for k,v in rumour.items():
            w.write(f"{k},{v}\n")