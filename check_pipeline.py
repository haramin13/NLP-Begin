from email.contentmanager import raw_data_manager
import regex as re
import normalized as norm
import preprocessing as pre
import sys
import os
# from underthesea import word_tokenize
from pyvi import ViTokenizer

# Load data
raw_data = "social_data/winmart9.txt"
txt = open(raw_data,'r', encoding='utf-8').read().strip() # open() == io.open() in Python 3 only. In Python 2 they are different.
# Check raw data and it's length
# print(txt)
# print(len(txt))
# sys.exit()

# Pipeline preprocessing data
# Step1: Xoa URL, HTML (neu co)
txt = pre.remove_url(txt)
txt = pre.remove_html(txt)

# Step2: Xoa Emoji, Hashtag, Tag, Spam
# Xoa emoji
txt = pre.remove_emoji(txt)
# Xoa hashtag
txt = pre.remove_hashtag(txt)
# Xoa spam
txt = pre.remove_spam(txt)
# Xoa khoang trang thua va ky tu xuong dong (\n)
txt = pre.remove_space(txt)

# Step3.1: Chuan hoa bang ma Unicode
# Data was loaded in unicode format so that it don't need convert anymore
# txt = norm.convert_unicode(txt)

# Step3.2: Chuan hoa kieu go dau tieng viet (Bao gom ca Step3.3)
txt = norm.chuan_hoa_dau_cau_tieng_viet(txt, False)

# Step4: Word segmentation (Using underthesea or pyvi)
# Underthesea
# txt = word_tokenize(txt, format="text")
# Pyvi
txt = ViTokenizer.tokenize(txt)

# Step5: Xoa cac ky tu dac biet
# Xoa cac ky tu dac biet va Stop Words
txt = pre.remove_special_character(txt)
txt = pre.remove_space(txt)

# Check clean data and it's length
print(txt)
print(len(txt))
sys.exit()

# Saving clean data
raw_data = raw_data.split('/')
# When loading data from file, the path of file in Windows will change "/" to "\\"
# clean_data = raw_data.split('\\')
clean_data = 'social_clean_data/' + raw_data[-1]

if os.path.exists(clean_data):
    with open(clean_data, "w", encoding="utf-8") as f:
        f.write(txt)
else:
    os.makedirs(os.path.dirname(clean_data), exist_ok=True)
    with open(clean_data, "w", encoding="utf-8") as f:
        f.write(txt)