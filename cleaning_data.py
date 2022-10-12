import os
import unicodedata
import string
import glob
import normalized as norm
import preprocessing as pre
# from underthesea import word_tokenize
from pyvi import ViTokenizer

path_read = 'social_data/*.txt'
path_write = 'social_clean_data/'
 
def find_files(path):
    return glob.glob(path)
    
# Read a file and split into lines
def read_lines(filename):
    text = open(filename, 'r', encoding='utf-8').read().strip() # open() == io.open() in Python 3 only. In Python 2 they are different
    return text

def clean_data(path_read, path_write):
    for filename in find_files(path_read):
        txt = read_lines(filename)
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

        # Step4: Word segmentation (Using Underthesea or Pyvi)
        # txt = word_tokenize(txt, format="text")
        txt = ViTokenizer.tokenize(txt)

        # Step5: Xoa cac ky tu dac biet
        # Xoa cac ky tu dac biet va Stop Words
        txt = pre.remove_spam(txt)
        txt = pre.remove_special_character(txt)
        txt = pre.remove_space(txt)

        # When loading data from file, the path of file in Windows will change "/" to "\\"
        # filename = filename.split('/')
        filename = filename.split('\\')
        filename2 = path_write + filename[-1]
        if os.path.exists(filename2):
            with open(filename2, "w", encoding="utf-8") as f:
                f.write(txt)
        else:
            os.makedirs(os.path.dirname(filename2), exist_ok=True)
            with open(filename2, "w", encoding="utf-8") as f:
                f.write(txt)

clean_data(path_read, path_write)