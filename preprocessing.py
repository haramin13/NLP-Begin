import unicodedata
import regex as re
import emoji

# Xoa HTML code
def remove_html(txt):
    return re.sub(r'<[^>]*>', '', txt)

# Xoa URL
def remove_url(txt):
    txt = re.sub(r'https\S+', '', txt, flags=re.MULTILINE)
    txt = re.sub(r'http\S+', '', txt, flags=re.MULTILINE)
    return txt

# Xoa emoji
def remove_emoji(txt):
    return emoji.replace_emoji(txt, replace='')

def remove_emoji_unicode_type(data):
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoj, '', data)

# Xoa khoang trang thua (bao gom ca ky tu xuong dong)
def remove_space(txt):
    return re.sub(r'\s+', ' ', txt).strip()

# Xoa hashtag
def remove_hashtag(txt):
    return re.sub("#[A-Za-z0-9_àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆĐÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴÂĂĐÔƠƯ]+","", txt)

# Xoa cac phan spam trong bai dang
def remove_spam(txt):
    c = '---'
    pattern = c + '.*'
    txt = re.sub(pattern, '', txt)
    # Xoa tu spam
    txt = re.sub(r"Cẩm nang mua sắm Miền Bắc:", '', txt)
    txt = re.sub(r"Cẩm nang mua sắm Miền Trung & Nam:", '', txt)
    txt = re.sub(r"App Store:", '', txt)
    txt = re.sub(r"Google Play:", '', txt)
    return txt

# Xoa cac ky tu dac biet va Stop words (tuy chinh theo dang du lieu ban dau)
def remove_special_character(txt):
    # Xoa dau ba cham
    txt = re.sub(r'(\W)\1+', '', txt)
    # Xoa dau gach noi
    txt = re.sub(r'-', '', txt)
    return txt