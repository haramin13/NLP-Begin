import regex as re
import normalized as norm
import preprocessing as pre
import sys
# from underthesea import word_tokenize

# txt = "👉 App Store: https://bom.to/5g0S0U 👉 Google Play: https://bom.to/kij1st #CircleKVietnam #VerytiệnVerylợi #AppthànhviênCKClub #CKClub"
# txt = norm.convert_unicode(txt)
# txt = norm.chuan_hoa_dau_cau_tieng_viet(txt)
# txt = re.sub("#[A-Za-z0-9_]+","", txt)
# print(txt)

# txt = "Hỏi gì đáp nấy 0 2 1"
# # (?i) là flag ignore case (không phân biệt ký tự thường và hoa)
# arr = regex.findall(r'(?i)\b\p{L}+\b', txt)
# print(arr)

# txt = "Moot ngaỳ đẹp trơì"
# txt = norm.convert_unicode(txt)
# print(txt)
# txt = norm.chuan_hoa_dau_cau_tieng_viet(txt)
# txt = pre.remove_emoji(txt)
# print(txt)

txt = "👉 App Store: https://bom.to/5g0S0U 👉 Google Play: https://bom.to/kij1st #CircleKVietnam #VerytiệnVerylợi #AppthànhviênCKClub #CKClub"
txt = norm.convert_unicode(txt)
print(txt)
txt = pre.remove_url(txt)
txt = pre.remove_emoji(txt)
txt = pre.remove_hashtag(txt)
txt = pre.remove_space(txt)
print(txt)
print(len(txt))