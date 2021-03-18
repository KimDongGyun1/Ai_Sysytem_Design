from gtts import gTTS
from playsound import playsound
import os

news = '안녕하세요 저는 김동균이라고 합니다.'

tts = gTTS(text=news, lang='ko')
tts.save("news_test.mp3")
playsound("news_test.mp3",False)
playsound("bgm.mp3",False)
os.remove("news_test.mp3")