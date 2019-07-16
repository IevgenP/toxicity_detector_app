import json
import requests
import pandas as pd
from pandas.io.json import json_normalize


request_body = {
    "data": [
        {
            "id": 1,
            "text": "This was hideous experience that I would like to encounter no more."
        },
        {
            "id": 2,
            "text": "It is funny you mentioned your idiotic considerations. Fuck you bro. And your opinion too."
        },
        {
            "id": 3,
            "text": "Shit, have you seen your fingers? They must be really twisted because you type such a nonsense. Probably your mum and dad didn't put much effort in making you."
        },
        {
            "id": 4,
            "text": "What a waste. Everybody involved had no idea how to make a movie."
        },
        {
            "id": 5,
            "text": "My name is Boris. This film sucks. I get up at 7 o'clock. Washing, dressing, doing morning excersises. I hate this movie."
        },
        {
            "id": 6,
            "text": "You are fucked pal. I took your photo and gave it to Head Hunters. Not the ones who hire people but the ones who find you and kick your ass till you half dead. Check your front door bud, they are probably already arrived."
        },
        {
            "id": 7,
            "text": "I will crush your head against the wall for such blasphemy! I will kill you! You are dead!"
        },
        {
            "id": 8,
            "text": "Yep, I know that you guys are homosexuals!"
        },
        {
            "id": 9,
            "text": "You stupid gays and lesbians! All of you are idiots!"
        },
        {
            "id": 10,
            "text": "As usual, this fishing season brought us lots of joy and competition."
        },
        {
            "id": 11,
            "text": "I think I got an info about location of your home. Well, I might visit you soon... No, I will visit you soon and rip your brains from your head. It won't change much as they provide you with zero mental capabilities. I will destroy you!"
        }
    ]
}

# BdRNN with attention
url = 'http://0.0.0.0:2006/toxicity_clf'
response  = requests.post(url, json=request_body)
print(url)
print(response.text)
print("-"*20)
