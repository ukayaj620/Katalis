import os
import discord
import numpy as np
import math
from dotenv import load_dotenv
from analiser import Analiser

load_dotenv()
token = os.getenv('DISCORD_TOKEN')

client = discord.Client()

print("Loading Model...")
an = Analiser()
model = an.load_model()


@client.event
async def on_message(message):
    if message.author == client.user:
        return

    # u"\u2588" u"\u2581"
    y = an.model_load.predict_proba(
        np.array([an.tfidf_data.transform(message.content)]))
    verdict = an.getBinaryResult(y)

    y1 = y[0][0]
    y2 = y[0][1]
    while math.floor(y1) == math.floor(y2):
        y1 = y1*10 - math.floor(y1)*10
        y2 = y2*10 - math.floor(y2)*10

    response = "|"
    for i in range(11):
        if(i < y1):
            response += u"\u2588"
        else:
            response += u"\u2581"
    response += "| %f%% Formal\n" % y[0][0]

    response += "|"
    for i in range(11):
        if(i < y2):
            response += u"\u2588"
        else:
            response += u"\u2581"
    response += "| %f%% Informal\n" % y[0][1]

    response += 'Your statement is ' + verdict

    await message.channel.send(response)

client.run(token)
