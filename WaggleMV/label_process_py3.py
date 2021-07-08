# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

Ingest label data and export mask

Vivien R
2020-03-06
"""
# -*- coding: utf-8 -*-

import json
from labelbox import Client

api_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjazI2NGZsZHlwNW44MDc1N3JjYXMwaHhoIiwib3JnYW5pemF0aW9uSWQiOiJjazI2NGZsZGdpZ3JiMDgzOGJqb3RncXZ5IiwiYXBpS2V5SWQiOiJjazdnamE3aHc2NTl4MDkxODExNXYydzBiIiwiaWF0IjoxNTgzNTIwNjA2LCJleHAiOjIyMTQ2NzI2MDZ9.3EhmC0KJILx828skp-SbuFM8EHQKPLZSQylHgAt0Aeo"
uri = "https://api.labelbox.com//masks//feature//ck62fkpnq13c00xapg9ejpat9?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjazI2NGZsZHlwNW44MDc1N3JjYXMwaHhoIiwib3JnYW5pemF0aW9uSWQiOiJjazI2NGZsZGdpZ3JiMDgzOGJqb3RncXZ5IiwiaWF0IjoxNTgyMzEyMzA2LCJleHAiOjE1ODQ5MDQzMDZ9.X1NFUG2vc5mI-XYZmXFlwZlRQdQryOtWEseXS-GPiJs"
fpath = 'C:\\Users\\Packman-Field\\Documents\\WaggleMV\\'
jfile = "export-2020-02-21T19_11_46.597Z.json"

f = open(fpath+jfile)
jf = json.load(f)

n = 700
# Masks:
## jf[n]["Label"]["objects"]

# Questionnaire responses:
## Water present: jf[n]["Label"]["classifications"][0]["answer"]["value"]
## Lighting: :    jf[n]["Label"]["classifications"][1]["answer"]["value"]
## Quality :      jf[n]["Label"]["classifications"][2]["answer"]["value"]
