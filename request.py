import requests

url = 'http://localhost:5000/api'

test_dict = {'followers_count': 36006,
 'has_mobile': 1,
 'has_photo': 1,
 'status': False,
 'city': 'Москва',
 'country': 'Россия',
 'occupation': 'work',
 'skype': False,
 'religion_id': 'nan',
 'smoking': 'nan',
 'life_main': 'nan',
 'relatives': False,
 'alcohol': 'nan',
 'university_name': 'nan',
 'instagram': False,
 'sex': 2}

r = requests.post(url, json=test_dict)
print(r.json())