import requests

def get_encoded_message(response):
	message = ""
	genexpr = (response[i] for i in range(0, 498*400, 498))
	return message.join(list(genexpr))

if __name__ == "__main__":
	payload = {'netid':'otaha2', 'name':'Omar Taha'}
	url = "https://courses.engr.illinois.edu/ece498icc/sp2020/lab1_string.php"
	
	r = requests.post(url, data=payload)

	encoded_message = get_encoded_message(r.text)
	print(encoded_message)
