# import requests
# import random
# import time

# url = 'https://localhost/vulnerabilities/sqli/'  # Update with your URL
# session = requests.Session()

# # Function to perform the request
# def make_request(id):
#     data = {
#         'Submit': 'Submit',
#         'id': str(id)
#     }
#     try:
#         response = session.post(url, data=data, verify=False)
#         if response.status_code == 200:
#             print(f"Request successful for ID: {id}")
#         else:
#             print(f"Request failed for ID: {id}. Status code:", response.status_code)
#     except Exception as e:
#         print(f"Exception occurred for ID: {id}. Exception:", str(e))

# # Run the requests for 10 minutes
# start_time = time.time()
# duration = 10 * 60  # 10 minutes
# while time.time() - start_time < duration:
#     # Generate a random ID between 1 and 5
#     random_id = random.randint(1, 5)
#     make_request(random_id)

#     # Generate a random delay between 1 and 3 seconds
#     random_delay = random.uniform(1, 30)
#     time.sleep(random_delay)


# import requests
# import random
# import time

# url = 'https://localhost/vulnerabilities/sqli/'  # Update with your URL
# session = requests.Session()

# # Function to perform the request
# def make_request(id):
#     methods = ['GET', 'POST']  # Simulating different request methods
#     headers = {
#         'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',  # Simulating different user agents
#         'Accept-Language': 'en-US,en;q=0.9'  # Simulating different language preferences
#     }
#     try:
#         method = random.choice(methods)
#         headers['Method'] = method
#         response = session.request(method, url, data={'id': str(id)}, headers=headers, verify=False)
#         if response.status_code == 200:
#             print(f"Request successful for ID: {id} using {method} method")
#         else:
#             print(f"Request failed for ID: {id}. Status code:", response.status_code)
#     except Exception as e:
#         print(f"Exception occurred for ID: {id}. Exception:", str(e))

# # Run the requests for 10 minutes
# start_time = time.time()
# duration = 10 * 60  # 10 minutes
# while time.time() - start_time < duration:
#     # Generate a random ID between 1 and 7 being 1 to 5 valid requests
#     random_id = random.randint(1,7)
#     make_request(random_id)

#     # Generate a random delay between 1 and 30 seconds
#     random_delay = random.uniform(1,30)
#     time.sleep(random_delay)
import requests
import random
import time

# Create a session object
with requests.Session() as session:
    url = 'https://localhost/vulnerabilities/sqli/'  # Update with your URL

    # Function to perform the request
    def make_request(id):
        data = {
            'Submit': 'Submit',
            'id': str(id)
        }
        try:
            response = session.post(url, data=data, verify=False)
            if response.status_code == 200:
                print(f"Request successful for ID: {id}")
            else:
                print(f"Request failed for ID: {id}. Status code:", response.status_code)
        except Exception as e:
            print(f"Exception occurred for ID: {id}. Exception:", str(e))

    # Run the requests for 10 minutes
    start_time = time.time()
    duration = 10 * 60  # 10 minutes
    while time.time() - start_time < duration:
        # Generate a random ID with higher probability for certain values
        random_id = random.choices(range(1, 6), weights=[4, 3, 2, 1, 1])[0]
        make_request(random_id)

        # Generate a more natural waiting time using Gaussian distribution (mean = 2, std = 0.5)
        random_delay = max(0, random.gauss(15,14))
        time.sleep(random_delay)

