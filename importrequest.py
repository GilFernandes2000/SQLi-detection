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

