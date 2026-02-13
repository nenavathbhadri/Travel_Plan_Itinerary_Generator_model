import requests
import os
PIXABAY_API_KEY = "25167829-692ac73e994129337d29215fc"  # Replace with your Pixabay API key

def search_images(query, num_images=5):
    url = f"https://pixabay.com/api/?key={PIXABAY_API_KEY}&q={query}&image_type=photo&per_page={num_images}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad status codes
        
        # Debugging: Print the response text
        print("API Response:", response.text)

        data = response.json()
        
        if "hits" in data:
            return [img['webformatURL'] for img in data['hits']]
        else:
            print("Error: No hits found in the response.")
            return []
    except requests.exceptions.RequestException as e:
        print(f"Error fetching images: {e}")
        return []
    except ValueError as e:
        print(f"Error decoding JSON: {e}")
        return []

def download_images(image_urls, save_dir="images"):
    os.makedirs(save_dir, exist_ok=True)
    
    for i, url in enumerate(image_urls):
        try:
            response = requests.get(url)
            response.raise_for_status()
            with open(f"{save_dir}/image_{i}.jpg", "wb") as f:
                f.write(response.content)
        except requests.exceptions.RequestException as e:
            print(f"Error downloading image {url}: {e}")

