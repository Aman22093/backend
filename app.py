
import os
import json
import pandas as pd
import urllib.request
from werkzeug.utils import secure_filename
from flask import Flask, json, jsonify, request
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from transformers.generation.utils import GreedySearchDecoderOnlyOutput
from flask_cors import CORS
from peft import PeftModel
import textwrap
import torch
import time
import os
import re
import requests
from bs4 import BeautifulSoup


user_search_history = []

import torch
torch.cuda.empty_cache()

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# CORS(app)
# cors = CORS(app, resource={
#     r"/*":{
#         "origins":"*"
#     }
# })

app.secret_key = "midas-lab-iiitd2023"
#llama2_7B_chat="NousResearch/Llama-2-13b-chat-hf"
llama2_7B_chat = "NousResearch/Llama-2-7b-hf"
#llama2_7B_chat="mistralai/Mistral-7B-v0.1"
device_index = int(os.environ.get('CUDA_VISIBLE_DEVICE', '6'))
device = torch.device(f'cuda:{device_index}')
model = AutoModelForCausalLM.from_pretrained(llama2_7B_chat, 
                                                load_in_4bit = True,
                                                torch_dtype=torch.float16,
                                                device_map={'':torch.cuda.current_device()})


tokenizer = AutoTokenizer.from_pretrained(llama2_7B_chat, model_max_length=4096)
tokenizer.pad_token = tokenizer.eos_token

def generate_response(prompt, model):
    encoding = tokenizer(prompt, return_tensors="pt")
    input_ids = encoding["input_ids"].cuda()
 
    generation_config = GenerationConfig(
        temperature=0.6,
        top_p=0.95,
        repetition_penalty=1.15,
        reset=False,
        seed=42,
        
    )
    with torch.inference_mode():
        return model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=2048,
        )

def format_response(response: GreedySearchDecoderOnlyOutput) -> str:
    decoded_output = tokenizer.decode(response.sequences[0])
    response = decoded_output.split("### Response:")[1].strip()
    print("\n".join(textwrap.wrap(response)))
    return "\n".join(textwrap.wrap(response))

def ask(prompt, model):
    response = generate_response(prompt, model)
    return format_response(response)


def generate_prompt(data):
    prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction: 

# Your task is to generate a structured summary based on the provided reviews
# Use the following template to structure the information:

# Pros:
# - Aspect: [Performance/Build Quality/Price-Value Ratio/Ease of Use]
#   - Positive feedback: [Percentage] of users expressed satisfaction.
#   - Reasons: Mention what users liked about this aspect.
# Cons:
# - Aspect: [Noise/Plastic Quality/Price/Shaking During Operation]
#   - Negative feedback: [Percentage] of users expressed dissatisfaction.
#   - Reasons: Mention the specific issues users had with this aspect.

# Your code should process the 'reviews' input and generate a summary in this structured format. Ensure that you calculate the percentages of user satisfaction or dissatisfaction based on the reviews.
# Reviews
{data}

### Response:
"""
    return prompt


def generate_question_prompt(response, question):
    prompt = f"""Use the following pieces of information to answer the user's question. If you don't know the answer, just say that you don't know, don't try to make up an answer.
### Context: {response}
### Question: {question}
Only return the helpful answer below and nothing else, and please be precise while answering the question.
### Response:
"""
    return prompt


@app.route('/api')
def home():
    data = {'message': "Server Working !!"}
    return data


@app.route('/api/get_all_url', methods=['GET'])
def get_all_url():
    data = pd.read_json('reviews.json')
    urls = list(data['URL'].values)
    names = list(data['name'].values)
    link="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRsWiHGWhvUts3ud-clad_6KDd3O1UNPx2yJL43wc_G6g&s"
    url_name_list = []
 
    for url, name in zip(urls, names):
      url_name_list.append({
        'name': name,
        'url': url,
        'img_link':link
    })

    response =jsonify(url_name_list)

    # response = jsonify({
    #     'urls': urls,
    #     'name':names
    # })
    response.status_code = 200
    return response
    
@app.route('/api/get_all_name',methods=['GET'])
def get_all_name():
    data = pd.read_json('reviews.json')
    names = list(data['name'].values)
    
    response = jsonify({
        'names': names
    })
    response.status_code = 200
    return response

@app.route('/api/get_search_history',methods=['GET'])
def get_search_history():
    # Remove duplicates based on product_url
    unique_search_history = {entry['product_url']: entry for entry in user_search_history}.values()

    response = jsonify({
        'search_history': list(unique_search_history)
    })
    response.status_code = 200
    return response

# def get_search_history():
   
#     response = jsonify({
#         'search_history':user_search_history 
#     })
#     response.status_code = 200
#     return response




@app.route('/api/set_search_history',methods=['POST'])
def set_search_history():
    search_term = request.json.get('name')
    product_url = request.json.get('url')
    
    # Check if product_url is already present in user_search_history
    for entry in user_search_history:
        print("a")
        if entry['product_url'] == product_url:
            return jsonify({'message': 'Product URL already present in search history'})
        

    # If product_url is not present, add it to user_search_history
    user_search_history.append({
        'search_term': search_term,
        'product_url': product_url
    })

    return jsonify({'message': 'Search history saved successfully'})

# def set_search_history():
#     search_term = request.json.get('name')
#     product_url = request.json.get('url')
    
#     user_search_history.append({
#         'search_term': search_term,
#         'product_url': product_url
        
#     })
#     return jsonify({'message': 'Search history saved successfully'})







# @app.route('/api/get_all_image',methods=['GET'])
# def get_all_image():
#     data = pd.read_json('reviews.json')
#     img_urls = list(data['img_url'].values)
    
#     response = jsonify({
#         'urls': img_urls
#     })
#     response.status_code = 200
#     return response
    
@app.route('/api/generate_summary', methods=['POST'])
def generate_summary():
    start = time.time()

    import os
    os.system("clear")
    print(request.json)
    print(request.data)
    product_link = request.json['product_link']
    
    print(product_link)
    print("abc")
    with open('reviews.json', 'r') as file:
      data = json.load(file)
    # data = pd.read_json('reviews.json')
    print("as")
    
    reviews = list(data['Reviews'])
    reviews_list = []
    token_size = 0

    for each_review in reviews:
      if token_size < 2048:
        reviews_list.append(each_review['Reviews'])
        token_size += len(each_review['Reviews'])
      else:
        break
    
    product_name = data['name']
    img_url = data['img_url']
    URL=data['URL']
    user_search_history.append({
        'search_term': product_name,
        'product_url': URL
    })
    # rating = data['rating']
    # reviews = list(data[data['URL'] == product_link]['Reviews'].values)
    # print(data)
    # reviews_list = []
    # print("a")
    # token_size = 0
    # for each_review in reviews[0]:
    #     if token_size < 2048:
    #         reviews_list.append(each_review['Reviews'])
    #         token_size += len(each_review['Reviews'])
    #     else:
    #         break
    # data = pd.read_json('reviews.json')   
    # product_name = data[data['URL'] == product_link]['name'].values[0]

    # # Fetch the image URL based on the product link
    # img_url = data[data['URL'] == product_link]['img_url'].values[0]
    # rating=data[data['URL'] == product_link]['rating'].values[0]
    prompt = generate_prompt(reviews_list)
    response = ask(prompt, model).replace('</s>', '').replace('\n', '').replace("Aspect X:", "\n-")
    response_text=response
    # pros = re.search(r'Pros:(.*?)Cons:', text, re.DOTALL).group(1).strip()
    # cons= re.search(r'Cons:(.*?)Overall,', text, re.DOTALL).group(1).strip()
    # Use regular expressions to extract Pros and Cons
    pros_match = re.search(r'Pros:(.*?)Cons:', response_text, re.DOTALL)
    cons_match = re.search(r'Cons:(.*?)Note:', response_text, re.DOTALL)
    
    if pros_match:
     pros = pros_match.group(1).strip()
    else:
     pros = "Pros not Found"

    if cons_match:
     cons = cons_match.group(1).strip()
    else:
     cons = "Cons not found"

   
    pros_list = []
    cons_list = []

    if pros_match:
      pros_text = pros_match.group(1).strip()
      pros_list = [line.strip() for line in pros_text.split('*') if line.strip()]
   
    if cons_match:
      cons_text = cons_match.group(1).strip()
      cons_list = [line.strip() for line in cons_text.split('*') if line.strip()]
    
    # Iterate through responseProsList and print each key term
    responseProsList_bold = []
    for item in pros_list:
      key_term, rest_of_text = item.split(':', 1)  # Split at the first colon to separate key term and the rest of the text
      bold_key_term_pros = f"<b>{key_term.strip()}:</b>"  # Wrap the key term with bold tags
      responseProsList_bold.append(bold_key_term_pros + rest_of_text)
    
    responseConsList_bold = []
    for item in cons_list:
      key_term, rest_of_text = item.split(':', 1)  # Split at the first colon to separate key term and the rest of the text
      bold_key_terms_cons = f"<b>{key_term.strip()}:</b>"  # Wrap the key term with bold tags
      responseConsList_bold.append(bold_key_terms_cons + rest_of_text)



    end = time.time()
    resp = jsonify({
    'execution_time': f"{(end - start)} seconds",
    'response': response,
    'img_url': img_url,
    'product_name': product_name,
    # 'rating': rating,
    'responsePros': pros,
    'responseCons': cons,
    'responseProsList':responseProsList_bold,
    'responseConsList':responseConsList_bold
    
     })
    
    resp.status_code = 200

    return resp


@app.route('/api/ask_question', methods=['POST'])
def ask_question():
    start = time.time()
    
    summary = request.json['summary']
    question = request.json['question']
    
    prompt = generate_question_prompt(summary, question)
    response = ask(prompt, model).replace('</s>', '').replace('\n', '')
    
    end = time.time()
    resp = jsonify({
        'question': question,
        'answer': response,
        'execution_time': f"{(end - start)} seconds"
    })
    resp.status_code = 200
    return resp

import requests
from bs4 import BeautifulSoup
import pandas as pd
import re

def expand_read_more(link):
    response = requests.get(link)
    soup = BeautifulSoup(response.text, "html.parser")
    expanded_text = soup.find("div", class_="expanded-content")  # Replace with the actual class used for expanded text
    return expanded_text.get_text() if expanded_text else ""
def remove_read_more(text):
    return re.sub(r'read\s*more', '', text, flags=re.IGNORECASE)
def extract_review_date(review):
    date_element = review.find("div", class_="_2sc7ZR")
    return date_element.get_text() if date_element else ""

def scrape_review_titles(url):
    # Send an HTTP GET request to the URL
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content of the page
        soup = BeautifulSoup(response.text, 'html.parser')
        # Find all review titles with class "_2-N8zT"
        review_titles = soup.find_all(class_="_2-N8zT")
        date=soup.find_all("p",class_="_2sc7ZR")
        certify=soup.find_all("p",class_="_2mcZGG")
# Extract the text from the review titles and store them in a list
        titles_list = [title.get_text() for title in review_titles]
        certify_list=[certi.get_text() for certi in certify]
        # Remove the second word from each element
        modified_list = [', '.join(item.split(', ')[:1]) for item in certify_list]
        modified_list=[]


        print(certify_list)
       # Find all review stars with class "_3LWZlK _1BLPMq" (assuming each title has a corresponding star element)
        review_stars = [star.get_text() for star in soup.find_all("div", class_=["_3LWZlK _1BLPMq", "_3LWZlK _32lA32 _1BLPMq","_3LWZlK _1rdVr6 _1BLPMq"])]
        date_list=[dates.get_text() for dates in date]
        filtered_date_list = date_list[1::2]

# Extract the text from the review stars and store them in a list
        stars_list = [stars  for stars in review_stars]
        review_elements = soup.find_all("div", class_="t-ZTKy")

        # Extract the text from each review element, expanding "Read More" links if present
# Initialize empty lists for reviews and expanded reviews
        reviews_list = []
        expanded_reviews_list = []
        review_dates_list = []
        for review in review_elements:
           review_text = review.get_text()
           read_more_link = review.find("a", string="Read More", href=True, case=False)  # Case-insensitive search
           if read_more_link:
               expanded_text = expand_read_more(read_more_link['href'])
               review_text += "\n" + expanded_text  # Append expanded text to the review
           reviews_list.append(review_text)
           expanded_reviews_list.append(expanded_text if read_more_link else "")  # Use an empty string if no "Read More" link

        lengths = [len(titles_list), len(stars_list), len(reviews_list), len(filtered_date_list), len(modified_list)]

# Find the minimum length to ensure all lists are of the same length
        min_length = min(lengths)

# Trim or pad lists to make them of equal length
        titles_list = titles_list[:min_length]
        stars_list = stars_list[:min_length]
        reviews_list = reviews_list[:min_length]
        filtered_date_list = filtered_date_list[:min_length]
        modified_list = modified_list[:min_length]



    # Create a DataFrame from the lists of titles and stars
        df = pd.DataFrame({'Review Titles': titles_list, 'Review Stars': stars_list,'Reviews':reviews_list,'Date':filtered_date_list,'Certify':modified_list})
        # Convert the 'Reviews' column to strings
        df['Reviews'] = df['Reviews'].astype(str)

        df['Reviews'] = df['Reviews'].str.replace(r'read\s*more', '', case=False)
        print(df)
        return df

    else:
        print(f"Failed to fetch data from {url}. Status code: {response.status_code}")
        return None

def multiple_url(original_url):
    # Step 1: Extract everything up to '/p/'
    match = re.search(r'(.*?)/p/', original_url)
    if match:
      prefix = match.group(1)

# Step 2: Replace 'p' with 'product-reviews'tcd cd
    reviews_url = original_url.replace('/p/', '/product-reviews/')


    df_1 = pd.DataFrame(columns=["Review Stars","Reviews","Date","Certify"])
# Step 3: Extract everything until 'marketplace=FLIPKART'
    match = re.search(r'(.*?)marketplace=FLIPKART', reviews_url)
    if match:
     reviews_url = match.group(1)+'marketplace=FLIPKART'
    num_pages = 5
    review_page_urls = [reviews_url + f'&page={x}' for x in range(1, num_pages + 1)]

# # Print the list of review page URLs
#     for page_url in review_page_urls:
#      df_new=scrape_review_titles(page_url)
#      df_1 = df_1.append(df_new, ignore_index=True)
#     return df_1

# Loop through review_page_urls and append data to df_1
    for page_url in review_page_urls:
     df_new = scrape_review_titles(page_url)
     
     df_1 = pd.concat([df_1, df_new], ignore_index=True)
     json_file_path = 'reviews_data.json'
     json_file_path = '/media/nas_mount/Aman_22093/backend/reviews_data.json'
     df_1.to_json(json_file_path, orient='records', lines=True)
    return df_1

def product_details(url):
   response = requests.get(url)
   soup = BeautifulSoup(response.text, "html.parser")

# Find all elements with class "_1AtVbE col-12-12"
   product_containers = soup.find_all("div", class_="_3k-BhJ")

# Initialize lists to store extracted data
   class_1hKmbr_list = []
   class_21lJbe_list = []

# Loop through product containers
   for product_container in product_containers:
    # Find elements with class "_1hKmbr col col-3-12"
    class_1hKmbr_element = product_container.find_all(class_="_1hKmbr col col-3-12")
    class_1hKmbr_text=[pro.get_text() for pro in class_1hKmbr_element]


    # Find elements with class "_21lJbe" and extract text individually
    class_21lJbe_elements = product_container.find_all(class_="_21lJbe")
    class_21lJbe_texts = [element.get_text() for element in class_21lJbe_elements]

    # Append the extracted data to the lists
    class_1hKmbr_list.append(class_1hKmbr_text)
    class_21lJbe_list.extend(class_21lJbe_texts)  # Extend the list to include all elements
# Ensure that both lists have the same length
   max_length = max(len(class_1hKmbr_list), len(class_21lJbe_list))
   class_1hKmbr_list += [None] * (max_length - len(class_1hKmbr_list))
   class_21lJbe_list += [None] * (max_length - len(class_21lJbe_list))
   flattened_list = [item for sublist in class_1hKmbr_list if sublist is not None for item in sublist]


   min_length = min(len(flattened_list), len(class_21lJbe_list))

# Create a DataFrame with columns of unequal lengths
   data = {'Keys': flattened_list[:min_length], 'values': class_21lJbe_list[:min_length]}
# Create a DataFrame from the extracted data
   df = pd.DataFrame(data)
   return df



def Qna(url):
# Send a GET request to the URL and parse the HTML content
   response = requests.get(url)
   soup = BeautifulSoup(response.text, "html.parser")

# Find all elements with class "_1RWRBu"
   elements_1RWRBu = soup.find_all(class_="_1RWRBu")

# Initialize lists to store extracted data
   data_3PSmm0 = []
   data_2yeNfb = []

# Loop through elements with class "_1RWRBu"
   for element_1RWRBu in elements_1RWRBu:
    # Find elements with class "_3PSmm0"
    element_3PSmm0 = element_1RWRBu.find(class_="_1xR0kG _3cziW5")
    if element_3PSmm0:
        data_3PSmm0.append(element_3PSmm0.get_text())
    else:
        data_3PSmm0.append(None)

    # Find elements with class "_2yeNfb"
    element_2yeNfb = element_1RWRBu.find(class_="_2yeNfb")
    if element_2yeNfb:
        data_2yeNfb.append(element_2yeNfb.get_text())
    else:
        data_2yeNfb.append(None)

# Create a DataFrame from the extracted data
   df = pd.DataFrame({'Questions': data_3PSmm0, 'Answers': data_2yeNfb})
# Remove "Q:" prefix from 'Questions' column
   df['Questions'] = df['Questions'].astype(str)
   df['Answers'] = df['Answers'].astype(str)
   df['Questions'] = df['Questions'].str.replace('^Q:', '', regex=True)

# Remove "A:" prefix from 'Answers' column
   df['Answers'] = df['Answers'].str.replace('^A:', '', regex=True)
   return df
# Now you have a DataFrame with columns 'Outer Text' and 'Inner Texts' containing the extracted elements



def scrape_image_url(url):
    # Fetch the webpage content
    response = requests.get(url)
    class_name="_396cs4 _2amPTt _3qGmMb"
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find the element with the specified class
        image_element = soup.find('img', class_=class_name)
        
        if image_element:
            # Extract the 'src' attribute value of the image tag
            image_url = image_element.get('src')
            return image_url
        else:
            print(f"No element found with class '{class_name}'")
    else:
        print(f"Failed to fetch the webpage. Status code: {response.status_code}")


def scrape_name_from_url(url):
    # Send a GET request to the URL
    try:
        # Send a GET request to the URL
        response = requests.get(url)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse the HTML content of the page using BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')

            # Find the element with class "B_NuCI"
            name_element = soup.find(class_='B_NuCI')

            if name_element:
                # Get the text within the element (the name)
                name = name_element.get_text(strip=True)
                return name
            else:
                return "Class 'B_NuCI' not found on the page."

        else:
            # If the request was unsuccessful, print an error message
            print("Failed to fetch the URL:", response.status_code)
            return None
    
    except Exception as e:
        print("An error occurred:", str(e))
        return None



def combined_df(df_1, df_2, df_3,url):
    json_data_1 = {
        "Reviews": df_1.to_dict(orient="records")
    }

    json_data_2 = {
        "Product_details": df_2.to_dict(orient="records")
    }

    json_data_3 = {
        "QNA": df_3.to_dict(orient="records")
    }
    
    # Merge the separate JSONs into one DataFrame
    combined_json = {

        "URL": url,
        "name":scrape_name_from_url(url),
        "img_url":scrape_image_url(url),
        "Reviews": json_data_1["Reviews"],
        "Product_details": json_data_2["Product_details"],
        "QNA": json_data_3["QNA"],
        "pros": [],  # Initialize pros as an empty list
        "cons": []   # Initialize cons as an empty list
    }
    return combined_json

@app.route('/api/given_url',methods=['POST'])
def get_link():
    data = request.get_json()
    url = data.get('url')  # Assuming the URL is sent as 'url' in the JSON payload

    if url:
        df_1 = multiple_url(url)
        df_2 = product_details(url)
        df_3 = Qna(url)
        combined_json = combined_df(df_1, df_2, df_3,url)
        
        # Define the file path
        file_path = '/media/nas_mount/Aman_22093/backend/reviews.json'

        # Check if the file exists before attempting to delete it
        if os.path.exists(file_path):
            os.remove(file_path)
            

        # Write the combined_json dictionary to a JSON file
        with open(file_path, 'w') as json_file:
            json.dump(combined_json, json_file, indent=4)
        
        return jsonify({"message": "Data scraped and saved as reviews.json"})
    else:
        return jsonify({"error": "Invalid URL"}), 400



if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001, debug=True, use_reloader=False)
"",