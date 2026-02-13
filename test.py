#initializing Gemini model api
import google.generativeai as genai
import PIL.Image
genai.configure(api_key="AIzaSyCSrdHqh7wkS7TiOhA8NebxDFiQO41gCJ4")
gemini_model = genai.GenerativeModel("gemini-2.0-flash-exp")
print("Gemini Model Loaded")

origin = ["India", "India", "India", "India", "India", "India", "India", "India", "India", "India"]
destination = ["USA", "Russia", "Turkey", "Singapore", "China", "France", "Poland", "Switzerland", "Malaysia", "UK"]
budget = "5000"

for i in range(len(origin)):
    start = origin[i]
    end = destination[i]
    prompt = f"""Generate a trip plan for from {start} to {end} with a budget of ${budget}.
             We are interested in a mix of historical sightseeing, cultural experiences, and delicious food.
             Provide a detailed itinerary for hotels and flights
             """
    plan = ""
    response = gemini_model.generate_content(prompt)
    for chunk in response:
        plan += chunk.text
    with open("model/"+origin[i]+"_"+destination[i]+"_"+budget+".txt", "wb") as file:
        file.write(plan.encode())
    file.close()
    print(plan)
    
