# Install required libraries
!pip install transformers accelerate bitsandbytes
!huggingface-cli login

# Import libraries
import transformers
import torch
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
import gc

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Load datasets
test_df = pd.read_csv('/content/drive/My Drive/TYP/final_test.csv')

# Check the dataset
print("Testing Data:")
print(test_df.head())

# Define the range for this chunk
start_index = 900  # Starting index for this chunk
end_index = 1058  # Ending index for this chunk

# Clear any existing GPU memory
torch.cuda.empty_cache()
gc.collect()

# Preprocess the data for this chunk
test_chunk = test_df[start_index:end_index]
test_texts = test_chunk['content'].tolist()
test_labels = test_chunk['det_fake_label'].tolist()

# Load Mistral tokenizer and model
model_id = "mistralai/Mistral-7B-Instruct-v0.3"

# Load tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

# Set a padding token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load model with 4-bit quantization
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    load_in_4bit=True 
)

# Create pipeline
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    do_sample=False,
    return_full_text=False
)

# Define the chat template structure
def create_chat_prompt(text):
    messages = [
        {"role": "system", "content": """You are a news classification assistant. Classify the following article into one of these categories:
        - Real News
        - Human-Generated Fake News
        - LLM-Generated Fake News
        
        Analyze the article carefully for:
        1. Factual inconsistencies
        2. Unusual phrasing or repetition
        3. Overly formulaic structure
        4. Lack of specific details that would be present in real news
        5. Stylistic patterns typical of LLMs
        
        Respond with ONLY the category name, nothing else."""},
        
        # Example 1
        {"role": "user", "content": """Dec 9 2022 Flu season is a problem for more than humans Veterinarians animal shelters and kennels are reporting outbreaks of canine flu in spots across the United States. In North Texas the dog flu spread quickly through a shelter called Operation Kindness where 86 of about 150 dogs kept there became sick by mid-November. The New York Times reported. This was just so fast-acting Ed Jamison the shelter's chief executive told the Times. The shelter suspended adoptions for a while. Dog flu is different from kennel cough. Symptoms of dog flu are loss of appetite a moist or dry cough runny nose low-grade fever and lack of energy. A case of dog flu usually takes 7 to 10 days to run its course. Lori Teller DVM a clinical associate professor at Texas A&M's College of Veterinary Medicine and Biomedical Sciences told The Dallas Morning News. It's not super common throughout the US but when it does occur in an area like the Dallas shelter or recently in Waco a lot of dogs can become infected by it she said."""},
        {"role": "assistant", "content": "Real News"},
        
        # Example 2
        {"role": "user", "content": """President Trump arrived like a boss to check out what's going on at the NATO headquarters. They just spent big bucks on a new headquarters Trump's probably wondering why they didn't use that money more wisely: NATO leaders have arranged an itinerary to appeal to the former real estate magnate: a ribbon-cutting of the alliance's glassy new headquarters, followed by a dinner where leaders will be held to a lightning-round speaking schedule to save time. Trump plans to press NATO leaders on defense spending, continuing a line of attack he started as a candidate last year, Secretary of State Rex Tillerson said Wednesday. You can expect the president to be very tough on them, Tillerson said, saying that he expected Trump to tell them: The American people are doing a lot for your security, for our joint security. You need to make sure you're doing your share for your own security as well."""},
        {"role": "assistant", "content": "Human-Generated Fake News"},
        
        # Example 3
        {"role": "user", "content": """A new study has found that pregnant women with COVID-19 are at a greater risk of developing common pregnancy complications such as cesarean delivery preterm birth and fetal and newborn death. The study which was funded by the National Institutes of Health included nearly 2400 pregnant women infected with SARS-CoV-2. The study found that those with moderate to severe infection were more likely to experience adverse pregnancy outcomes including hypertensive disorders of pregnancy postpartum hemorrhage and infection other than SARS-CoV-2. However it is important to note that mild or asymptomatic infection was not associated with increased pregnancy risks. The study's lead author Torri D Metz MD emphasized the importance of vaccination and other precautions to protect pregnant women and their babies from the virus."""},
        {"role": "assistant", "content": "LLM-Generated Fake News"},
        
        # Current query (NO truncation)
        {"role": "user", "content": f"Article: {text}"}
    ]
    
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

# Enhanced post-processing function
def post_process_output(output):
    output = output.strip().lower()
    
    if "real news" in output:
        return "Real News"
    elif "human-generated fake news" in output or ("human" in output and "fake" in output):
        return "Human-Generated Fake News"
    elif "llm-generated fake news" in output or ("llm" in output and "fake" in output) or ("ai" in output and "fake" in output):
        return "LLM-Generated Fake News"
    else:
        return "Real News"  # Default to Real News

# Map predicted categories to labels - identical to Llama
def map_category_to_label(category):
    category = category.lower()
    if "real" in category:
        return 0  # Real News
    elif "human" in category:
        return 1  # Human-Generated Fake News"
    elif "llm" in category or "ai" in category or "generated" in category:
        return 2  # LLM-Generated Fake News
    else:
        return 0  # Default to Real News

# Process data in batches with error handling
def batch_predict(texts, batch_size=5):
    all_predictions = []
    
    for i in range(0, len(texts), batch_size):
        print(f"Processing batch {i} to {min(i+batch_size, len(texts))}")
        
        # Clear memory before each batch
        torch.cuda.empty_cache()
        gc.collect()
        
        # Get batch texts
        batch_texts = texts[i:min(i+batch_size, len(texts))]
        batch_predictions = []
        
        try:
            for text in tqdm(batch_texts):
                prompt = create_chat_prompt(text)
                output = pipeline(
                    prompt,
                    max_new_tokens=15, 
                    temperature=0.0,    # Deterministic output for consistency
                    top_p=1.0
                )
                
                # Extract and post-process the output
                predicted_category = post_process_output(output[0]['generated_text'])
                batch_predictions.append(predicted_category)
                
                # Print for debugging
                if i < 2:
                    print(f"Raw output: {output[0]['generated_text']}")
                    print(f"Processed category: {predicted_category}")
            
            all_predictions.extend(batch_predictions)
            
        except RuntimeError as e:
            print(f"RuntimeError encountered: {e}")
            print("Trying to continue with individual processing...")
            
            # Try processing one by one
            for text in batch_texts:
                try:
                    # Clear some memory
                    torch.cuda.empty_cache()
                    
                    prompt = create_chat_prompt(text)
                    output = pipeline(
                        prompt,
                        max_new_tokens=15,
                        temperature=0.0,
                        top_p=1.0
                    )
                    
                    predicted_category = post_process_output(output[0]['generated_text'])
                    batch_predictions.append(predicted_category)
                    
                except Exception as inner_e:
                    print(f"Error processing individual text: {inner_e}")
                    batch_predictions.append("Real News")  # Default prediction
            
            all_predictions.extend(batch_predictions)
    
    return all_predictions

# Make predictions in batches
print("Making predictions with Mistral model...")
test_predictions = batch_predict(test_texts, batch_size=5)
test_predicted_labels = [map_category_to_label(pred) for pred in test_predictions]

# Save predictions for this chunk
test_chunk['predicted_label'] = test_predicted_labels
test_chunk.to_csv(f'/content/drive/My Drive/TYP/Mistral/mistral_optimized_predictions_{start_index}_{end_index}.csv', index=False)

# Calculate and display accuracy
accuracy = accuracy_score(test_labels, test_predicted_labels)
report = classification_report(test_labels, test_predicted_labels, target_names=['Real News', 'Human-Generated Fake', 'LLM-Generated Fake'])

print(f"\nAccuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(report)

print(f"\nPredictions for chunk {start_index}-{end_index} saved.")
