import requests
import argparse
import os

def get_response(prompt, model_name="google/flan-t5-base", api_key=None):
    """
    Get a response from the model

    Args:
        prompt (str): The prompt to send to the model
        model_name (str): Name of the model to use
        api_key (str): Hugging Face API key (required for most models)

    Returns:
        str: The model's response
    """
    api_url = f"https://api-inference.huggingface.co/models/{model_name}"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    payload = {"inputs": prompt}

    try:
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()

        # Most models return a list with 'generated_text'
        if isinstance(result, list) and "generated_text" in result[0]:
            return result[0]["generated_text"]

        # Some models return a dict with error or other formats
        if isinstance(result, dict) and "error" in result:
            return f"[ERROR] {result['error']}"

        return str(result)
    except requests.exceptions.RequestException as e:
        return f"[REQUEST FAILED] {e}"

def run_chat(model_name="google/flan-t5-base", api_key=None):
    """
    Run an interactive chat session with the LLM
    """
    print("Welcome to the Simple LLM Chat! Type 'exit' to quit.")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        response = get_response(user_input, model_name, api_key)
        print(f"LLM: {response}")

def main():
    parser = argparse.ArgumentParser(description="Chat with a Hugging Face LLM")
    parser.add_argument('--model', type=str, default="google/flan-t5-base",
                        help="Hugging Face model name")
    parser.add_argument('--prompt', type=str, help="One-off input prompt")
    parser.add_argument('--api_key', type=str, required=True,
                        help="Hugging Face API key (get one at https://huggingface.co/settings/tokens)")
    args = parser.parse_args()

    if args.prompt:
        print("You:", args.prompt)
        print("LLM:", get_response(args.prompt, args.model, args.api_key))
    else:
        run_chat(model_name=args.model, api_key=args.api_key)

if __name__ == "__main__":
    main()
