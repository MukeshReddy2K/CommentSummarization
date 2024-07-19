#---------------------------------------------------------------------------------------------------------
# interact with ollama api
import ollama
from pprint import pprint

#---------------------------------------------------------------------------------------------------------
def ollama_calls():
    # pprint(ollama.list(), indent=2)
    pprint(ollama.chat(model='llama3', messages=[{'role': 'user', 'content': 'Why is the sky blue?'}]))
    # pprint(ollama.show('llama3'))
    # ollama_chat_call();
    # ollama_streaming_call()

#---------------------------------------------------------------------------------------------------------
def ollama_chat_call():
    response = ollama.chat(model='llama3', messages=[
                            {
                                'role': 'user',
                                'content': 'Why is the sky blue?',
                            },
                            ])
    print(response['message']['content'])

#---------------------------------------------------------------------------------------------------------
def ollama_streaming_call():
    stream = ollama.chat(
                        model='llama3',
                        messages=[{'role': 'user', 'content': 'Why is the sky blue?'}],
                        stream=True,
                        )

    for chunk in stream:
        print(chunk['message']['content'], end='', flush=True)

#---------------------------------------------------------------------------------------------------------
def main():
    print(f"Ollama Python API Client")
    ollama_calls()


#---------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()

#---------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------
