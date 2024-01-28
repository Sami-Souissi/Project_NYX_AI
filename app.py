from typing import List, Union
import re
import sys

from dotenv import load_dotenv, find_dotenv
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import streamlit as st
from ctransformers import AutoModelForCausalLM
import time
import pandas as pd
import subprocess
import os
from deploy_script import udocker_init

def init_page() -> None:
    st.set_page_config(
        page_title="NYX AI",
        layout="wide"
    )
    st.header("NYX AI")
    image_url = "./logo.png" 
    st.sidebar.image(image_url, use_column_width=True, caption="Project NYX AI By Sami Souissi", output_format="auto")

    st.sidebar.title("Options")


def execute_script(script_content: str):
    """
    Function to execute the provided script content.
    """
    try:
        exec(script_content)
    except Exception as e:
        st.error(f"Error executing script: {str(e)}")

def init_messages() -> None:
    col3, col4= st.sidebar.columns(2)
    with col3:
      clear_button = st.button("Clear chat", key="clear",type="primary")
    with col4:
      hint_button = st.button("❔", key="inform")
    if hint_button:
      deploy_message = "To deploy a simple hello world Docker container, simply run `/deploy`"
      st.toast(f"**info** : \n{deploy_message}\n", icon='👋')
      time.sleep(3)
      docker_message = "To interact directly with Docker, run ```/docker [instruction]```"
      st.toast(f"**info** : \n{docker_message}\n", icon='🐳')
      time.sleep(2)
      genie_message = "To Prompt the ai to execute commands dirctly, run `/wish [Prompt]`"
      st.toast(f"**info** : \n{genie_message}\n", icon='🤖')
      time.sleep(.10)
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(
                content="You are a helpfull AI assistant . Your answers Should be in markdown format limited to 500 words.")
        ]
        st.session_state.costs = []
    


def select_llm() -> AutoModelForCausalLM:
    with st.sidebar.expander("Model Selection"):
      model_id = st.radio("Choose LLM:",
                        ("TheBloke/Llama-2-7B-chat-GGML",
                          "TheBloke/Llama-2-7B-GGML",
                          "TheBloke/Llama-2-13B-chat-GGML",
                          "TheBloke/Llama-2-13B-GGML"))

    # Use ctransformers to initialize AutoModelForCausalLM
    

    
    # Check ctransformers doc for more configs
    config = {'max_new_tokens': 512, 'repetition_penalty': 1.1,
              'temperature': 0.8, 'stream': True}
    # Create sliders for each parameter
    with st.sidebar.expander("Model Settings"):
      max_new_tokens = st.slider('max_new_tokens', min_value=1, max_value=1000, value=config['max_new_tokens'])
      repetition_penalty = st.slider('repetition_penalty', min_value=0.1, max_value=2.0, value=config['repetition_penalty'])
      temperature = st.slider('temperature', min_value=0.1, max_value=1.0, value=config['temperature'])
      stream = st.checkbox('stream', value=config['stream'])

      # Update the configuration based on slider values
      config['max_new_tokens'] = max_new_tokens
      config['repetition_penalty'] = repetition_penalty
      config['temperature'] = temperature
      config['stream'] = stream

    llm = AutoModelForCausalLM.from_pretrained(
          model_id,
          model_type="llama",
          # lib='avx2', for CPU use
          gpu_layers=130,  # 110 for 7b, 130 for 13b
          **config
    )
    
    return llm



# Import the StreamHandler

def get_answer(llm, messages) -> str:
    start = time.time()
    NUM_TOKENS = 0
    print('-'*4 + 'Start Generation' + '-'*4)
    
    # Convert messages to a single string before tokenization
    input_text = ' '.join([message.content for message in messages])
    tokens = llm.tokenize(input_text)

    response = ''  # Initialize an empty string to store the generated response

    for token in llm.generate(tokens):
        response += llm.detokenize(token)
        print(llm.detokenize(token), end='', flush=True)
        NUM_TOKENS += 1
        if NUM_TOKENS >= 512:
            break

    time_generate = time.time() - start
    time_generate = time.time() - start
    if NUM_TOKENS>= 512 :
      st.error('Warning: Token limit surpassed', icon="⚠️")
      # st.toast('Warning: Token limit surpassed', icon="⚠️")


      

    # Display metrics using Streamlit components

    # Create a DataFrame for visualization
    data = {
        'Metric': ['Num of generated tokens', 'Time for complete generation', 'Tokens per second', 'Time per token'],
        'Value': [NUM_TOKENS, time_generate, NUM_TOKENS/time_generate, (time_generate/NUM_TOKENS)*1000]
    }

    df = pd.DataFrame(data)
    st.sidebar.markdown("## Language Model Generation Metrics")
    st.sidebar.markdown(f"**Total Num of generated tokens: {NUM_TOKENS}**")

    # Display the DataFrame
    st.sidebar.dataframe(df)

    # Plot a bar chart
    st.sidebar.area_chart(df.set_index('Metric'))
    print('\n')
    print('-'*4 + 'End Generation' + '-'*4)
    print(f'Num of generated tokens: {NUM_TOKENS}')
    print(f'Time for complete generation: {time_generate}s')
    print(f'Tokens per second: {NUM_TOKENS/time_generate}')
    print(f'Time per token: {(time_generate/NUM_TOKENS)*1000}ms')

    return response







def find_role(message: Union[SystemMessage, HumanMessage, AIMessage]) -> str:
    """
    Identify role name from langchain.schema object.
    """
    if isinstance(message, SystemMessage):
        return "system"
    if isinstance(message, HumanMessage):
        return "user"
    if isinstance(message, AIMessage):
        return "assistant"
    raise TypeError("Unknown message type.")


def convert_langchainschema_to_dict(
        messages: List[Union[SystemMessage, HumanMessage, AIMessage]]) \
        -> List[dict]:
    """
    Convert the chain of chat messages in list of langchain.schema format to
    list of dictionary format.
    """
    return [{"role": find_role(message),
             "content": message.content
             } for message in messages]


def llama_v2_prompt(messages: List[dict]) -> str:
    """
    Convert the messages in list of dictionary format to Llama2 compliant format.
    """
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    BOS, EOS = "<s>", "</s>"
    DEFAULT_SYSTEM_PROMPT = f"""You are a helpful, respectful and honest assistant. Always answer as helpfully as possible. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

    if messages[0]["role"] != "system":
        messages = [
            {
                "role": "system",
                "content": DEFAULT_SYSTEM_PROMPT,
            }
        ] + messages
    messages = [
        {
            "role": messages[1]["role"],
            "content": B_SYS + messages[0]["content"] + E_SYS + messages[1]["content"],
        }
    ] + messages[2:]

    messages_list = [
        f"{BOS}{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} {EOS}"
        for prompt, answer in zip(messages[::2], messages[1::2])
    ]
    messages_list.append(
        f"{BOS}{B_INST} {(messages[-1]['content']).strip()} {E_INST}")

    return "".join(messages_list)

if "disabled" not in st.session_state:
    st.session_state["disabled"] = False

def disable():
    st.session_state["disabled"] = True


def execute_docker_script(input_text):
    command = f"udocker {input_text}"
    
    # Use subprocess to run the command and capture the output
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    # Return the terminal output
    return result.stdout
def extract_docker_command(response):
    # Case 2: Plain text with command hint
    match_plain_text = re.search(r'docker rm -f (\S+)', response)
    if match_plain_text:
        return match_plain_text.group(1).strip()

    # Case 1: Markdown-style code block
    match_markdown = re.search(r'```bash\n(.*?)\n```', response, flags=re.DOTALL)
    if match_markdown:
        return match_markdown.group(1).strip()

    # Default: No command found
    return "No command found"


def main() -> None:
    _ = load_dotenv(find_dotenv())

    init_page()
    llm = select_llm()
    init_messages()
    st.sidebar.markdown("---")
    col1, col2= st.sidebar.columns(2)


    with col1:
      button_1 = st.button("Scenario 1")




    # Button in the second column
    with col2:
      button_2 = st.button("Scenario 2")



    # Check if any button is clicked
    if button_2:
      st.session_state.disabled = True
      with st.status("Preparing scenario...", expanded=True) as status:
        st.write("Checking Logs...")
        command_output = subprocess.check_output("sudo chmod 000 logs.txt", shell=True, text=True)
        time.sleep(2)
        st.write("Backing up content")
        command_output = subprocess.check_output("cp logs.txt logs_backup.txt", shell=True, text=True)
        time.sleep(1)
        st.write("Making Backup Directory...")
        command_output = subprocess.check_output("mkdir backup", shell=True, text=True)
        time.sleep(1)
        status.update(label="Download complete!", state="complete", expanded=False)
          # Prompt for Scenario 1
      errorlog1 = "Failed to move logs_backup.txt to backup/"
      st.error(errorlog1,icon="🚨")
      # Run the command and capture the output
      command = f'yes Y | shell-genie ask "Fix {errorlog1}"'
      command_output = subprocess.check_output(command, shell=True, text=True)

      # Extract the command
      command_start_index = command_output.find("Command: ") + len("Command: ")
      command_end_index = command_output.find("\n", command_start_index)
      extracted_command = command_output[command_start_index:command_end_index].strip()
      try:
        command_output2 = subprocess.check_output(extracted_command, shell=True, text=True)
      except subprocess.CalledProcessError as e:
          command_output2 = e.output
      test = subprocess.check_output("ls backup/", shell=True, text=True)
      if test:
        success_message = "Logs Have been backed up successfully"
        st.session_state.messages.append(AIMessage(content=f"Command ```{extracted_command}```"))
        st.success(success_message, icon="✅")
      else :
        st.error("Scenario Failed !!",icon="🚨")   
      st.session_state.disabled = False
    if button_1:
        st.session_state.disabled = True
        # Prompt for Scenario 1
        st.error("Error response from daemon: conflict: unable to remove container",icon="🚨")
        scen_input = "I encountered the Docker error 'Error response from daemon: conflict: unable to remove container' while trying to remove a container. What command can I use to resolve this issue in the Linux operating system? Your answer should be limited to 100 words and format the command using markdown."
        st.session_state.messages.append(SystemMessage(content=scen_input))

        # Get LLM response
        with st.spinner("NYX is typing ..."):
          answer= get_answer(llm, st.session_state.messages)
        st.session_state.messages.append(AIMessage(content=answer))
        
        st.session_state.disabled = False


        
   
    # Supervise user input
    if user_input := st.chat_input("Input your question!",disabled=st.session_state.disabled):
        st.session_state.messages.append(HumanMessage(content=user_input))
    if user_input and user_input.startswith('/deploy'):
      st.toast('Executing deploy script...', icon='🐳')
      time.sleep(.5)



      try:
          # Run the deploy script using subprocess
          udocker = udocker_init()
          # Run udocker with the provided command and store the output in st.session_state.messages
          udocker_output = udocker("run hello-world")
          formatted_output = f"```\n{udocker_output}\n```"          
          st.session_state.messages.append(AIMessage(content=formatted_output))
          st.toast('Deploy script executed successfully', icon='✅')
          time.sleep(.5)


          

      except subprocess.CalledProcessError as e:
          # Handle error if the command execution fails
          st.error(f"Error executing deploy script: {e}")
          st.text("Error output:")
          st.text(e.stderr)


    elif user_input and user_input.startswith('/docker'):
      # Extracting the input text after '/docker'
      input_text = user_input[len('/docker'):].strip()

      st.toast(f"Executing udocker with input: {input_text}", icon='📜')
      time.sleep(.5)


      try:
          # Run udocker with the extracted input text
          udocker = udocker_init()
          # Run udocker with the provided command and store the output in st.session_state.messages
          udocker_output = udocker(input_text)
          formatted_output = f"```\n{udocker_output}\n```"

          st.session_state.messages.append(AIMessage(content=formatted_output))
          # st.text("Output of udocker:")
          # st.text(udocker_result.stdout)

      except subprocess.CalledProcessError as e:
          # Handle error if udocker command execution fails
          st.error(f"Error executing udocker: {e}")
          st.text("Error output:")
          st.text(e.stderr)
    elif user_input and user_input.startswith('/wish'):
      # Extracting the input text after '/docker'
      input_text = user_input[len('/wish'):].strip()

      

      command = 'yes Y | shell-genie ask "{}"'.format(input_text)
      command_output = subprocess.check_output(command, shell=True, text=True)
      # st.write(command_output)

      # Extract the command
      command_start_index = command_output.find("Command: ") + len("Command: ")
      command_end_index = command_output.find("\n", command_start_index)
      extracted_command = command_output[command_start_index:command_end_index].strip()
      st.toast(f"Executing genie with input: {extracted_command}", icon='📜')
      time.sleep(.5)
      try:
        command_output2 = subprocess.check_output(extracted_command, shell=True, text=True)
      except subprocess.CalledProcessError as e:
          command_output2 = e.output

      if not command_output2.strip():
          st.session_state.messages.append(AIMessage(content=f"Command ```{extracted_command}``` Output:\n```\n Terminal is empty.\n```"))
      else:
          st.session_state.messages.append(AIMessage(content=f"Command ```{extracted_command}``` Output:\n```\n{command_output2}\n```"))


    elif user_input:
        with st.spinner("NYX is typing ..."):
            answer= get_answer(llm, st.session_state.messages)

        st.session_state.messages.append(AIMessage(content=answer))
        

    # Display chat history
    messages = st.session_state.get("messages", [])
    for message in messages:
        if isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)

    
    


# streamlit run app.py
if __name__ == "__main__":
    main()