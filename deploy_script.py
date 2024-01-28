import os

def udocker_init():
    if not os.path.exists("/home/user"):
        os.system("pip install udocker > /dev/null")
        os.system("udocker --allow-root install > /dev/null")
        os.system("useradd -m user > /dev/null")
    print(f'Docker-in-Colab 1.1.0\n')
    print(f'Usage:     udocker("--help")')
    print(f'Examples:  https://github.com/indigo-dc/udocker?tab=readme-ov-file#examples')

    def execute(command: str):
        user_prompt = "\033[1;32muser@pc\033[0m"
        full_command = f"su - user -c 'udocker {command}'"
        print(f"{user_prompt}$ {full_command}")

        # Capture the output of the command and return it as a string
        output = os.popen(full_command).read()
        return output

    return execute

def capture_output(command: str):
    # Capture the output of the command and return it as a string
    return os.popen(command).read()

# Initialize udocker
udocker = udocker_init()

# Example of using udocker to execute commands and capturing output
output1 = udocker("run ubuntu echo 'Hello, Udocker!'")
output2 = udocker("run hello-world")

# Example of using the capture_output function
output3 = capture_output("ls -l")

# Print or use the captured output as needed
print("Output 1:", output1)
print("Output 2:", output2)
print("Output 3:", output3)
