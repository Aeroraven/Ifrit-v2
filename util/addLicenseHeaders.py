import os
str_header = ""

def scan_subdir_recursive(path:str):
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".glsl"):
                file_path = os.path.join(root, file)
                with open(file_path, "r") as f:
                    content = f.read()
                with open(file_path, "w") as f:
                    f.write(str_header + '\n\n' + content)
